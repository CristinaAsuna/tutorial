"""CPU checks for Flamingo's interleaved-media teaching implementation."""
import torch

from conversation import build_interleaved_sft_example, right_pad_examples
from reference_flamingo import IMAGE_TOKEN_INDEX, build_toy_flamingo
from edu_core.training import seed_everything


def main() -> None:
    seed_everything(23)
    model = build_toy_flamingo(vocab_size=48)
    examples = [
        build_interleaved_sft_example([("text", [6, 7]), ("image", []), ("text", [8]), ("image", []), ("text", [9])], [10, 11]),
        build_interleaved_sft_example([("text", [12]), ("image", []), ("text", [13]), ("image", [])], [14]),
    ]
    input_ids, labels, attention_mask = right_pad_examples(examples)
    images = torch.randn(2, 2, 3, 8, 8)

    model.train()
    assert not model.vision_encoder.training
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    assert all(not p.requires_grad for p in model.decoder.blocks.parameters())
    out = model(input_ids, images, attention_mask=attention_mask, labels=labels)
    assert out["visual_memory"].shape == (2, 2, 3, 32)
    assert torch.isfinite(out["loss"])
    assert torch.all(out["labels"][input_ids == IMAGE_TOKEN_INDEX] == -100)

    # Before the first image, no visual key is visible.  After the first image,
    # only its three resampled latents are visible; after the second, all six.
    mask = out["media_attention_mask"][0]
    assert not mask[1].any() and mask[3].sum() == 3 and mask[5].sum() == 6
    with torch.no_grad():
        for connector in model.decoder.gated_cross_attn.values():
            connector.attn_gate.fill_(1.0)
        changed_future = images.clone(); changed_future[0, 1].add_(1000)
        a = model(input_ids, images, attention_mask=attention_mask)["logits"][0, 3]
        b = model(input_ids, changed_future, attention_mask=attention_mask)["logits"][0, 3]
    assert torch.equal(a, b), "a token must not read a future image"
    with torch.no_grad():
        for connector in model.decoder.gated_cross_attn.values():
            connector.attn_gate.zero_()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    optimizer.zero_grad(); out = model(input_ids, images, attention_mask=attention_mask, labels=labels); out["loss"].backward()
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    assert all(p.grad is None for p in model.decoder.blocks.parameters())
    assert any(p.grad is not None for p in model.decoder.gated_cross_attn.parameters())
    optimizer.step()
    # Once the zero gates have moved, the resampler receives a connector gradient.
    optimizer.zero_grad(); model(input_ids, images, attention_mask=attention_mask, labels=labels)["loss"].backward()
    assert any(p.grad is not None for p in model.resampler.parameters())

    model.eval()
    with torch.no_grad():
        first = model.generate(input_ids, images, attention_mask=attention_mask, max_new_tokens=3)
        second = model.generate(input_ids, images, attention_mask=attention_mask, max_new_tokens=3)
    assert torch.equal(first, second) and first.shape == (2, 3)
    print(f"Flamingo CPU demo passed; loss={out['loss'].item():.4f}, visual memory={tuple(out['visual_memory'].shape)}")


if __name__ == "__main__":
    main()
