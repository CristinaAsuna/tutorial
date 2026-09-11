"""End-to-end CPU checks for the complete reference solution only."""
import torch
from edu_core.training import seed_everything

from conversation import build_sft_example, right_pad_examples
from reference_llava import IMAGE_TOKEN_INDEX, IGNORE_INDEX, build_toy_llava


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main():
    seed_everything(7)
    model = build_toy_llava(vocab_size=48)
    first = build_sft_example([3, 4], [5, 6], [7, 8])
    second = build_sft_example([3], [9], [10])
    input_ids, labels, attention_mask = right_pad_examples([first, second])
    images = torch.randn(2, 3, 8, 8)

    # Vision output has CLS + four patches.  LLaVA explicitly drops CLS.
    raw_features = model.vision_encoder(images)
    projected = model.encode_images(images)
    assert raw_features.shape == (2, 5, 24)
    assert projected.shape == (2, 4, 32)
    assert not model.vision_encoder.training

    out = model(input_ids, images, attention_mask, labels)
    assert out["inputs_embeds"].shape[1] == input_ids.shape[1] + 3
    for row in range(input_ids.size(0)):
        image_at = (input_ids[row] == IMAGE_TOKEN_INDEX).nonzero().item()
        assert torch.all(out["labels"][row, image_at:image_at + 4] == IGNORE_INDEX)
    assert torch.isfinite(out["loss"])

    # Padding token IDs are ignored by both attention and labels, hence loss is invariant.
    altered = input_ids.clone()
    altered[~attention_mask] = 11
    same = model(altered, images, attention_mask, labels)["loss"]
    assert torch.allclose(out["loss"], same), "right-padding must not affect SFT loss"

    # Contract failures are caught before a negative sentinel reaches Embedding.
    no_image = input_ids.clone()
    no_image[no_image == IMAGE_TOKEN_INDEX] = 3
    two_images = input_ids.clone()
    two_images[0, 0] = IMAGE_TOKEN_INDEX
    bad_negative = input_ids.clone()
    bad_negative[0, 0] = -7
    expect_value_error(lambda: model(no_image, images, attention_mask, labels))
    expect_value_error(lambda: model(two_images, images, attention_mask, labels))
    expect_value_error(lambda: model(bad_negative, images, attention_mask, labels))
    expect_value_error(lambda: model(input_ids, images[:1], attention_mask, labels))
    expect_value_error(lambda: model(input_ids, images, attention_mask[:, :-1], labels))
    bad_mask = attention_mask.clone().long()
    bad_mask[0, 0] = 2
    expect_value_error(lambda: model(input_ids, images, bad_mask, labels))

    # Stage 1: only projector receives gradients and changes.
    model.set_training_stage("pretrain_projector")
    model.train()
    before = model.projector.layers[0].weight.detach().clone()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    optimizer.zero_grad()
    stage1_loss = model(input_ids, images, attention_mask, labels)["loss"]
    stage1_loss.backward()
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    assert all(p.grad is None for p in model.llm.parameters())
    assert any(p.grad is not None for p in model.projector.parameters())
    optimizer.step()
    assert not torch.equal(before, model.projector.layers[0].weight)

    # Stage 2: frozen vision stays eval/no-grad; projector and LLM learn.
    model.set_training_stage("instruction_tuning")
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05)
    optimizer.zero_grad()
    stage2_loss = model(input_ids, images, attention_mask, labels)["loss"]
    stage2_loss.backward()
    assert not model.vision_encoder.training
    assert any(p.grad is not None for p in model.projector.parameters())
    assert any(p.grad is not None for p in model.llm.parameters())
    optimizer.step()

    model.eval()
    g1 = model.generate(input_ids, images, attention_mask, max_new_tokens=3)
    g2 = model.generate(input_ids, images, attention_mask, max_new_tokens=3)
    assert torch.equal(g1, g2) and g1.shape == (2, 3)
    print("LLaVA toy demo passed: packing, masks, two stages, and greedy generation.")


if __name__ == "__main__":
    main()
