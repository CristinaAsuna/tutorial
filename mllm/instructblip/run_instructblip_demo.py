"""CPU smoke test for instruction-conditioned visual query extraction."""
import torch

from edu_core.training import seed_everything
from reference_instructblip import build_toy_instructblip


def main(implementation="reference") -> None:
    if implementation == "practice":
        from practice_instructblip import build_toy_instructblip, training_step
    else:
        from reference_instructblip import build_toy_instructblip
        def training_step(model, optimizer, batch):
            model.train(); optimizer.zero_grad()
            loss=model(**batch)["loss"]; loss.backward(); optimizer.step()
            return loss.detach()
    seed_everything(31)
    model = build_toy_instructblip()
    images = torch.randn(2, 3, 8, 8)
    instruction_ids = torch.tensor([[4, 5, 0], [6, 7, 8]])
    instruction_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    prompt_ids = torch.tensor([[9, 10, 0], [11, 12, 13]])
    prompt_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    answer_ids = torch.tensor([[14, 15], [16, 0]])
    answer_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    model.train()
    assert not model.vision_encoder.training and not model.llm.training
    out = model(images, instruction_ids, prompt_ids, answer_ids,
                instruction_attention_mask=instruction_mask, llm_prompt_attention_mask=prompt_mask,
                answer_attention_mask=answer_mask)
    assert out["visual_prefix"].shape == (2, 4, 32)
    assert torch.isfinite(out["loss"])
    assert torch.all(out["labels"][:, :6] == -100)  # 4 visual query slots + at least 2 prompt slots

    model.eval()
    with torch.no_grad():
        changed_instruction = instruction_ids.clone(); changed_instruction[0, 0] = 20
        query_a = model.encode_instruction_aware_queries(images[:1], instruction_ids[:1], instruction_mask[:1])
        query_b = model.encode_instruction_aware_queries(images[:1], changed_instruction[:1], instruction_mask[:1])
        assert not torch.equal(query_a, query_b), "instruction must change visual queries"
        changed_pad = instruction_ids.clone(); changed_pad[0, 2] = 21
        query_pad = model.encode_instruction_aware_queries(images[:1], changed_pad[:1], instruction_mask[:1])
        assert torch.equal(query_a, query_pad), "masked instruction padding must not affect queries"

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch=dict(pixel_values=images,instruction_ids=instruction_ids,llm_prompt_ids=prompt_ids,answer_ids=answer_ids,
               instruction_attention_mask=instruction_mask,llm_prompt_attention_mask=prompt_mask,answer_attention_mask=answer_mask)
    training_step(model, optimizer, batch)
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    assert all(p.grad is None for p in model.llm.parameters())
    assert any(p.grad is not None for p in model.qformer.parameters())
    assert any(p.grad is not None for p in model.llm_proj.parameters())

    model.eval()
    with torch.no_grad():
        first = model.generate(images, instruction_ids, prompt_ids, instruction_attention_mask=instruction_mask,
                               llm_prompt_attention_mask=prompt_mask, max_new_tokens=3)
        second = model.generate(images, instruction_ids, prompt_ids, instruction_attention_mask=instruction_mask,
                                llm_prompt_attention_mask=prompt_mask, max_new_tokens=3)
    assert torch.equal(first, second) and first.shape == (2, 3)
    print(f"InstructBLIP CPU demo passed; loss={out['loss'].item():.4f}, prefix={tuple(out['visual_prefix'].shape)}")


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--implementation",choices=["reference","practice"],default="reference")
    args=parser.parse_args()
    try: main(args.implementation)
    except NotImplementedError as exc: parser.exit(2,str(exc)+"\n")
