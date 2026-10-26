"""Use the cached SmolVLM language decoder; visual branches remain toy mechanisms."""
import argparse
import torch
from edu_core.backbones import load_backbone
from backbone import build_with_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--implementation", choices=("reference", "practice"), default="reference")
    args = parser.parse_args()
    torch.manual_seed(23); torch.set_num_threads(2)
    backend = load_backbone("smolvlm", model_path=args.model_path, frozen=True)
    model = build_with_backbone(backend, args.implementation)
    try:
        assert not set(model.action_vocabulary.ids.tolist()) & set(backend.tokenizer.all_special_ids)
        original = torch.arange(model.vocab_size-256, model.vocab_size)
        mapped = model.action_vocabulary.to_model(original, model.vocab_size)
        torch.testing.assert_close(model.action_vocabulary.to_canonical(mapped, model.vocab_size), original)
        images = torch.rand(1, 3, 8, 8)
        prompts = [torch.tensor([1, 8, 9])]
        stats = model.ops.fit_action_stats(torch.randn(101, 7), torch.ones(7, dtype=torch.bool))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        before = model.projector[0].weight.detach().clone()
        loss = model.ops.training_step(model, images, prompts, torch.rand(1, 7), stats, optimizer)
        assert torch.isfinite(loss)
        assert not torch.equal(before, model.projector[0].weight)
        assert all(p.grad is None for p in model.language.parameters())
        assert model.first.patch.weight.grad.abs().sum() > 0 and model.second.patch.weight.grad.abs().sum() > 0
        actions = model.predict_action(images, prompts, stats)
        assert actions.shape == (1, 7) and torch.isfinite(actions).all()
        print(f"OpenVLA component smoke PASS: loss={loss.item():.4f}; safe action IDs, frozen pretrained LM, two toy vision branches")
    except NotImplementedError as error:
        print(f"未完成练习：{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
