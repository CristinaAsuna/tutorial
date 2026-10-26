"""CPU pi0 training and cached Euler inference; optional local pretrained backbone."""
import argparse
import torch
from reference_pi0 import build_toy_pi0


def make_batch(model, batch_size=2, cameras=2):
    device = next(model.parameters()).device
    episode = torch.linspace(-1.8, 1.8, 12, device=device)[:, None].expand(-1, 7)
    actions = torch.stack([model.ops.action_chunk(episode, now, model.horizon) for now in range(batch_size)])
    ids = torch.tensor([[1, 8, 9], [1, 10, 0]], device=device)[:batch_size]
    return {"images": torch.rand(batch_size, cameras, 3, 8, 8, device=device),
            "camera_valid": torch.ones(batch_size, cameras, dtype=torch.bool, device=device),
            "text_ids": ids, "text_valid": ids != 0,
            "state": torch.rand(batch_size, 7, device=device), "actions": actions}


def run_demo(builder=build_toy_pi0):
    torch.manual_seed(19)
    torch.set_num_threads(2)
    model = builder()
    pretrained = not any(p.requires_grad for p in model.backbone.parameters())
    batch = make_batch(model, batch_size=1 if pretrained else 2, cameras=1 if pretrained else 2)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
    noise = torch.randn_like(batch["actions"])
    loss = model.ops.training_step(model, batch, optimizer, noise=noise)
    assert torch.isfinite(loss)
    for prefix in ("state_proj", "action_in", "action_time", "expert_layers", "action_out"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix)), prefix
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(prefix) and n in before), prefix
    if pretrained:
        assert all(p.grad is None for p in model.backbone.parameters())
        assert not model.backbone.training
    else:
        for prefix in ("backbone.vision", "backbone.connector", "backbone.layers", "backbone.embedding"):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix)), prefix
    calls = []
    vision = model.backbone.vision if hasattr(model.backbone, "vision") else model.backbone.model.model.vision_model
    handle = vision.register_forward_hook(lambda *args: calls.append(1))
    predicted = model.predict_action(batch, noise)
    handle.remove()
    assert len(calls) == 1, "vision must run once per planning round"
    torch.testing.assert_close(predicted, model.predict_action(batch, noise), rtol=0, atol=0)
    assert predicted.shape == batch["actions"].shape and torch.isfinite(predicted).all()
    events = []
    def observe():
        events.append("observe")
        return batch
    def execute(chunk):
        assert chunk.shape[1] == 3
        events.append("execute")
    model.ops.receding_control(model, observe, execute, [noise]*2)
    assert events == ["observe", "execute", "observe", "execute"]
    print(f"pi0 CPU: loss={loss.item():.4f}; gradients/update/Euler/cache/replanning passed; frozen_backbone={pretrained}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("reference", "practice"), default="reference")
    parser.add_argument("--backbone", choices=("tiny", "smolvlm"), default="tiny")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    if args.implementation == "practice":
        from practice_pi0 import practice_ops
    else:
        from reference_pi0 import reference_ops as practice_ops
    from scaffolding import Pi0Policy
    from edu_core.backbones import load_backbone
    def builder():
        backbone = load_backbone(args.backbone, model_path=args.model_path, frozen=args.backbone == "smolvlm")
        return Pi0Policy(backbone=backbone, ops=practice_ops())
    try:
        run_demo(builder)
    except NotImplementedError as error:
        print(f"未完成练习：{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
