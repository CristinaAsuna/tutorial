"""Reference/practice CPU training + full-vocabulary action generation."""
import argparse
import torch
from reference_openvla import build_toy_openvla


def run_demo(builder=build_toy_openvla):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = builder()
    images = torch.randn(2, 3, 8, 8)
    prompts = [torch.tensor([1, 8, 9]), torch.tensor([1, 10])]
    train_actions = torch.randn(101, 7)
    train_actions[:, -1] = torch.arange(101) % 2
    stats = model.ops.fit_action_stats(train_actions, torch.tensor([True]*6+[False]))
    actions = train_actions[:2]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = model.ops.training_step(model, images, prompts, actions, stats, optimizer)
    assert torch.isfinite(loss)
    for prefix in ("first", "second", "projector", "language"):
        parameters = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in parameters), prefix
        assert any(not torch.equal(before[n], p) for n, p in parameters), prefix
    predicted = model.predict_action(images, prompts, stats)
    repeated = model.predict_action(images, prompts, stats)
    assert predicted.shape == (2, 7) and torch.isfinite(predicted).all()
    torch.testing.assert_close(predicted, repeated, rtol=0, atol=0)
    assert torch.all(predicted[:, :6] >= stats["q01"][:6])
    assert torch.all(predicted[:, :6] <= stats["q99"][:6])
    print(f"OpenVLA CPU: loss={loss.item():.4f}; both vision branches/projector/LM updated; actions={tuple(predicted.shape)}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("reference", "practice"), default="reference")
    args = parser.parse_args()
    if args.implementation == "practice":
        from practice_openvla import build_toy_openvla as builder
    else:
        builder = build_toy_openvla
    try:
        run_demo(builder)
    except NotImplementedError as error:
        print(f"未完成练习：{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
