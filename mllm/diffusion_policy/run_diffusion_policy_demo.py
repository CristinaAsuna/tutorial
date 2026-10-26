"""CPU loss/update/EMA/DDPM/DDIM/receding-horizon checks for both denoisers."""
import argparse
import torch
from reference_diffusion_policy import build_toy_diffusion_policy, make_ema


def run_demo(builder=build_toy_diffusion_policy, backbone="cnn"):
    torch.manual_seed(8)
    torch.set_num_threads(1)
    model = builder(backbone)
    episode_states = torch.arange(48).float().reshape(12, 4)/48
    episode_images = torch.randn(12, 3, 8, 8)
    episode_actions = torch.stack((torch.linspace(-2, 2, 12), torch.linspace(2, -2, 12)), dim=-1)
    stats = model.ops.fit_bounds(episode_actions)
    model.action_low.copy_(stats["low"])
    model.action_high.copy_(stats["high"])
    windows = [model.ops.trajectory_window(episode_states, episode_actions, t) for t in (0, 5)]
    image_windows = [model.ops.trajectory_window(episode_images, episode_actions, t)[0] for t in (0, 5)]
    batch = {"images": torch.stack(image_windows), "states": torch.stack([x[0] for x in windows]),
             "actions": torch.stack([x[1] for x in windows])}
    ema = make_ema(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema_before = {n: p.detach().clone() for n, p in ema.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    noise, times = torch.randn(2, 8, 2), torch.tensor([2, 9])
    loss = model.ops.training_step(model, ema, batch, optimizer, noise, times)
    assert torch.isfinite(loss)
    for prefix in ("encoder", "denoiser"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix))
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(prefix))
    for n, parameter in ema.named_parameters():
        torch.testing.assert_close(parameter, .95*ema_before[n]+.05*dict(model.named_parameters())[n])
        assert parameter.grad is None and not parameter.requires_grad
    initial, noises = torch.randn(2, 8, 2), torch.randn(16, 2, 8, 2)
    for sampler in ("ddpm", "ddim"):
        calls = []
        handle = ema.encoder.register_forward_hook(lambda *args: calls.append(1))
        result = ema.predict_action(batch["images"], batch["states"], initial, noises, sampler)
        handle.remove()
        assert len(calls) == 1, "observation encoder belongs outside denoising loop"
        repeated = ema.predict_action(batch["images"], batch["states"], initial, noises, sampler)
        torch.testing.assert_close(result["action_pred"], repeated["action_pred"], rtol=0, atol=0)
        assert result["action"].shape == (2, 3, 2)
        assert torch.isfinite(result["action_pred"]).all()
        torch.testing.assert_close(result["action"], result["action_pred"][:, 1:4])
    observations, executed = [], []
    def observe():
        observations.append(len(executed))
        # Minimal control adapter: next frame/state responds to the last executed chunk.
        offset = 0. if not executed else executed[-1].mean().item()
        return batch["images"]+offset, batch["states"]+offset
    def execute(chunk):
        executed.append(chunk.clone())
    chunks = ema.ops.receding_rollout(ema, observe, execute, 2, [initial]*2, [noises]*2)
    assert observations == [0, 1] and len(chunks) == 2
    print(f"Diffusion Policy {backbone} CPU: loss={loss.item():.4f}; update/EMA/DDPM/DDIM/replanning passed")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("reference", "practice"), default="reference")
    parser.add_argument("--backbone", choices=("cnn", "transformer", "all"), default="cnn")
    args = parser.parse_args()
    if args.implementation == "practice":
        from practice_diffusion_policy import build_toy_diffusion_policy as builder
    else:
        builder = build_toy_diffusion_policy
    try:
        for backbone in (("cnn", "transformer") if args.backbone == "all" else (args.backbone,)):
            run_demo(builder, backbone)
    except NotImplementedError as error:
        print(f"未完成练习：{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
