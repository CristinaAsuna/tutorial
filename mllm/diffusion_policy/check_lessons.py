"""Hand-computable local checks; selected lesson is isolated using supplied peripheral ops."""
import argparse
import importlib
from types import SimpleNamespace
import torch
from torch import nn
import reference_diffusion_policy as reference
from lesson_registry import GROUPS


def selected_ops(number, implementation):
    ops = reference.reference_ops()
    if implementation == "practice":
        module, names = GROUPS[number-1]
        lesson = importlib.import_module(module)
        for name in names:
            setattr(ops, name, getattr(lesson, name))
    return ops


def check(number, ops, backbone):
    torch.manual_seed(12)
    if number == 1:
        obs, actions = torch.arange(5), torch.arange(5)+10
        window, chunk = ops.trajectory_window(obs, actions, 0, 2, 4)
        assert window.tolist() == [0, 0] and chunk.tolist() == [10, 10, 11, 12]
        window, chunk = ops.trajectory_window(obs, actions, 4, 2, 4)
        assert window.tolist() == [3, 4] and chunk.tolist() == [13, 14, 14, 14]
        stats = ops.fit_bounds(torch.tensor([[0., 4.], [2., 8.]]))
        torch.testing.assert_close(stats["low"], torch.tensor([0., 4.]))
        torch.testing.assert_close(stats["high"], torch.tensor([2., 8.]))
        x = torch.tensor([[0., 4.], [1., 6.], [2., 8.]])
        expected = torch.tensor([[-1., -1.], [0., 0.], [1., 1.]])
        torch.testing.assert_close(ops.normalize(x, stats), expected)
        torch.testing.assert_close(ops.unnormalize(expected, stats), x)
    elif number == 2:
        beta, a = ops.cosine_schedule(16)
        # Independent scalar oracle, not the reference implementation under test.
        import math
        f = lambda t: math.cos((t+.008)/1.008*math.pi/2)**2
        expected = torch.tensor([min(1-f((i+1)/16)/f(i/16), .999) for i in range(16)])
        torch.testing.assert_close(beta, expected)
        torch.testing.assert_close(a, (1-beta).cumprod(0))
        assert (a[1:] < a[:-1]).all()
        clean = torch.full((2, 3, 1), 2.)
        noisy = ops.add_noise(clean, torch.zeros_like(clean), torch.tensor([0, 1]), torch.tensor([.25, 1.]))
        torch.testing.assert_close(noisy[:, 0, 0], torch.tensor([1., 2.]))
    elif number == 3:
        # Encoder returns numbered content: verify B/To order, state concatenation and gradients.
        class Encoder(nn.Module):
            def forward(self, x):
                return x.mean((2, 3))
        images = torch.arange(4.).reshape(2, 2, 1, 1, 1).expand(-1, -1, 3, 2, 2).clone().requires_grad_()
        state = torch.tensor([[[10.], [11.]], [[12.], [13.]]])
        out = ops.encode_observations(Encoder(), images, state)
        torch.testing.assert_close(out[..., 0], torch.tensor([[0., 1.], [2., 3.]]))
        torch.testing.assert_close(out[..., -1:], state)
        out.sum().backward()
        assert images.grad.abs().sum() > 0
    elif number == 4:
        embed = ops.timestep_embedding(torch.tensor([0, 1]), 16)
        torch.testing.assert_close(embed[0, :8], torch.zeros(8))
        torch.testing.assert_close(embed[0, 8:], torch.ones(8))
        torch.testing.assert_close(embed[1, 0], torch.tensor(1.).sin())
        torch.testing.assert_close(embed[1, 7], torch.tensor(.0001).sin())
        torch.testing.assert_close(ops.film_modulate(torch.ones(1, 2, 3), torch.tensor([[2., 2., 3., 3.]])), torch.full((1, 2, 3), 5.))
        block = reference.ConditionalResidual(2, 4, ops)
        block.conv1 = nn.Identity()
        block.conv2 = nn.Identity()
        block.skip = nn.Identity()
        block.condition = nn.Identity()
        torch.testing.assert_close(ops.residual_forward(block, torch.ones(1, 2, 3), torch.tensor([[2., 2., 3., 3.]])), torch.full((1, 2, 3), 6.))
        model = reference.ConditionalUNet1D(ops)
        x, t, obs = torch.randn(2, 8, 2), torch.tensor([1, 4]), torch.randn(2, 2, 16)
        cond = torch.randn(2, 48)
        down, skip = ops.unet_down(model, x.transpose(1, 2), cond)
        assert down.shape == (2, 32, 4) and skip.shape == (2, 16, 8)
        assert ops.unet_up(model, down, skip, cond).shape == (2, 16, 8)
        prediction = ops.unet_forward(model, x, t, obs)
        assert prediction.shape == x.shape
        assert not torch.allclose(ops.unet_forward(model, x, t, obs+1), prediction)
        assert not torch.allclose(ops.unet_forward(model, x, t+1, obs), prediction)
        prediction.square().mean().backward()
        assert model.down_block.conv1[0].weight.grad.abs().sum() > 0
    elif number == 5:
        torch.testing.assert_close(ops.noise_loss(torch.tensor([1., 3.]), torch.tensor([0., 1.])), torch.tensor(2.5))
        from run_diffusion_policy_demo import run_demo
        run_demo(lambda b: reference.DiffusionPolicy(b, ops), backbone)
    elif number == 6:
        # At final step beta=.5,a=.5, eps=0, x=.2 -> sqrt(2)*.2; noise ignored.
        x, prediction = torch.tensor([[[.2]]]), torch.zeros(1, 1, 1)
        betas, cumulative = torch.tensor([.5, .5]), torch.tensor([.5, .25])
        expected = torch.tensor([[[.2 * 2**.5]]])
        torch.testing.assert_close(ops.ddpm_step(x, prediction, 0, betas, cumulative, torch.ones_like(x)*100), expected)
        torch.testing.assert_close(ops.ddim_step(x, prediction, 0, -1, cumulative), expected)
        # t=1: posterior mean coefficients sqrt(.5)*.5/.75 and sqrt(.5)*.5/.75.
        mean = (2**-.5)*.5/.75*(.4+.2)
        variance = .5*.5/.75
        torch.testing.assert_close(ops.ddpm_step(x, prediction, 1, betas, cumulative, torch.ones_like(x)), torch.tensor([[[mean+variance**.5]]]))
        model = reference.DiffusionPolicy(backbone, ops)
        images, states = torch.randn(2, 2, 3, 8, 8), torch.randn(2, 2, 4)
        initial, noises = torch.randn(2, 8, 2), torch.randn(16, 2, 8, 2)
        for sampler in ("ddpm", "ddim"):
            calls = []
            handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
            result = ops.sample_actions(model, images, states, initial, noises, sampler)
            handle.remove()
            assert len(calls) == 1 and result.shape == initial.shape
            other = ops.sample_actions(model, images, states, initial, noises, sampler)
            torch.testing.assert_close(result, other, rtol=0, atol=0)
    elif number == 7:
        numbered = torch.arange(8.).reshape(1, 8, 1)
        assert ops.execution_chunk(numbered).flatten().tolist() == [1., 2., 3.]
        events = []
        def observe():
            events.append("observe")
            return torch.tensor(len(events)), None
        def execute(chunk):
            events.append("execute")
        def predict(images, states, initial, noises):
            events.append(f"predict:{initial}")
            return {"action": numbered[:, 1:4]}
        chunks = ops.receding_rollout(SimpleNamespace(predict_action=predict), observe, execute, 2, [3, 4], [None]*2)
        assert events == ["observe", "predict:3", "execute", "observe", "predict:4", "execute"]
        assert len(chunks) == 2
    elif number == 8:
        action_mask, memory_mask = ops.transformer_masks(3, 2)
        assert action_mask.tolist() == [[True, False, False], [True, True, False], [True, True, True]]
        assert memory_mask.tolist() == [[True, True, False], [True, True, True], [True, True, True]]
        model = reference.TransformerDenoiser(ops)
        x, t, obs = torch.randn(2, 8, 2), torch.tensor([0, 5]), torch.randn(2, 2, 16)
        action, memory = ops.transformer_tokens(model, x, t, obs)
        assert action.shape == (2, 8, 32) and memory.shape == (2, 3, 32)
        prediction = ops.transformer_forward(model, x, t, obs)
        assert not torch.allclose(ops.transformer_forward(model, x, t+1, obs), prediction)
        changed = x.clone()
        changed[:, 4:] += 50
        torch.testing.assert_close(ops.transformer_forward(model, changed, t, obs)[:, :4], prediction[:, :4])
        changed_obs = obs.clone()
        changed_obs[:, 1] += 50
        torch.testing.assert_close(ops.transformer_forward(model, x, t, changed_obs)[:, 0], prediction[:, 0])
        prediction.square().mean().backward()
        assert model.obs_in.weight.grad.abs().sum() > 0
    else:
        raise ValueError(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", type=int, choices=range(1, 9))
    parser.add_argument("--implementation", choices=("reference", "practice"), default="practice")
    parser.add_argument("--backbone", choices=("cnn", "transformer", "all"), default="cnn")
    args = parser.parse_args()
    torch.set_num_threads(1)
    failed = False
    for backbone in (("cnn", "transformer") if args.backbone == "all" else (args.backbone,)):
        for number in ([args.lesson] if args.lesson else range(1, 9)):
            try:
                check(number, selected_ops(number, args.implementation), backbone)
                print(f"lesson {number}: PASS ({args.implementation}/{backbone})")
            except NotImplementedError as error:
                print(f"lesson {number}: 未完成 {error}")
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
