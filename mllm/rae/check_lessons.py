"""逐关独立语义验收，默认practice。仅选中关使用学生函数，其余为已提供外围。"""
import argparse
import torch
from types import SimpleNamespace
from . import reference_rae as ref
from .lesson_registry import LESSONS
from .scaffolding import RAECourse, toy_images


def check(number, implementation='practice'):
    torch.manual_seed(17)
    ops = ref.reference_ops()
    if implementation == 'practice':
        from .practice_rae import practice_ops
        student = practice_ops()
        for name in LESSONS[number-1][1]:
            setattr(ops, name, getattr(student, name))
    model = RAECourse(ops)
    images = toy_images()
    if number == 1:
        x = torch.tensor([[[99., 99.], [88., 88.], [1., 3.], [2., 6.]]])
        got = ops.normalize_tokens(x, 2)
        torch.testing.assert_close(got, torch.tensor([[[-1., 1.], [-1., 1.]]]), atol=1e-4, rtol=1e-4)
        tokens = torch.arange(12).reshape(1, 6, 2)
        grid = ops.tokens_to_grid(tokens, (2, 3))
        assert grid[0, 0].tolist() == [[0, 2, 4], [6, 8, 10]]
        torch.testing.assert_close(ops.grid_to_tokens(grid), tokens)
    elif number == 2:
        z = model.rae.encode(images)
        got = ops.decode_tokens(model.rae, z)
        torch.testing.assert_close(got, ref.decode_tokens(model.rae, z))
        assert got.shape == images.shape
        got.sum().backward()
        assert model.rae.decoder_out.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.rae.encoder.parameters())
    elif number == 3:
        z = torch.zeros(2, 3, 4)
        got = ops.augment_latent(z, .2, torch.tensor([-2., 1.]), torch.ones_like(z))
        torch.testing.assert_close(got[0], torch.full((3, 4), .4))
        torch.testing.assert_close(got[1], torch.full((3, 4), .2))
        torch.testing.assert_close(ops.augment_latent(z, 0), z)
    elif number == 4:
        p = torch.tensor([1., 3.], requires_grad=True)
        loss = ops.reconstruction_loss(p, torch.tensor([0., 1.]))
        torch.testing.assert_close(loss, torch.tensor(1.5))
        loss.backward(); torch.testing.assert_close(p.grad, torch.tensor([.5, .5]))
        model.set_stage(1)
        frozen = {k: v.clone() for k,v in model.rae.encoder.state_dict().items()}
        before = model.rae.decoder_out.weight.clone()
        opt = torch.optim.Adam(model.rae.parameters(), lr=.002)
        result = ops.stage1_step(model, images, opt)
        assert torch.isfinite(result) and not result.requires_grad
        assert not torch.equal(before, model.rae.decoder_out.weight)
        assert all(torch.equal(v, model.rae.encoder.state_dict()[k]) for k,v in frozen.items())
        assert not model.rae.encoder.training
    elif number == 5:
        times = torch.tensor([0., .5, 1.])
        torch.testing.assert_close(ops.shift_time(times, 16, 4), torch.tensor([0., 2/3, 1.]))
        torch.testing.assert_close(ops.shift_time(times, 1024), torch.tensor([0., 1/3, 1.]))
        torch.testing.assert_close(ops.shift_time(times, 4, 4), times)
        x, u = ops.flow_pair(torch.full((1, 2, 3), 2.), torch.tensor([.25]), torch.zeros(1, 2, 3))
        torch.testing.assert_close(x, torch.full_like(x, 1.5)); torch.testing.assert_close(u, torch.full_like(u, -2.))
        pred = torch.tensor([1., 3.], requires_grad=True)
        target = torch.tensor([0., 1.], requires_grad=True)
        loss = ops.velocity_loss(pred, target)
        torch.testing.assert_close(loss, torch.tensor(2.5))
        loss.backward(); assert target.grad is None
        torch.testing.assert_close(pred.grad, torch.tensor([1., 2.]))
    elif number == 6:
        # Activate zero-init layers so tests observe both routes immediately.
        for p in model.generator.parameters():
            if p.requires_grad:
                with torch.no_grad(): p.add_(torch.randn_like(p)*.03)
        z = torch.randn(2,16,64, requires_grad=True); t = torch.tensor([.2,.7])
        got = ops.ddt_forward(model.generator, z, t)
        torch.testing.assert_close(got, ref.ddt_forward(model.generator, z, t))
        got.square().mean().backward()
        assert z.grad.abs().sum() > 0
        for module in (model.generator.noisy_embed, model.generator.base_embed, model.generator.condition_projector):
            assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    elif number == 7:
        model.set_stage(2)
        frozen = {k:v.clone() for k,v in model.rae.state_dict().items()}
        before = {k:v.clone() for k,v in model.generator.state_dict().items()}
        opt = torch.optim.Adam(model.generator.parameters(), lr=.002)
        for _ in range(5):
            old = {k:v.clone() for k,v in model.ema.state_dict().items()}
            result = ops.stage2_step(model, images, opt, momentum=.75)
            assert torch.isfinite(result) and not result.requires_grad
            for key,p in model.generator.named_parameters():
                torch.testing.assert_close(model.ema.state_dict()[key], old[key]*.75+p.detach()*.25)
        for key in ('base_embed.weight', 'noisy_embed.weight', 'condition_projector.weight', 'out.weight'):
            assert not torch.equal(before[key], model.generator.state_dict()[key]), key
        assert all(torch.equal(v, model.rae.state_dict()[k]) for k,v in frozen.items())
        assert all(p.grad is None for p in model.rae.parameters())
        model.train()
        assert not model.rae.training and not model.ema.training and model.generator.training
    else:
        class Constant:
            def __init__(self): self.ops=ops; self.times=[]
            def __call__(self,x,t): self.times.append(t[0].item()); return torch.full_like(x,-2.)
        constant=Constant(); initial=torch.zeros(1,16,64)
        got=ops.euler_sample(constant, initial, 2)
        torch.testing.assert_close(got, torch.full_like(initial,2.))
        assert constant.times == [1., 1/3] or abs(constant.times[1]-1/3)<1e-6
        assert not initial.any()
        for steps in (0,-1):
            try: ops.euler_sample(constant,initial,steps)
            except ValueError: pass
            else: raise AssertionError('steps<=0 must fail')
        a=ops.sample_images(model,initial,2); b=ops.sample_images(model,initial,2)
        torch.testing.assert_close(a,b)
        assert a.shape==(1,3,16,16) and torch.isfinite(a).all() and not a.requires_grad
    print(f'RAE lesson {number} {implementation}: passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', type=int, choices=range(1,9))
    parser.add_argument('--implementation', choices=('reference','practice'), default='practice')
    args=parser.parse_args(); failed=False
    for number in ([args.lesson] if args.lesson else range(1,9)):
        try: check(number,args.implementation)
        except NotImplementedError as exc:
            print(f'RAE lesson {number}: {exc}'); failed=True
    return 2 if failed else 0

if __name__=='__main__':
    raise SystemExit(main())
