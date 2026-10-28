"""Semantic checks; one selected student lesson with reference surroundings."""
import argparse
import torch
from torch import nn
from . import reference_repa as ref
from .scaffolding import RepaConfig, toy_batch


def selected_ops(lesson, implementation):
    ops = ref.reference_ops()
    if implementation == 'practice':
        from .practice_repa import practice_ops, LESSON_FUNCTIONS
        student = practice_ops()
        for name in LESSON_FUNCTIONS[lesson]:
            setattr(ops,name,getattr(student,name))
    return ops


def check_lesson(lesson, implementation='reference', *, ops=None):
    torch.manual_seed(43)
    ops = ops or selected_ops(lesson,implementation)
    system = ref.build_toy_repa(ops=ops)
    images,labels = toy_batch()
    if lesson == 1:
        source = images.clone().requires_grad_()
        clean,targets = ops.clean_targets(system,source)
        torch.testing.assert_close(clean,system.codec.encode(images))
        torch.testing.assert_close(targets,system.teacher(images))
        assert not clean.requires_grad and not targets.requires_grad
        assert clean.shape == (4,4,8,8) and targets.shape == (4,16,24)
        changed = images.clone(); changed[:,:,4:8,8:12] += 3
        _,other = ops.clean_targets(system,changed)
        difference = (other-targets).abs().sum(-1)>1e-6
        assert difference[:,6].all() and difference.sum() == 4, 'width-first clean patch routing'
        system.train()
        assert not system.teacher.training and not system.codec.training
    elif lesson == 2:
        clean = torch.full((2,4,2,2),2.)
        noise = torch.zeros_like(clean)
        xt,target = ops.flow_pair(clean,noise,torch.tensor([.25,1.]))
        torch.testing.assert_close(xt[0],torch.full_like(clean[0],1.5))
        torch.testing.assert_close(xt[1],noise[1])
        torch.testing.assert_close(target,noise-clean)
        torch.testing.assert_close(ops.flow_pair(clean,noise,torch.zeros(2))[0],clean)
        try: ops.flow_pair(clean,noise,torch.tensor([-1.,0.]))
        except ValueError: pass
        else: raise AssertionError('invalid time accepted')
    elif lesson == 3:
        model = system.denoiser
        block = model.blocks[0]
        x,c = torch.randn(2,16,32),torch.randn(2,32)
        torch.testing.assert_close(ops.conditioned_block(block,x,c),x)
        # Nonzero modulation makes ignoring condition/gates detectably wrong.
        nn.init.normal_(block.modulation[-1].weight,std=.1)
        torch.testing.assert_close(ops.conditioned_block(block,x,c),ref.conditioned_block(block,x,c))
        nn.init.normal_(model.blocks[1].modulation[-1].weight,std=.1)
        latent = torch.randn(2,4,8,8); time=torch.tensor([.2,.8]); y=torch.tensor([0,1])
        velocity,projected = model(latent,time,y)
        expected=[]
        # Capture projector INPUT: it must be after block1, not block2 or output.
        h = model.projector.register_forward_pre_hook(lambda module,args:expected.append(args[0]))
        model(latent,time,y); h.remove()
        from edu_core.vision import patchify
        tokens=model.patch(patchify(latent,2))+model.positions
        condition=model.time_embed(time[:,None])+model.label_embed(y)
        torch.testing.assert_close(expected[0],ops.conditioned_block(model.blocks[0],tokens,condition))
        torch.testing.assert_close(velocity,torch.zeros_like(velocity))
        assert projected.shape==(2,16,24)
        def forbidden_projection(*args): raise AssertionError('project=False still calls projector')
        hook=model.projector.register_forward_pre_hook(forbidden_projection)
        _,off=model(latent,time,y,project=False)
        hook.remove()
        assert off is None
    elif lesson == 4:
        a=torch.tensor([[[1.,0.],[0.,1.]]],requires_grad=True)
        target=a.detach().clone().requires_grad_()
        torch.testing.assert_close(ops.alignment_loss(a,target),torch.tensor(-1.))
        torch.testing.assert_close(ops.alignment_loss(a,-target),torch.tensor(1.))
        torch.testing.assert_close(ops.alignment_loss(a,target.flip(1)),torch.tensor(0.))
        # A non-stationary cosine is needed to test meaningful student gradients.
        b=torch.tensor([[[1.,1.],[1.,1.]]],requires_grad=True)
        loss=ops.alignment_loss(b,target); loss.backward()
        assert b.grad is not None and b.grad.abs().sum()>0 and target.grad is None
        prediction=torch.tensor([2.,2.],requires_grad=True)
        result=ops.joint_objective(prediction,torch.zeros(2),torch.tensor(-1.),.5)
        torch.testing.assert_close(result['loss'],torch.tensor(3.5))
        torch.testing.assert_close(result['velocity_loss'],torch.tensor(4.))
        result['loss'].backward(); torch.testing.assert_close(prediction.grad,torch.tensor([2.,2.]))
        torch.testing.assert_close(ops.joint_objective(prediction,torch.zeros(2),torch.tensor(-1.),0)['loss'],torch.tensor(4.))
    elif lesson == 5:
        ema,student = nn.Linear(1,1,bias=False),nn.Linear(1,1,bias=False)
        with torch.no_grad(): ema.weight.fill_(2); student.weight.fill_(6)
        ops.ema_update(ema,student,.75)
        torch.testing.assert_close(ema.weight,torch.tensor([[3.]]))
        exercise_training(system,steps=4)
    elif lesson == 6:
        class ConstantVelocity:
            def __call__(self,x,t,y,*,project):
                assert not project
                return torch.full_like(x,-2),None
        initial=torch.zeros(2,4,8,8)
        result=ops.euler_sample(ConstantVelocity(),torch.zeros(2,dtype=torch.long),initial,2)
        torch.testing.assert_close(result,torch.full_like(initial,2))
        assert not initial.any()
        try: ops.euler_sample(ConstantVelocity(),labels,initial,0)
        except ValueError: pass
        else: raise AssertionError('zero steps accepted')
        torch.testing.assert_close(ops.decode_latents(system.codec,result),system.codec.decode(result))
        # Clean teacher and projector must NEVER be reached in inference.
        def forbidden(*args,**kwargs): raise AssertionError('training-only path used during sample')
        system.teacher.forward=forbidden
        system.denoiser.projector.forward=forbidden
        system.ema_denoiser.projector.forward=forbidden
        a=system.sample(labels,torch.zeros(4,4,8,8),2)
        b=system.sample(labels,torch.zeros(4,4,8,8),2)
        torch.testing.assert_close(a,b)
        assert a.shape==(4,3,16,16) and torch.isfinite(a).all()
    else:
        raise ValueError('lesson in 1..6 required')
    print(f'REPA lesson {lesson} {implementation}: passed')


def exercise_training(system, steps=4):
    if steps < 4:
        raise ValueError('at least four updates required to check zero-init warm-up')
    images,labels=toy_batch()
    before={name:p.detach().clone() for name,p in system.named_parameters()}
    opt=torch.optim.AdamW(system.denoiser.parameters(),lr=.003,weight_decay=0)
    clean=system.codec.encode(images).detach()
    noise=torch.randn_like(clean); time=torch.tensor([.1,.3,.6,.9])
    records=[]
    for _ in range(steps):
        result=system.ops.training_step(system,opt,images,labels,noise=noise,time=time)
        assert all(v.ndim==0 and torch.isfinite(v) and not v.requires_grad for v in result.values())
        records.append({key:float(value) for key,value in result.items()})
    for prefix in ('codec.','teacher.'):
        for name,p in system.named_parameters():
            if name.startswith(prefix):
                assert p.grad is None and torch.equal(before[name],p), name
    for group in ('patch.','blocks.0.attention.','blocks.1.attention.','projector.','output.'):
        params=[(name,p) for name,p in system.denoiser.named_parameters() if name.startswith(group)]
        assert any(p.grad is not None and p.grad.abs().sum()>0 for _,p in params), f'nonzero gradient missing: {group}'
        assert any(not torch.equal(before['denoiser.'+name],p) for name,p in params), f'update missing: {group}'
    assert all(not p.requires_grad and p.grad is None for p in system.ema_denoiser.parameters())
    assert any(not torch.equal(before['ema_denoiser.'+name],p) for name,p in system.ema_denoiser.named_parameters())
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson',type=int,choices=range(1,7))
    parser.add_argument('--implementation',choices=('reference','practice'),default='practice')
    args=parser.parse_args(); incomplete=False
    for number in ([args.lesson] if args.lesson else range(1,7)):
        try: check_lesson(number,args.implementation)
        except NotImplementedError as exc: print(exc); incomplete=True
    return 2 if incomplete else 0


if __name__=='__main__':
    raise SystemExit(main())
