"""Independent numerical checks. practice never falls back to answers."""
import argparse
import math
import torch
import torch.nn.functional as F
from . import reference_dinov2 as ref


def check(lesson,implementation='practice'):
    torch.manual_seed(11);torch.set_num_threads(1)
    practice=implementation=='practice'
    if lesson==1:
        if practice:
            from . import lesson1_masking as ops
        else:ops=ref
        mask=ops.rectangle_mask((2,3),(0,1,2,1))
        assert torch.equal(mask,torch.tensor([[False,True,False],[False,True,False]]))
        assert ops.mask_count(16,False,.25)==0 and ops.mask_count(16,True,.25)==4
        assert ops.mask_count(4,True,.01)==1
        kw={'rectangle_fn':ops.rectangle_mask,'count_fn':ops.mask_count}
        torch.manual_seed(1);a=ref.sample_masks(16,(4,4),'cpu',**kw)
        torch.manual_seed(1);b=ref.sample_masks(16,(4,4),'cpu',**kw)
        assert torch.equal(a,b) and (a.sum(1)==0).any() and (a.sum(1)>0).any()
        assert a.sum(1).unique().numel()>2
    elif lesson==2:
        if practice:
            from . import lesson2_heads as ops
            from .scaffolding import build_modules
            s,t=build_modules(ops.choose_patch_head,ops.initialize_teacher,16,8)
        else:s,t=ref.build_modules(16,8)
        assert s['dino_head'] is not s['ibot_head']
        assert set(map(id,s['dino_head'].parameters())).isdisjoint(map(id,s['ibot_head'].parameters()))
        for k,v in s.state_dict().items():torch.testing.assert_close(v,t.state_dict()[k],rtol=0,atol=0)
        assert all(not p.requires_grad for p in t.parameters()) and not t.training
        assert any(p.requires_grad for p in s.parameters())
        # Explicit compatibility mode must share object identity.
        if practice:s,_=build_modules(ops.choose_patch_head,ops.initialize_teacher,16,8,True)
        else:s,_=ref.build_modules(16,8,True)
        assert s['dino_head'] is s['ibot_head']
    elif lesson==3:
        if practice:
            from .lesson3_sinkhorn import sinkhorn as fn
        else:fn=ref.sinkhorn
        x=torch.tensor([[.1,.3,-.2],[.7,.1,.2],[.2,.8,.1],[.9,.2,.1]],requires_grad=True)
        p=fn(x,temperature=.5,iterations=100)
        assert not p.requires_grad
        torch.testing.assert_close(p.sum(-1),torch.ones(4))
        torch.testing.assert_close(p.sum(0),torch.full((3,),4/3),rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(fn(torch.zeros(2,2)),torch.full((2,2),.5))
        shifted=fn(x.detach()+10000,.5,100)
        torch.testing.assert_close(p,shifted,atol=3e-4,rtol=3e-4)
        assert torch.isfinite(fn(torch.tensor([[1e4,-1e4],[-1e4,1e4]]))).all()
    elif lesson==4:
        if practice:
            from .lesson4_ibot import ibot_loss as fn
        else:fn=ref.ibot_loss
        # row 0 one target with CE log(2), row1 three with CE log(4).
        s=torch.zeros(2,3,2,requires_grad=True)
        with torch.no_grad():s[1,:,1]=math.log(3)
        target=torch.zeros_like(s,requires_grad=True)
        with torch.no_grad():target[:,:,0]=1
        mask=torch.tensor([[True,False,False],[True,True,True]])
        loss=fn(s,target,mask,1.)
        torch.testing.assert_close(loss,torch.tensor((math.log(2)+math.log(4))/2))
        loss.backward();assert target.grad is None and s.grad[0,1:].abs().sum()==0
        z=s.detach().clone().requires_grad_()
        empty=fn(z,target,torch.zeros_like(mask),1.)
        assert empty.requires_grad and float(empty.detach())==0
        empty.backward();assert torch.equal(z.grad,torch.zeros_like(z))
        one=fn(s.detach(),target,torch.tensor([[True,False,False],[False,False,False]]),1.)
        torch.testing.assert_close(one,torch.tensor(math.log(2)/2))
    elif lesson==5:
        if practice:
            from .lesson5_koleo import koleo as fn
        else:fn=ref.koleo
        features=torch.eye(2,requires_grad=True)
        result=fn(features)
        torch.testing.assert_close(result,torch.tensor(-math.log(math.sqrt(2))))
        result.backward();assert torch.isfinite(features.grad).all()
        torch.testing.assert_close(fn(features.detach()*7),result.detach())
        assert torch.isfinite(fn(torch.ones(3,2)))
        try:fn(torch.ones(1,2))
        except ValueError:pass
        else:raise AssertionError('B<2 must be rejected')
    elif lesson==6:
        if practice:
            from . import lesson6_training as ops
        else:ops=ref
        assert abs(ops.teacher_temperature(0)-.04)<1e-8
        assert abs(ops.teacher_temperature(2)-.055)<1e-8
        assert abs(ops.teacher_temperature(4)-.07)<1e-8
        assert abs(ops.teacher_temperature(9)-.07)<1e-8
        assert abs(ops.teacher_temperature(0,0)-.07)<1e-8
        assert abs(ops.ema_momentum(0,8)-.994)<1e-8 and ops.ema_momentum(8,8)==1
        model=ref.build_toy_dinov2();model.ops.teacher_temperature=ops.teacher_temperature;model.ops.ema_momentum=ops.ema_momentum
        optimizer=torch.optim.SGD(model.student.parameters(),lr=.01)
        crops,masks=model.views(torch.rand(4,3,40,40))
        before=next(model.student.parameters()).detach().clone()
        teacher_before=next(model.teacher.parameters()).detach().clone()
        out=ops.training_step(model,crops,masks,optimizer,0,8)
        assert torch.isfinite(out['loss']) and not out['loss'].requires_grad
        assert not torch.equal(before,next(model.student.parameters()))
        torch.testing.assert_close(next(model.teacher.parameters()),teacher_before*.994+next(model.student.parameters()).detach()*.006)
        assert all(p.grad is None for p in model.teacher.parameters())
    else:raise ValueError('lesson 1..6')
    print(f'DINOv2 lesson {lesson} {implementation}: passed')


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lesson',type=int,choices=range(1,7))
    p.add_argument('--implementation',choices=['reference','practice'],default='practice');a=p.parse_args()
    try:
        for n in ([a.lesson] if a.lesson else range(1,7)):check(n,a.implementation)
    except NotImplementedError as exc:p.exit(2,f'未完成：{exc}；见HINTS.md。\n')

if __name__=='__main__':cli()
