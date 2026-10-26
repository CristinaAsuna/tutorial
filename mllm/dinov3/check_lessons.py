"""Gram properties, phase semantics and coordinate alignment, one lesson at a time."""
import argparse
import copy
from functools import partial
import torch
from . import reference_dinov3 as ref


def check(lesson,implementation='practice'):
    torch.manual_seed(3);torch.set_num_threads(1);practice=implementation=='practice'
    if lesson==1:
        if practice:
            from .lesson1_gram import gram_matrix as fn
        else:fn=ref.gram_matrix
        x=torch.tensor([[[1.,0.],[0.,2.]]],requires_grad=True)
        torch.testing.assert_close(fn(x),torch.eye(2)[None])
        rotation=torch.tensor([[0.,-1.],[1.,0.]])
        torch.testing.assert_close(fn(x@rotation),fn(x))
        fn(x).sum().backward();assert x.grad is not None
    elif lesson==2:
        if practice:
            from .lesson2_anchoring import gram_loss
            fn=partial(gram_loss,gram_fn=ref.gram_matrix)
        else:fn=ref.gram_loss
        x=torch.eye(2)[None].requires_grad_();teacher=torch.ones(1,2,3,requires_grad=True)
        torch.testing.assert_close(fn(x,teacher),torch.tensor(.5))
        torch.testing.assert_close(fn(x,teacher,reduction='sum'),torch.tensor(2.))
        fn(x,teacher).backward();assert x.grad is not None and teacher.grad is None
        torch.testing.assert_close(fn(x,x),torch.tensor(0.))
        anti=torch.tensor([[[1.,0.],[-1.,0.]]])
        assert fn(anti,teacher,remove_neg=True) < fn(anti,teacher)
    elif lesson==3:
        if practice:
            from . import lesson3_teacher as ops
        else:ops=ref
        assert [ops.gram_event(i) for i in range(1,9)]==[None,'snapshot',None,None,None,'refresh',None,'refresh']
        source=torch.nn.Linear(2,2);target=copy.deepcopy(source)
        with torch.no_grad():source.weight.add_(3)
        ops.sync_gram_teacher(target,source)
        torch.testing.assert_close(target.weight,source.weight)
        assert not target.training and all(not p.requires_grad for p in target.parameters())
        assert target.weight.data_ptr()!=source.weight.data_ptr() and source.weight.requires_grad
    elif lesson==4:
        if practice:
            from . import lesson4_resolution as ops
        else:ops=ref
        images=torch.arange(64.).reshape(1,1,8,8)
        boxes=torch.tensor([[2,3,4,4]])
        low,high=ops.aligned_crops(images,boxes,4,8)
        torch.testing.assert_close(low,images[:,:,2:6,3:7])
        from mllm.dino.multicrop import crop_and_resize
        torch.testing.assert_close(high,crop_and_resize(images,boxes,8))
        x=torch.arange(12.).reshape(1,4,3)
        actual=ops.downsample_features(x,(2,2),(1,1))
        torch.testing.assert_close(actual,x.mean(1,keepdim=True))
        rect=torch.arange(24.).reshape(1,6,4)
        torch.testing.assert_close(ops.downsample_features(rect,(2,3),(2,3)),rect)
    elif lesson==5:
        if practice:
            from . import lesson5_refinement as ops
        else:ops=ref
        total,gram=ops.refinement_loss(torch.tensor(2.),[torch.tensor(1.),torch.tensor(3.)],
                                      [torch.tensor(0.),torch.tensor(0.)],lambda s,t:s,.5)
        assert total==3 and gram==2
        model=ref.build_toy_dinov3();model.ops.refinement_loss=ops.refinement_loss
        opt=torch.optim.SGD(model.student.parameters(),lr=.001)
        for step in range(6):
            crops,masks,high=model.views(torch.rand(4,3,80,80))
            old=next(model.teacher.parameters()).detach().clone()
            result=ops.training_step(model,crops,masks,high,opt,step,8)
            assert int(model.completed_steps)==step+1 and not result['loss'].requires_grad
            m=model.ops.ema_momentum(step,8)
            torch.testing.assert_close(next(model.teacher.parameters()),old*m+next(model.student.parameters()).detach()*(1-m))
            if step <4:assert result['gram']==0
            else:assert result['gram']>0
        assert int(model.last_gram_update)==6
    else:raise ValueError('lesson 1..5')
    print(f'DINOv3 lesson {lesson} {implementation}: passed')


def cli():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lesson',type=int,choices=range(1,6))
    p.add_argument('--implementation',choices=['reference','practice'],default='practice');a=p.parse_args()
    try:
        for n in ([a.lesson] if a.lesson else range(1,6)):check(n,a.implementation)
    except NotImplementedError as exc:p.exit(2,f'未完成：{exc}；见HINTS.md。\n')

if __name__=='__main__':cli()
