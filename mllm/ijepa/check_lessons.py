"""局部数值、无泄漏与 full-image teacher 回归检查。"""
import argparse
import traceback
import torch
from reference_ijepa import IJEPA, sample_block_masks

def check_full_teacher():
    model=IJEPA(dim=16,heads=4,depth=1,predictor_dim=16,predictor_depth=1)
    images=torch.randn(2,3,32,32);masks=sample_block_masks((4,4),num_targets=1,block_size=(2,2))
    seen=[]
    handle=model.target_encoder.blocks[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].shape[1]))
    out=model(images,masks);handle.remove()
    assert seen==[16], 'teacher 必须先编码全部16 patches，再 gather target'
    with torch.no_grad():full,_=model.target_encoder(images)
    torch.testing.assert_close(out['targets'],full.index_select(1,out['target_indices']))
    assert not out['targets'].requires_grad
    changed=images.clone();gh,gw=4,4
    # Perturb only target pixels: visible context representations must remain identical.
    for idx in out['target_indices']:
        r,c=divmod(int(idx),gw);changed[:,:,r*8:(r+1)*8,c*8:(c+1)*8]+=20
    a,_=model.context_encoder.patch_tokens(images);b,_=model.context_encoder.patch_tokens(changed)
    torch.testing.assert_close(model.context_encoder.encode(a,out['context_indices']),model.context_encoder.encode(b,out['context_indices']))

def check(lesson,implementation):
    torch.manual_seed(7);ref=implementation=='reference'
    if lesson==1:
        if ref:m=sample_block_masks((4,4),num_targets=2,block_size=(2,2));targets,context=m.targets,m.context
        else:
            from lesson1_block_masks import sample_block_masks as f, rectangle_mask, context_complement
            fixed=rectangle_mask((2,3),(0,0),(1,2))
            torch.testing.assert_close(fixed,torch.tensor([True,True,False,False,False,False]))
            torch.testing.assert_close(context_complement([fixed]),~fixed)
            targets,context=f((4,4),2,(2,2))
        assert len(targets)==2 and all(t.dtype==torch.bool and t.shape==(16,) and t.sum()==4 for t in targets)
        assert not (targets[0]&targets[1]).any();assert torch.equal(context,~torch.stack(list(targets)).any(0))
    elif lesson==2:
        from lesson2_predictor_packing import pack_predictor_tokens
        c=torch.tensor([[[1.],[2.]]]);m=torch.tensor([[[10.]]]);p=torch.tensor([[[3.],[4.]]])
        result=torch.cat((c,m.expand(1,2,1)+p),1) if ref else pack_predictor_tokens(c,m,p)
        torch.testing.assert_close(result,torch.tensor([[[1.],[2.],[13.],[14.]]]))
    elif lesson==3:
        from edu_core.training import update_ema,freeze_and_keep_eval
        from lesson3_ema_teacher import update_target
        t=torch.nn.Linear(1,1,bias=False);s=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():t.weight.fill_(2);s.weight.fill_(6)
        if ref:update_ema(t,s,.75);freeze_and_keep_eval(t)
        else:update_target(t,s,.75)
        torch.testing.assert_close(t.weight,torch.tensor([[3.]]));assert not t.training and not t.weight.requires_grad
    else:
        if not ref:
            from lesson4_training_probe import latent_regression_loss
            pred=torch.zeros(1,2,3,requires_grad=True);target=torch.ones(1,2,3,requires_grad=True)
            loss=latent_regression_loss(pred,target)
            torch.testing.assert_close(loss,torch.tensor(.5));loss.backward()
            assert target.grad is None and pred.grad is not None, "target 必须 stop-gradient"
        check_full_teacher()
        if ref:from run_ijepa_demo import main;main('reference')
        else:
            from lesson4_training_probe import jepa_training_step
            model=IJEPA(dim=16,heads=4,depth=1,predictor_dim=16,predictor_depth=1);opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
            before=next(model.context_encoder.parameters()).detach().clone();tb=next(model.target_encoder.parameters()).detach().clone()
            loss=jepa_training_step(model,torch.randn(2,3,32,32),sample_block_masks((4,4)),opt)
            assert loss.ndim==0 and torch.isfinite(loss)
            after=next(model.context_encoder.parameters()).detach();ta=next(model.target_encoder.parameters()).detach()
            assert not torch.equal(before,after);torch.testing.assert_close(ta,tb*.9+after*.1)
            assert all(p.grad is None for p in model.target_encoder.parameters())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lesson',type=int,choices=range(1,5));p.add_argument('--implementation',choices=['reference','practice'],default='practice');a=p.parse_args()
    try:
        for lesson in ([a.lesson] if a.lesson is not None else range(1,5)):
            check(lesson,a.implementation)
            print(f"lesson {lesson} {a.implementation} passed")
    except NotImplementedError:traceback.print_exc();p.exit(2,f'未完成：I-JEPA 第 {lesson} 关；见 traceback 和 HINTS.md。\n')
    print(f'I-JEPA lesson {lesson} {a.implementation} passed')
