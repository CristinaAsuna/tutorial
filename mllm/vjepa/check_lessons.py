"""逐关检查原始 tubelet 内容、mask 合同、predictor 与训练步骤。"""
import argparse
import traceback
import torch
import torch.nn.functional as F
from reference_vjepa import MiniVideoViT, LatentPredictor, VJEPA, sample_spatiotemporal_masks

def patchify_oracle(x,t,p):
    b,c,frames,h,w=x.shape
    return x.reshape(b,c,frames//t,t,h//p,p,w//p,p).permute(0,2,4,6,3,5,7,1).reshape(b,-1,c*t*p*p)

def check(lesson,implementation):
    torch.manual_seed(7);ref=implementation=='reference'
    if lesson==1:
        from lesson1_tubelet_embed import tubelet_patchify
        x=torch.arange(2*2*4*4*6).reshape(2,2,4,4,6).float()
        expected=patchify_oracle(x,2,2);result=expected if ref else tubelet_patchify(x,2,2)
        torch.testing.assert_close(result,expected)
        embed=MiniVideoViT(video_size=(4,4,6),tubelet_size=2,patch_size=2,embed_dim=8,depth=1,in_chans=2).tubelet_embed
        projected=F.linear(result,embed.proj.weight.permute(0,2,3,4,1).flatten(1),embed.proj.bias)
        torch.testing.assert_close(projected,embed(x)[0])
    elif lesson==2:
        if ref:target,context=sample_spatiotemporal_masks(2,(2,4,4),(1,2,2),2)
        else:
            from lesson2_spatiotemporal_masks import make_masks, cuboid_mask, context_complement
            fixed=cuboid_mask((2,2,2),(1,0,1),(1,1,1))
            expected=torch.zeros(8,dtype=torch.bool);expected[5]=True
            torch.testing.assert_close(fixed,expected)
            torch.testing.assert_close(context_complement(fixed.unsqueeze(0)),(~fixed).unsqueeze(0))
            target,context=make_masks(2,(2,4,4),(1,2,2),2)
        assert target.dtype==context.dtype==torch.bool and target.shape==context.shape==(2,32)
        assert (target.sum(1)==8).all() and torch.equal(context,~target)
        # Exactly two disjoint cuboids, not arbitrary random 8 positions.
        for row in target:
            occupied=row.reshape(2,4,4)
            blocks=[]
            for t in range(2):
                for h in range(3):
                    for w in range(3):
                        m=torch.zeros_like(occupied);m[t:t+1,h:h+2,w:w+2]=True
                        if (occupied[m]).all():blocks.append(m)
            assert any(not (a&b).any() and torch.equal(a|b,occupied) for a in blocks for b in blocks), 'mask 不是两块不重叠 cuboids'
    elif lesson==3:
        if ref:cls=LatentPredictor
        else:from lesson3_latent_predictor import Predictor as cls
        model=cls(16,predictor_dim=12,depth=1,num_heads=4)
        oracle=LatentPredictor(16,predictor_dim=12,depth=1,num_heads=4);oracle.load_state_dict(model.state_dict())
        c=torch.randn(2,3,16,requires_grad=True);p=torch.randn(2,2,16)
        result=model(c,p);torch.testing.assert_close(result,oracle(c,p))
        result.square().mean().backward();assert c.grad is not None and model.mask_token.grad is not None
    else:
        if not ref:
            from lesson4_ema_training import latent_regression_loss
            pred=torch.zeros(1,2,3,requires_grad=True);target=torch.ones(1,2,3,requires_grad=True)
            loss=latent_regression_loss(pred,target)
            torch.testing.assert_close(loss,torch.tensor(.5));loss.backward()
            assert target.grad is None and pred.grad is not None, "target 必须 stop-gradient"
        if ref:from run_vjepa_demo import main;main('reference')
        else:
            from lesson4_ema_training import vjepa_training_step
            model=VJEPA(MiniVideoViT(embed_dim=16,depth=1),LatentPredictor(16,predictor_dim=12,depth=1))
            target,context=sample_spatiotemporal_masks(2,model.context_encoder.base_grid)
            opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
            before=next(model.context_encoder.parameters()).detach().clone();tb=next(model.target_encoder.parameters()).detach().clone()
            loss=vjepa_training_step(model,torch.randn(2,3,8,32,32),target,context,opt,.75)
            assert loss.ndim==0 and torch.isfinite(loss)
            after=next(model.context_encoder.parameters()).detach();ta=next(model.target_encoder.parameters()).detach()
            assert not torch.equal(before,after);torch.testing.assert_close(ta,tb*.75+after*.25)
            assert all(p.grad is None for p in model.target_encoder.parameters()) and not model.target_encoder.training

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lesson',type=int,choices=range(1,5));p.add_argument('--implementation',choices=['reference','practice'],default='practice');a=p.parse_args()
    try:
        for lesson in ([a.lesson] if a.lesson is not None else range(1,5)):
            check(lesson,a.implementation)
            print(f"lesson {lesson} {a.implementation} passed")
    except NotImplementedError:traceback.print_exc();p.exit(2,f'未完成：V-JEPA 第 {lesson} 关；见 traceback 和 HINTS.md。\n')
    print(f'V-JEPA lesson {lesson} {a.implementation} passed')
