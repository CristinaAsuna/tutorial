"""独立关卡语义验收。reference 是 oracle；practice 不自动借用答案。"""
import argparse
import torch
import reference_flamingo as ref
from edu_core.attention import GatedCrossAttentionBlock

def check(lesson,implementation="practice"):
    torch.manual_seed(23); practice=implementation=="practice"
    if lesson==1:
        if practice:
            from lesson1_perceiver_resampler import PerceiverResampler
        else: PerceiverResampler=ref.PerceiverResampler
        oracle=ref.PerceiverResampler(8,12,num_latents=3,num_heads=3,depth=1)
        model=PerceiverResampler(8,12,num_latents=3,num_heads=3,depth=1); model.load_state_dict(oracle.state_dict())
        x=torch.randn(2,2,5,8); result=model(x)
        torch.testing.assert_close(result,oracle(x))
        changed=x.clone(); changed[:,1]+=10
        torch.testing.assert_close(result[:,0],model(changed)[:,0])
        assert not torch.allclose(result[:,1],model(changed)[:,1])
        result.sum().backward(); assert model.latents.grad is not None and model.vision_proj.weight.grad is not None
    elif lesson==2:
        model=ref.build_toy_flamingo(); ids=torch.tensor([[1,-200,7,-200,8,0]])
        valid=torch.tensor([[1,1,1,1,1,0]],dtype=torch.bool)
        if practice:
            from lesson2_media_causal_mask import build_media_mask
            fn=lambda x,v:build_media_mask(x,-200,2,3,v)
        else: fn=lambda x,v:model.build_media_attention_mask(x,2,v)
        actual=fn(ids,valid)
        expected=torch.tensor([[[0]*6,[1]*3+[0]*3,[1]*3+[0]*3,[1]*6,[1]*6,[0]*6]],dtype=torch.bool)
        assert torch.equal(actual,expected)
        try: fn(torch.tensor([[1,-200,7,8,8,0]]),valid)
        except ValueError: pass
        else: raise AssertionError("missing sentinel must fail")
    elif lesson==3:
        if practice:
            from lesson3_gated_cross_attention import GatedCrossAttention
            block=GatedCrossAttention(12,3)
        else: block=GatedCrossAttentionBlock(12,3,mlp_ratio=2.)
        oracle=GatedCrossAttentionBlock(12,3,mlp_ratio=2.); block.load_state_dict(oracle.state_dict())
        x=torch.randn(2,4,12); memory=torch.randn(2,6,12)
        allow=torch.ones(2,4,6,dtype=torch.bool); allow[:,0]=False
        assert torch.equal(block(x,memory,attention_mask=allow),x)
        with torch.no_grad():
            block.attn_gate.fill_(.7); block.ff_gate.fill_(.4)
        oracle.load_state_dict(block.state_dict())
        torch.testing.assert_close(block(x,memory,attention_mask=allow),oracle(x,memory,attention_mask=allow))
        # 单独关FFN门时，全遮蔽query不能接收视觉输出投影bias。
        with torch.no_grad(): block.ff_gate.zero_()
        torch.testing.assert_close(block(x,memory,attention_mask=allow)[:,0],x[:,0])
    elif lesson==4:
        model=ref.build_toy_flamingo(); decoder=model.decoder
        assert set(decoder.gated_cross_attn)=={"1","3"}
        ids=torch.tensor([[1,3,7,3,8]]); visual=torch.randn(1,2,3,32); valid=torch.ones_like(ids,dtype=torch.bool)
        media=model.build_media_attention_mask(torch.tensor([[1,-200,7,-200,8]]),2,valid)
        for gate in decoder.gated_cross_attn.values():
            with torch.no_grad(): gate.attn_gate.fill_(.5)
        if practice:
            from lesson4_training_and_generation import decoder_forward,connector_training_step
            actual=decoder_forward(decoder,ids,visual,media,valid)
        else: actual=decoder(ids,visual,media,valid)
        torch.testing.assert_close(actual,decoder(ids,visual,media,valid))
        if practice:
            raw=torch.tensor([[1,-200,7,-200,8]]); labels=torch.tensor([[-100,-100,-100,-100,8]])
            before=decoder.gated_cross_attn['1'].attn_gate.detach().clone()
            out=connector_training_step(model,torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01),raw,torch.randn(1,2,3,8,8),valid,labels)
            assert torch.isfinite(out['loss']) and not torch.equal(before,decoder.gated_cross_attn['1'].attn_gate)
            assert all(p.grad is None for p in decoder.blocks.parameters())
    print(f"Flamingo lesson {lesson} {implementation}: passed")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--lesson",type=int,choices=range(1,5),default=None)
    p.add_argument("--implementation",choices=("reference","practice"),default="practice")
    args=p.parse_args()
    for lesson in ([args.lesson] if args.lesson is not None else range(1,5)):
        check(lesson,args.implementation)
