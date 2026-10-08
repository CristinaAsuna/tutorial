"""局部语义验收；只调用所选关卡，默认检查学生实现。"""
import argparse
import math
import torch
import reference_act as ref

def check(lesson, implementation="practice"):
    torch.manual_seed(41)
    practice=implementation=="practice"
    if lesson==1:
        if practice:
            from lesson1_action_chunks import sample_future_chunks as fn
        else: fn=ref.sample_action_chunks
        actions=torch.tensor([[[10.],[20.],[30.]]])
        chunks,valid=fn(actions,2)
        assert torch.equal(chunks,torch.tensor([[[[10.],[20.]],[[20.],[30.]],[[30.],[30.]]]]))
        assert torch.equal(valid,torch.tensor([[[1,1],[1,1],[1,0]]],dtype=torch.bool))
        big,mask=fn(actions,5); assert big.shape==(1,3,5,1) and mask.sum()==6
    elif lesson==2:
        if practice:
            from lesson2_cvae_style import StyleEncoder,reparameterize,cvae_loss
        else:
            StyleEncoder=ref.StyleEncoder
            def reparameterize(m,l,noise=None): return m+torch.exp(.5*l)*(torch.randn_like(m) if noise is None else noise)
            def cvae_loss(p,t,v,m,l,b):
                rec=ref.masked_l1(p,t,v); kl=ref.kl_to_standard_normal(m,l)
                return rec+b*kl,rec,kl
        oracle=ref.StyleEncoder(2,2,3,8,2,2).eval(); student=StyleEncoder(2,2,3,8,2,2).eval()
        student.load_state_dict(oracle.state_dict())
        q=torch.randn(2,2); a=torch.randn(2,3,2); mask=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
        actual=student(q,a,mask); expected=oracle(q,a,mask)
        for x,y in zip(actual,expected): torch.testing.assert_close(x,y)
        changed=a.clone(); changed[~mask]=999
        for x,y in zip(actual,student(q,changed,mask)): torch.testing.assert_close(x,y)
        m=torch.ones(1,2); l=torch.zeros_like(m)
        torch.testing.assert_close(reparameterize(m,l,torch.zeros_like(m)),m)
        loss,rec,kl=cvae_loss(torch.zeros_like(a),a,mask,m,l,.5)
        torch.testing.assert_close(kl,torch.tensor(1.)); torch.testing.assert_close(loss,rec+.5)
        torch.testing.assert_close(reparameterize(m,torch.full_like(m,math.log(4)),torch.ones_like(m)),m+2)
        try: cvae_loss(torch.zeros_like(a),a,torch.zeros_like(mask),m,l,.5)
        except ValueError: pass
        else: raise AssertionError("empty action mask must fail")
        changed=a.clone(); changed[~mask]=999
        torch.testing.assert_close(loss,cvae_loss(torch.zeros_like(a),changed,mask,m,l,.5)[0])
    elif lesson==3:
        oracle=ref.build_toy_act().eval()
        images=torch.randn(2,2,3,8,8); q=torch.randn(2,4); z=torch.randn(2,8)
        if practice:
            from lesson3_transformer_policy import encode_observation,decode_action_chunk
            memory=encode_observation(oracle,images,q,z)
            actions=decode_action_chunk(memory,oracle.action_queries,oracle.action_decoder,oracle.action_head)
        else: memory=oracle._encode_observation(images,q,z); actions=oracle._decode(images,q,z)
        torch.testing.assert_close(memory,oracle._encode_observation(images,q,z))
        torch.testing.assert_close(actions,oracle._decode(images,q,z))
        actions.sum().backward(); assert oracle.action_queries.grad is not None
    elif lesson==4:
        if practice:
            from lesson4_temporal_ensemble import ensemble_current_action
            value=ensemble_current_action([torch.tensor([[1.],[2.],[3.]]),torch.tensor([[10.],[20.],[30.]])],math.log(2))
            from practice_act import TemporalEnsembler
        else:
            TemporalEnsembler=ref.TemporalEnsembler
            en=TemporalEnsembler(3,1,math.log(2)); en.add(torch.tensor([[1.],[2.],[3.]])); value=en.add(torch.tensor([[10.],[20.],[30.]]))
        torch.testing.assert_close(value,torch.tensor([14/3]))
        en=TemporalEnsembler(3,1)
        for i in range(6): assert torch.isfinite(en.add(torch.ones(3,1)*i)).all()
        assert len(en._history)==3
    elif lesson==5:
        # 用已完成的外围部件隔离测试本关 glue；最后另用 practice demo 验全部接线。
        model=ref.build_toy_act(); images=torch.randn(2,2,3,8,8); q=torch.randn(2,4)
        a=torch.randn(2,5,4); valid=torch.ones(2,5,dtype=torch.bool)
        if practice:
            import lesson5_training_and_inference as student
            infer=student.policy_forward(model,images,q)
            torch.manual_seed(9); actual=student.policy_forward(model,images,q,a,valid,beta=.01)
        else:
            infer=model(images,q); torch.manual_seed(9); actual=model(images,q,a,valid,beta=.01)
        assert infer["loss"] is None
        torch.testing.assert_close(infer["actions"],model(images,q)["actions"])
        torch.manual_seed(9); expected=model(images,q,a,valid,beta=.01)
        for key in ("actions","loss","reconstruction_loss","kl_loss","mean","logvar"):
            torch.testing.assert_close(actual[key],expected[key])
        if practice:
            before=model.action_head.weight.detach().clone()
            result=student.training_step(model,torch.optim.AdamW(model.parameters(),lr=.001),images,q,a,valid)
            assert torch.isfinite(result["loss"]) and not torch.equal(before,model.action_head.weight)
    print(f"ACT lesson {lesson} {implementation}: passed")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--lesson",type=int,choices=range(1,6),default=None)
    p.add_argument("--implementation",choices=("reference","practice"),default="practice")
    args=p.parse_args()
    for lesson in ([args.lesson] if args.lesson is not None else range(1,6)):
        check(lesson,args.implementation)
