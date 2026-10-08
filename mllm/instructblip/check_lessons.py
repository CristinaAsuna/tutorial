"""局部关卡独立验收，不要求先完成其他关。"""
import argparse
import torch
import reference_instructblip as ref


def check(lesson,implementation):
    torch.manual_seed(9)
    if implementation=='practice':
        import lesson1_instruction_queries as l1
        import lesson2_query_only_vision as l2
        import lesson3_dual_tokenizer_prefix as l3
        from lesson4_training_and_generation import training_step
    if lesson==1:
        query=torch.arange(8,dtype=torch.float).reshape(1,2,4).requires_grad_()
        text=torch.randn(2,3,4); mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
        got=l1.instruction_aware_queries(query,text,mask) if implementation=='practice' else (torch.cat((query.expand(2,-1,-1),text),1),torch.cat((torch.ones(2,2,dtype=torch.bool),mask),1))
        torch.testing.assert_close(got[0][:,:2],query.expand(2,-1,-1)); torch.testing.assert_close(got[0][:,2:],text)
        assert got[1].tolist()==[[True,True,True,True,False],[True]*5]
        got[0].sum().backward(); torch.testing.assert_close(query.grad,torch.full_like(query,2))
    elif lesson==2:
        hidden=torch.randn(2,5,8); vision=torch.randn(2,3,6)
        layer=ref.InstructionQFormerLayer(8,2,6)
        expected=torch.cat((hidden[:,:2]+layer.cross_attn(layer.norm2(hidden[:,:2]),vision),hidden[:,2:]),1)
        got=l2.query_only_cross_attention(hidden,2,vision,layer.cross_attn,layer.norm2) if implementation=='practice' else expected
        torch.testing.assert_close(got,expected); assert torch.equal(got[:,2:],hidden[:,2:])
        changed=vision+torch.randn_like(vision)
        other=l2.query_only_cross_attention(hidden,2,changed,layer.cross_attn,layer.norm2) if implementation=='practice' else torch.cat((hidden[:,:2]+layer.cross_attn(layer.norm2(hidden[:,:2]),changed),hidden[:,2:]),1)
        assert not torch.equal(got[:,:2],other[:,:2]) and torch.equal(got[:,2:],other[:,2:])
    elif lesson==3:
        queries=torch.randn(2,2,4); proj=torch.nn.Linear(4,6); embedding=torch.nn.Embedding(20,6)
        prompt=torch.tensor([[9,0],[10,11]]); answer=torch.tensor([[14,0],[15,16]])
        pm=prompt!=0; am=answer!=0
        if implementation=='practice':
            missing=[]
            try:
                projected=l3.project_visual_queries(queries,proj)
                torch.testing.assert_close(projected,proj(queries))
                assert projected.requires_grad
            except NotImplementedError as exc: missing.append(str(exc))
            try:
                row=l3.concatenate_prefix_row(torch.ones(2,6),torch.full((1,6),2.),torch.full((1,6),3.))
                assert row[:,0].tolist()==[1.,1.,2.,3.]
                empty=l3.concatenate_prefix_row(torch.ones(2,6),torch.full((1,6),2.),torch.empty(0,6))
                assert empty.shape==(3,6)
            except NotImplementedError as exc: missing.append(str(exc))
            try:
                target=l3.prefix_answer_labels(2,1,torch.tensor([14]))
                assert target.tolist()==[-100,-100,-100,14]
                assert l3.prefix_answer_labels(2,1,torch.empty(0,dtype=torch.long)).tolist()==[-100]*3
            except NotImplementedError as exc: missing.append(str(exc))
            if missing: raise NotImplementedError('\n'.join(missing))
        prefix=proj(queries)
        rows=[torch.cat((prefix[r],embedding(prompt[r,pm[r]]),embedding(answer[r,am[r]]))) for r in range(2)]
        expected=torch.nn.utils.rnn.pad_sequence(rows,batch_first=True)
        labels=torch.tensor([[-100,-100,-100,14,-100,-100],[-100,-100,-100,-100,15,16]])
        mask=torch.tensor([[1,1,1,1,0,0],[1]*6],dtype=torch.bool)
        got=l3.build_llm_prefix(queries,llm_proj=proj,embed_tokens=embedding,prompt_ids=prompt,answer_ids=answer,prompt_mask=pm,answer_mask=am) if implementation=='practice' else (expected,mask,labels)
        for a,b in zip(got,(expected,mask,labels)): torch.testing.assert_close(a,b)
        got[0].sum().backward(); assert proj.weight.grad is not None
    else:
        model=ref.build_toy_instructblip(); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=.001)
        batch=dict(pixel_values=torch.randn(1,3,8,8),instruction_ids=torch.tensor([[4,5]]),llm_prompt_ids=torch.tensor([[9,10]]),answer_ids=torch.tensor([[14,15]]))
        before=model.qformer.query_tokens.detach().clone()
        if implementation=='practice': loss=training_step(model,opt,batch)
        else:
            model.train(); opt.zero_grad(); loss=model(**batch)['loss']; loss.backward(); opt.step(); loss=loss.detach()
        assert torch.isfinite(loss) and not loss.requires_grad
        assert not torch.equal(before,model.qformer.query_tokens)
        assert all(p.grad is None for p in model.vision_encoder.parameters()) and all(p.grad is None for p in model.llm.parameters())
        assert any(p.grad is not None for p in model.llm_proj.parameters())
    print(f'InstructBLIP lesson {lesson} passed ({implementation})')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--lesson',type=int,choices=range(1,5))
    parser.add_argument('--implementation',choices=['reference','practice'],default='practice'); args=parser.parse_args()
    failed=False
    for lesson in ([args.lesson] if args.lesson else range(1,5)):
        try: check(lesson,args.implementation)
        except NotImplementedError as exc:
            print(str(exc))
            failed=True
    if failed: parser.exit(2)

