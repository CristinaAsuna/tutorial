"""逐关数值与机制验收；默认检查学生实现。"""
import argparse
import torch
import reference_llava as ref


def check(lesson, implementation):
    torch.manual_seed(9)
    model=ref.build_toy_llava()
    if implementation=='practice':
        import lesson1_vision_and_projector as l1
        import lesson2_image_token_packing as l2
        import lesson3_sft_loss as l3
        from lesson4_training_and_generation import train_one_step
    ids=torch.tensor([[3,-200,7,0],[4,-200,8,2]])
    valid=torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    labels=torch.tensor([[-100,-100,7,-100],[-100,-100,8,2]])
    if lesson==1:
        images=torch.randn(2,3,8,8)
        got=l1.vision_to_llm_tokens(model.vision_encoder,model.projector,images) if implementation=='practice' else model.encode_images(images)
        expected=model.projector(model.vision_encoder(images)[:,1:])
        torch.testing.assert_close(got,expected)
        got.sum().backward()
        assert any(p.grad is not None for p in model.projector.parameters())
        assert all(p.grad is None for p in model.vision_encoder.parameters())
    elif lesson==2:
        if implementation=='practice':
            missing=[]
            row_ids=torch.tensor([3,-200,7])
            image=torch.arange(64,dtype=torch.float).reshape(2,32)
            try:
                embedded=l2.replace_image_sentinel(row_ids,image,model.llm.embed_tokens,1)
                expected_row=torch.cat((model.llm.embed_tokens(row_ids[:1]),image,model.llm.embed_tokens(row_ids[2:])))
                torch.testing.assert_close(embedded,expected_row)
            except NotImplementedError as exc: missing.append(str(exc))
            try:
                mask,target=l2.expand_image_supervision(torch.tensor([1,1,0],dtype=torch.bool),torch.tensor([-100,-100,7]),1,2)
                assert mask.tolist()==[True,True,True,False]
                assert target.tolist()==[-100,-100,-100,7]
                _,none=l2.expand_image_supervision(torch.ones(3,dtype=torch.bool),None,1,2)
                assert none is None
            except NotImplementedError as exc: missing.append(str(exc))
            if missing: raise NotImplementedError('\n'.join(missing))
        features=torch.arange(2*2*32,dtype=torch.float).reshape(2,2,32)
        fn=(lambda i:l2.pack_one_image(i,features,model.llm.embed_tokens,valid,labels)) if implementation=='practice' else (lambda i:model.pack_multimodal_inputs(i,features,valid,labels))
        got=fn(ids); expected=model.pack_multimodal_inputs(ids,features,valid,labels)
        for a,b in zip(got,expected): torch.testing.assert_close(a,b)
        torch.testing.assert_close(got[0][:,1:3],features)
        assert got[2].tolist()==[[-100,-100,-100,7,-100],[-100,-100,-100,8,2]]
        for bad in [torch.tensor([[3,5,7,0],[4,-200,8,2]]),torch.tensor([[-200,-200,7,0],[4,-200,8,2]])]:
            try: fn(bad)
            except ValueError: pass
            else: raise AssertionError('invalid sentinel count must fail')
    elif lesson==3:
        assistant=torch.tensor([[0,0,1,1],[0,1,1,1]],dtype=torch.bool)
        got=l3.assistant_only_labels(ids,assistant_mask=assistant,attention_mask=valid) if implementation=='practice' else torch.where(assistant&valid&(ids!=-200),ids,-100)
        torch.testing.assert_close(got,labels)
    else:
        model.set_training_stage('pretrain_projector')
        opt=torch.optim.SGD(filter(lambda p:p.requires_grad,model.parameters()),lr=.1)
        batch=dict(input_ids=ids,pixel_values=torch.randn(2,3,8,8),attention_mask=valid,labels=labels)
        before=model.projector.layers[0].weight.detach().clone()
        if implementation=='practice': loss=train_one_step(model,opt,batch)
        else:
            opt.zero_grad(); loss=model(**batch)['loss']; loss.backward(); opt.step(); loss=loss.detach()
        assert loss.ndim==0 and torch.isfinite(loss) and not loss.requires_grad
        assert not torch.equal(before,model.projector.layers[0].weight)
        assert all(p.grad is None for p in model.llm.parameters())
    print(f'LLaVA lesson {lesson} passed ({implementation})')

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

