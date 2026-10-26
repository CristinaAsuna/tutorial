"""Two-phase CPU demo: early snapshot, base training, Gram refinement and refresh."""
import argparse
import torch
from edu_core.training import seed_everything
from mllm.dino.diagnostics import patterned_images,feature_diagnostics,write_diagnostics
from .reference_dinov3 import build_toy_dinov3


def main(implementation='reference',steps=8,output_dir=None,model=None):
    if steps < 6:
        raise ValueError('use at least 6 steps to exercise snapshot/refinement/refresh')
    seed_everything(7);torch.set_num_threads(1)
    if model is None:
        if implementation=='practice':
            from .practice_dinov3 import build_toy_dinov3 as factory
        else:factory=build_toy_dinov3
        model=factory()
    optimizer=torch.optim.AdamW(model.student.parameters(),lr=1e-3)
    images=patterned_images();metrics,initial=feature_diagnostics(model,images)
    records=[{'step':0,**metrics}];matrices={'initial':initial}
    before=[p.detach().clone() for p in model.student.parameters()]
    events=[]
    for step in range(int(model.completed_steps),steps):
        crops,masks,high=model.views(images)
        gram_before={k:v.clone() for k,v in model.gram_teacher.state_dict().items()}
        ema_before=next(model.teacher.parameters()).detach().clone()
        losses=model.ops.training_step(model,crops,masks,high,optimizer,step,steps)
        assert int(model.completed_steps)==step+1
        assert all(torch.isfinite(v) for v in losses.values())
        m=model.ops.ema_momentum(step,steps)
        torch.testing.assert_close(next(model.teacher.parameters()),ema_before*m+next(model.student.parameters()).detach()*(1-m))
        event=model.ops.gram_event(step+1,model.snapshot_step,model.refine_start,model.refresh_every)
        events.append(event)
        expected=model.teacher['backbone'].state_dict() if event else gram_before
        for k,v in model.gram_teacher.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
        assert all(p.grad is None and not p.requires_grad for p in model.teacher.parameters())
        assert all(p.grad is None and not p.requires_grad for p in model.gram_teacher.parameters())
        assert not model.teacher.training and not model.gram_teacher.training
        if step < model.refine_start:assert float(losses['gram'])==0
        metrics,gram=feature_diagnostics(model,images)
        records.append({'step':step+1,'event':event,'gram_ready':bool(model.gram_ready),
                        **{k:float(v) for k,v in losses.items()},**metrics})
        if step+1==model.refine_start:matrices['before refinement']=gram
    assert 'snapshot' in events and 'refresh' in events
    assert any(row.get('gram',0)>0 for row in records)
    assert any(not torch.equal(a,b) for a,b in zip(before,model.student.parameters()))
    _,matrices['after refinement']=feature_diagnostics(model,images)
    if output_dir:write_diagnostics(output_dir,records,matrices)
    print(f'DINOv3 Gram CPU {implementation}: {steps} steps; snapshot/refinement/alignment/refresh passed')
    print(records[-1])
    return model,records


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation',choices=['reference','practice'],default='reference')
    p.add_argument('--steps',type=int,default=8);p.add_argument('--output-dir')
    a=p.parse_args()
    try:main(a.implementation,a.steps,a.output_dir)
    except NotImplementedError as exc:p.exit(2,f'未完成练习：{exc}\n')

if __name__=='__main__':cli()
