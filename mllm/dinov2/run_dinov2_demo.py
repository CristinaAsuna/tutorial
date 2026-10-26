"""Short structured-data CPU training; diagnostics are not paper benchmark metrics."""
import argparse
import torch
from edu_core.training import seed_everything
from mllm.dino.diagnostics import patterned_images,feature_diagnostics,write_diagnostics
from .reference_dinov2 import build_toy_dinov2


def main(implementation='reference',steps=8,output_dir=None,model=None):
    if steps <= 0:
        raise ValueError('steps must be positive')
    seed_everything(7);torch.set_num_threads(1)
    if model is None:
        if implementation=='practice':
            from .practice_dinov2 import build_toy_dinov2 as factory
        else:
            factory=build_toy_dinov2
        model=factory()
    images=patterned_images()
    optimizer=torch.optim.AdamW(model.student.parameters(),lr=1e-3)
    before=[p.detach().clone() for p in model.student.parameters()]
    records=[];metrics,initial=feature_diagnostics(model,images)
    records.append({'step':0,**metrics})
    for step in range(steps):
        crops,masks=model.views(images)
        old=next(model.teacher.parameters()).detach().clone()
        losses=model.ops.training_step(model,crops,masks,optimizer,step,steps)
        assert all(torch.isfinite(v) for v in losses.values())
        momentum=model.ops.ema_momentum(step,steps)
        torch.testing.assert_close(next(model.teacher.parameters()),old*momentum+next(model.student.parameters()).detach()*(1-momentum))
        assert all(p.grad is None and not p.requires_grad for p in model.teacher.parameters())
        assert not model.teacher.training
        metrics,_=feature_diagnostics(model,images)
        records.append({'step':step+1,'temperature':model.ops.teacher_temperature(step),
                        'ema_momentum':momentum,**{k:float(v) for k,v in losses.items()},**metrics})
    assert any(not torch.equal(a,b) for a,b in zip(before,model.student.parameters()))
    assert model.student['dino_head'] is not model.student['ibot_head']
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.student['dino_head'].parameters())
    _,final=feature_diagnostics(model,images)
    if output_dir:
        write_diagnostics(output_dir,records,{'initial':initial,'final':final})
    print(f'DINOv2 CPU {implementation}: {steps} steps; dual heads/SK/iBOT/KoLeo/EMA passed')
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
