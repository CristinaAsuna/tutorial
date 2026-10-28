"""Offline four-step reference/practice mechanism demonstration."""
import argparse
import json
from pathlib import Path
import torch
from .check_lessons import exercise_training
from .scaffolding import toy_batch


def write_diagnostics(output_dir, records, images, samples):
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    (destination/'losses.json').write_text(json.dumps({'mechanism_only':True,'paper_metrics_reproduced':False,'records':records},indent=2)+'\n')
    # Render values for inspection; clipping is display-only, not a model output transform.
    rectangles=[]
    for panel, tensor in enumerate((images[0],samples[0])):
        rgb=((tensor.detach().cpu().clamp(-1,1)+1)*127.5).round().to(torch.int64)
        for y in range(16):
            for x in range(16):
                r,g,b=rgb[:,y,x].tolist()
                rectangles.append(f'<rect x="{panel*144+x*8}" y="{y*8+24}" width="8" height="8" fill="rgb({r},{g},{b})"/>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="280" height="160"><text x="0" y="16">input</text><text x="144" y="16">toy sample</text>'+''.join(rectangles)+'</svg>'
    (destination/'input_sample.svg').write_text(svg)


def run_demo(implementation='reference',steps=4,system=None,output_dir=None):
    torch.manual_seed(43)
    if system is None:
        if implementation=='practice': from .practice_repa import build_toy_repa
        else: from .reference_repa import build_toy_repa
        system=build_toy_repa()
    records=exercise_training(system,steps)
    _,labels=toy_batch()
    noise=torch.randn(4,4,8,8)
    a=system.sample(labels,noise,8); b=system.sample(labels,noise,8)
    torch.testing.assert_close(a,b)
    assert a.shape==(4,3,16,16) and torch.isfinite(a).all()
    if output_dir is not None:
        write_diagnostics(output_dir,records,toy_batch()[0],a)
    print(f'REPA CPU {implementation}: {steps} updates; gradients/EMA/teacher-free Euler passed')
    print(records[-1])
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation',choices=('reference','practice'),default='reference')
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    try: run_demo(a.implementation,a.steps,output_dir=a.output_dir)
    except NotImplementedError as exc: print(exc); return 2
    return 0


if __name__=='__main__': raise SystemExit(main())
