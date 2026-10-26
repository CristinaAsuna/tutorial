"""Numerical oracles, explicit practice wiring, TODO/CLI failure propagation."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from edu_core.training import freeze_and_keep_eval
from mllm.dinov2 import reference_dinov2 as ref
from mllm.dinov2.check_lessons import check
from mllm.dinov2 import lesson1_masking as l1, lesson2_heads as l2, lesson3_sinkhorn as l3
from mllm.dinov2 import lesson4_ibot as l4, lesson5_koleo as l5, lesson6_training as l6


@pytest.mark.parametrize('lesson',range(1,7))
def test_reference_numerical_contracts(lesson):check(lesson,'reference')


def test_all_practice_helpers_and_training_reached(monkeypatch):
    seen=set()
    def instrument(name,fn):
        def call(*args,**kwargs):seen.add(name);return fn(*args,**kwargs)
        return call
    def init(s,t):t.load_state_dict(s.state_dict());freeze_and_keep_eval(t)
    replacements=[(l1,'rectangle_mask',ref.rectangle_mask),(l1,'mask_count',ref.mask_count),
                  (l2,'choose_patch_head',lambda cls,candidate,shared=False:cls if shared else candidate),
                  (l2,'initialize_teacher',init),(l3,'sinkhorn',ref.sinkhorn),
                  (l4,'ibot_loss',ref.ibot_loss),(l5,'koleo',ref.koleo),
                  (l6,'teacher_temperature',ref.teacher_temperature),(l6,'ema_momentum',ref.ema_momentum),
                  (l6,'training_step',ref.training_step)]
    sources={Path(module.__file__):Path(module.__file__).read_bytes() for module,_,_ in replacements}
    for module,name,fn in replacements:monkeypatch.setattr(module,name,instrument(name,fn))
    for lesson in range(1,7):check(lesson,'practice')
    from mllm.dinov2.run_dinov2_demo import main
    main('practice')
    assert seen=={name for _,name,_ in replacements}
    assert all(path.read_bytes()==data for path,data in sources.items())


def test_mask_extremes_and_patch_gradient():
    assert not ref.sample_masks(3,(4,4),'cpu',probability=0).any()
    assert ref.sample_masks(3,(4,4),'cpu',probability=1,ratio_range=(1,1)).all()
    torch.manual_seed(7)
    model=ref.build_toy_dinov2()
    crops=[torch.rand(3,3,32,32),torch.rand(3,3,32,32)]
    mask=[torch.zeros(3,16,dtype=torch.bool) for _ in range(2)]
    out=model.losses(crops,mask,0)
    assert out['ibot']==0 and torch.isfinite(out['loss'])
    out['loss'].backward()
    assert all(p.grad is None for p in model.teacher.parameters())


def test_bad_sinkhorn_answer_fails_semantic_check(monkeypatch):
    monkeypatch.setattr(l3,'sinkhorn',lambda logits,temperature=.07,iterations=3:logits.detach().softmax(-1))
    with pytest.raises(AssertionError):check(3,'practice')


def test_unfinished_practice_cli_is_nonzero():
    root=Path(__file__).resolve().parents[3]
    env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+str(root/'general_utils/edu_core')
    for module,extra,expected in [('check_lessons',['--lesson','1'],'lesson1.rectangle_mask'),
                                   ('run_dinov2_demo',[],'lesson2.choose_patch_head')]:
        result=subprocess.run([sys.executable,'-m',f'mllm.dinov2.{module}','--implementation','practice',*extra],
                              cwd=root,env=env,text=True,capture_output=True)
        assert result.returncode==2 and expected in result.stderr
