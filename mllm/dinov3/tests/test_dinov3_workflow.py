"""Gram invariants, both teachers, checkpoint continuity and student call paths."""
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from mllm.dinov3 import reference_dinov3 as ref
from mllm.dinov3.check_lessons import check
from mllm.dinov3 import lesson1_gram as l1, lesson2_anchoring as l2, lesson3_teacher as l3
from mllm.dinov3 import lesson4_resolution as l4, lesson5_refinement as l5


@pytest.mark.parametrize('lesson',range(1,6))
def test_reference_numerical_contracts(lesson):check(lesson,'reference')


def test_all_practice_helpers_and_training_reached(monkeypatch):
    seen=set()
    def instrument(name,fn):
        def call(*args,**kwargs):seen.add(name);return fn(*args,**kwargs)
        return call
    replacements=[(l1,'gram_matrix',ref.gram_matrix),(l2,'gram_loss',ref.gram_loss),
                  (l3,'gram_event',ref.gram_event),(l3,'sync_gram_teacher',ref.sync_gram_teacher),
                  (l4,'aligned_crops',ref.aligned_crops),(l4,'downsample_features',ref.downsample_features),
                  (l5,'refinement_loss',ref.refinement_loss),(l5,'training_step',ref.training_step)]
    sources={Path(m.__file__):Path(m.__file__).read_bytes() for m,_,_ in replacements}
    for module,name,fn in replacements:monkeypatch.setattr(module,name,instrument(name,fn))
    for lesson in range(1,6):check(lesson,'practice')
    from mllm.dinov3.run_dinov3_demo import main
    main('practice')
    assert seen=={name for _,name,_ in replacements}
    assert all(p.read_bytes()==data for p,data in sources.items())


def test_anchor_gradient_goes_to_student_not_either_teacher():
    torch.manual_seed(3)
    model=ref.build_toy_dinov3();model.after_step(2)
    crops,masks,high=model.views(torch.rand(3,3,80,80))
    losses=model.losses(crops,masks,4,high_crops=high)
    losses['gram'].backward()
    assert model.student['backbone'].patch_embed.proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.gram_teacher.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())


def test_checkpoint_restores_phase_and_next_update():
    torch.manual_seed(5)
    model=ref.build_toy_dinov3();optimizer=torch.optim.AdamW(model.student.parameters(),lr=.001)
    images=torch.rand(3,3,80,80)
    for step in range(5):
        crops,masks,high=model.views(images)
        model.ops.training_step(model,crops,masks,high,optimizer,step,8)
    buffer=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},buffer)
    buffer.seek(0);saved=torch.load(buffer,weights_only=True)
    restored=ref.build_toy_dinov3(snapshot_step=1,refine_start=2,refresh_every=9)
    restored.load_state_dict(saved['model'])
    opt2=torch.optim.AdamW(restored.student.parameters(),lr=.001);opt2.load_state_dict(saved['optimizer'])
    assert (restored.snapshot_step,restored.refine_start,restored.refresh_every)==(2,4,2)
    assert int(restored.completed_steps)==5 and bool(restored.gram_ready)
    for target,opt in [(model,optimizer),(restored,opt2)]:
        torch.set_rng_state(saved['rng']);crops,masks,high=target.views(images)
        target.ops.training_step(target,crops,masks,high,opt,5,8)
    for k,value in model.state_dict().items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,restored.state_dict()[k],rtol=0,atol=0)
    assert int(restored.last_gram_update)==6


def test_bad_feature_mse_is_rejected(monkeypatch):
    monkeypatch.setattr(l1,'gram_matrix',lambda x:x@x.transpose(-1,-2))
    with pytest.raises(AssertionError):check(1,'practice')


def test_unfinished_cli_locates_todo():
    root=Path(__file__).resolve().parents[3];env=os.environ.copy()
    env['PYTHONPATH']=str(root)+os.pathsep+str(root/'general_utils/edu_core')
    for module,extra,expected in [('check_lessons',['--lesson','1'],'lesson1.gram_matrix'),
                                   ('run_dinov3_demo',[],'lesson4.aligned_crops')]:
        p=subprocess.run([sys.executable,'-m',f'mllm.dinov3.{module}','--implementation','practice',*extra],cwd=root,env=env,capture_output=True,text=True)
        assert p.returncode==2 and expected in p.stderr
