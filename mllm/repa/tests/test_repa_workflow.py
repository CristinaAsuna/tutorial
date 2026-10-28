"""Maintenance tests use substitutes; they never overwrite student answers."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest
import torch
from mllm.repa import reference_repa as ref
from mllm.repa.check_lessons import check_lesson
from mllm.repa.run_repa_demo import run_demo
from mllm.repa.scaffolding import RepaConfig


@pytest.mark.parametrize('lesson',range(1,7))
def test_reference_numerical_contracts(lesson):
    check_lesson(lesson)


def test_all_practice_helpers_training_and_sampling_reached(monkeypatch):
    from mllm.repa import practice_repa as practice
    calls={}
    modules=[practice.l1,practice.l2,practice.l3,practice.l4,practice.l5,practice.l6]
    for module,functions in zip(modules,practice.LESSON_FUNCTIONS.values()):
        for name in functions:
            calls[name]=0
            def wrapper(*args,_name=name,**kwargs):
                calls[_name]+=1
                return getattr(ref,_name)(*args,**kwargs)
            monkeypatch.setattr(module,name,wrapper)
    run_demo('practice',system=practice.build_toy_repa())
    assert all(value>0 for value in calls.values()),calls


def test_wrong_constant_losses_rejected():
    ops=ref.reference_ops(); ops.alignment_loss=lambda *a:torch.tensor(1.)
    with pytest.raises(AssertionError): check_lesson(4,ops=ops)
    ops=ref.reference_ops()
    ops.joint_objective=lambda *a,**kw:{'loss':torch.tensor(1.),'velocity_loss':torch.tensor(1.),'alignment_loss':torch.tensor(1.)}
    with pytest.raises(AssertionError): check_lesson(4,ops=ops)


def test_cli_todo_specific_and_nonzero_independent_of_student_progress():
    # Inject an unfinished helper in a CHILD process rather than assuming current TODOs.
    source="""
import sys
from mllm.repa import lesson2_flow
def missing(*a,**kw): raise NotImplementedError('injected lesson2.flow_pair TODO')
lesson2_flow.flow_pair=missing
from mllm.repa.check_lessons import main
sys.argv=['check','--lesson','2','--implementation','practice']
raise SystemExit(main())
"""
    result=subprocess.run([sys.executable,'-c',source],capture_output=True,text=True)
    assert result.returncode==2 and 'injected lesson2.flow_pair TODO' in result.stdout


def test_real_errors_propagate(monkeypatch):
    from mllm.repa import lesson2_flow
    from mllm.repa.check_lessons import selected_ops
    def broken(*a,**kw): raise RuntimeError('semantic regression')
    monkeypatch.setattr(lesson2_flow,'flow_pair',broken)
    with pytest.raises(RuntimeError,match='semantic regression'):
        check_lesson(2,ops=selected_ops(2,'practice'))


def test_zero_initialization_and_depth_validation():
    model=ref.build_toy_repa().denoiser
    layers=[b.modulation[-1] for b in model.blocks]+[model.final_modulation[-1],model.output]
    assert all(not layer.weight.count_nonzero() and not layer.bias.count_nonzero() for layer in layers)
    with pytest.raises(ValueError): RepaConfig(alignment_depth=0)
    with pytest.raises(ValueError): RepaConfig(hidden_dim=7)


def test_alignment_gradient_does_not_reach_later_block():
    torch.manual_seed(5); system=ref.build_toy_repa()
    _,projected=system.denoiser(torch.randn(2,4,8,8),torch.tensor([.2,.8]),torch.tensor([0,1]))
    target=torch.randn_like(projected)
    ref.alignment_loss(projected,target).backward()
    assert system.denoiser.patch.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in system.denoiser.blocks[1].parameters())


def test_recipe_and_diagnostics(tmp_path):
    from mllm.repa.recipe.validate_config import validate_config
    path=Path(__file__).parents[1]/'recipe'/'config.json'
    config=json.loads(path.read_text()); config['data_path']=str(tmp_path/'missing')
    assert validate_config(config)['data_exists'] is False
    with pytest.raises(FileNotFoundError): validate_config(config,True)
    with pytest.raises(ValueError): validate_config({**config,'prediction':'clean-minus-noise'})
    run_demo(output_dir=tmp_path)
    data=json.loads((tmp_path/'losses.json').read_text())
    assert len(data['records'])==4 and data['paper_metrics_reproduced'] is False
    ET.parse(tmp_path/'input_sample.svg')
