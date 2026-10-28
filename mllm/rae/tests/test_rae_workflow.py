"""维护验收：测试替身只存在测试进程，不写入学生练习。"""
import importlib
from collections import Counter
from types import SimpleNamespace
import pytest
import torch
from mllm.rae import reference_rae as ref
from mllm.rae.check_lessons import check
from mllm.rae.lesson_registry import LESSONS
from mllm.rae.practice_rae import build_toy_rae
from mllm.rae.run_rae_demo import run


@pytest.mark.parametrize('number',range(1,9))
def test_reference_contract(number):
    check(number,'reference')


def install_substitutes(monkeypatch):
    calls=Counter()
    for module_name,names in LESSONS:
        module=importlib.import_module('mllm.rae.'+module_name)
        for name in names:
            def counted(*args,_name=name,**kwargs):
                calls[_name]+=1
                return getattr(ref,_name)(*args,**kwargs)
            monkeypatch.setattr(module,name,counted)
    return calls


def test_all_practice_helpers_reached(monkeypatch):
    calls=install_substitutes(monkeypatch)
    run('practice',model=build_toy_rae())
    assert set(ref.reference_ops().__dict__)<=set(calls)
    for number in range(1,9): check(number,'practice')


def test_todo_propagates_without_fallback(monkeypatch):
    install_substitutes(monkeypatch)
    module=importlib.import_module('mllm.rae.lesson5_flow')
    def unfinished(*args,**kwargs): raise NotImplementedError('injected flow TODO')
    monkeypatch.setattr(module,'velocity_loss',unfinished)
    with pytest.raises(NotImplementedError,match='injected flow TODO'):
        run('practice',model=build_toy_rae())


def test_semantic_error_propagates(monkeypatch):
    install_substitutes(monkeypatch)
    module=importlib.import_module('mllm.rae.lesson6_ddt')
    def invalid(*args,**kwargs): raise ValueError('injected real failure')
    monkeypatch.setattr(module,'ddt_forward',invalid)
    with pytest.raises(ValueError,match='injected real failure'):
        run('practice',model=build_toy_rae())


@pytest.mark.parametrize('name,lesson', [('reconstruction_loss',4),('velocity_loss',5)])
def test_constant_losses_rejected(monkeypatch,name,lesson):
    install_substitutes(monkeypatch)
    module=importlib.import_module('mllm.rae.'+LESSONS[lesson-1][0])
    monkeypatch.setattr(module,name,lambda *args,**kwargs: torch.tensor(1.))
    with pytest.raises(AssertionError): check(lesson,'practice')


def test_encoder_per_token_not_batch_normalization():
    ops=ref.reference_ops()
    x=torch.randn(2,18,64)
    result=ops.normalize_tokens(x)
    torch.testing.assert_close(result.mean(-1),torch.zeros(2,16),atol=1e-6,rtol=0)
    changed=x.clone();changed[1]*=100
    torch.testing.assert_close(result[0],ops.normalize_tokens(changed)[0])


def test_stage1_eval_no_noise_and_stage2_train_contract():
    model=ref.build_toy_rae();images=torch.randn(2,3,16,16)
    model.rae.eval()
    torch.testing.assert_close(model.rae(images),model.rae(images))
    model.set_stage(2);model.train()
    assert not model.rae.training and not model.rae.encoder.training and not model.ema.training
    assert not any(p.requires_grad for p in model.rae.parameters())
    assert all(p.requires_grad for p in model.generator.parameters())


def test_noise_amplitude_is_shared_per_sample_and_can_exceed_tau():
    z=torch.zeros(2,16,64)
    got=ref.augment_latent(z,.8,torch.tensor([-2.,.5]),torch.ones_like(z))
    torch.testing.assert_close(got[:,0,0],torch.tensor([1.6,.4]))
    assert torch.equal(got,got[:,:1,:1].expand_as(got))


def test_head_has_direct_noise_route_and_tokenwise_condition():
    torch.manual_seed(4)
    model=ref.build_toy_rae().generator
    # nonzero initialization avoids trivial zero-output equality.
    with torch.no_grad():
        for p in model.parameters():p.add_(torch.randn_like(p)*.05)
    noisy=torch.randn(2,16,64)
    seen={}
    h1=model.noisy_embed.register_forward_pre_hook(lambda module,args:seen.update(noisy=args[0]))
    h2=model.head[0].register_forward_pre_hook(lambda module,args:seen.update(condition=args[1]))
    result=model(noisy,torch.tensor([.2,.7]));h1.remove();h2.remove()
    assert seen['noisy'] is noisy
    assert seen['condition'].shape==(2,16,96)
    assert not torch.equal(seen['condition'][:,0],seen['condition'][:,1])
    assert torch.isfinite(result).all()


def test_shifted_euler_uses_grid_deltas():
    # velocity=t: shifted steps 1 -> 1/3 -> 0, left Euler gives -(2/3+1/9).
    class TimeVelocity:
        ops=ref.reference_ops()
        def __call__(self,x,t):return t[:,None,None].expand_as(x)
    initial=torch.zeros(1,16,64)
    got=ref.euler_sample(TimeVelocity(),initial,2)
    torch.testing.assert_close(got,torch.full_like(got,-7/9))


def test_cli_reports_injected_todo_and_nonzero(monkeypatch,capsys):
    install_substitutes(monkeypatch)
    lesson=importlib.import_module('mllm.rae.lesson1_encoding')
    def todo(*args,**kwargs):raise NotImplementedError('injected normalization TODO')
    monkeypatch.setattr(lesson,'normalize_tokens',todo)
    monkeypatch.setattr('sys.argv',['check_lessons','--lesson','1'])
    from mllm.rae.check_lessons import main
    assert main()==2
    assert 'injected normalization TODO' in capsys.readouterr().out


def test_recipe_metadata_rejects_main_noise_and_wrong_time():
    import json
    from pathlib import Path
    from mllm.rae.recipe.validate_config import validate
    config=json.loads((Path(__file__).parents[1]/'recipe/config.json').read_text())
    assert validate(config)['metadata_valid']
    for key,value in [('decoder_noise_distribution','uniform'),('time_direction','noise_0_data_1'),('reference_dim',64)]:
        with pytest.raises(ValueError):validate({**config,key:value})


def test_diagnostics_artifacts_are_structurally_valid(tmp_path):
    import json
    run(output_dir=tmp_path)
    data=json.loads((tmp_path/'losses.json').read_text())
    assert len(data['stage1_l1'])==len(data['stage2_flow'])==5
    assert data['paper_metrics_reproduced'] is False
    for name in ('input','reconstruction','generated'):
        payload=(tmp_path/(name+'.ppm')).read_bytes()
        header=b'P6\n16 16\n255\n'
        assert payload.startswith(header) and len(payload)==len(header)+16*16*3
