"""Isolated semantic checks. Selected lesson is student code; peripherals are supplied."""
import argparse
import importlib
import torch
import reference_pi0 as reference
from lesson_registry import GROUPS


def selected_ops(number, implementation):
    ops = reference.reference_ops()
    if implementation == "practice":
        module, names = GROUPS[number-1]
        lesson = importlib.import_module(module)
        for name in names:
            setattr(ops, name, getattr(lesson, name))
    return ops


def compare_joint_cached(model):
    """Compare outputs AND gradients with one-sided block routing and padded prefix."""
    prefix = torch.randn(2, 5, 64, requires_grad=True)
    suffix = torch.randn(2, 9, 32, requires_grad=True)
    valid = torch.tensor([[True]*5, [True, True, False, False, False]])
    full = model.ops.joint_forward(model, prefix, valid, suffix)
    cache = model.ops.cache_prefix(model, prefix, valid)
    saved = [(k.detach().clone(), v.detach().clone()) for k, v in cache.key_values]
    cached = model.ops.cached_forward(model, cache, suffix)
    torch.testing.assert_close(full, cached, rtol=2e-5, atol=2e-6)
    parameters = [prefix, suffix, model.backbone.layers[0].k_proj.weight, model.expert_layers[0].q_proj.weight]
    g1 = torch.autograd.grad(full.square().mean(), parameters)
    g2 = torch.autograd.grad(cached.square().mean(), parameters)
    for first, second in zip(g1, g2):
        torch.testing.assert_close(first, second, rtol=1e-4, atol=2e-6)
    assert g2[0][1, 2:].eq(0).all(), "padded prefix must have zero influence"
    with torch.no_grad():
        changed = model.ops.cached_forward(model, cache, suffix+1)
        assert not torch.allclose(changed, cached)
    for (k, v), (old_k, old_v) in zip(cache.key_values, saved):
        torch.testing.assert_close(k, old_k, rtol=0, atol=0)
        torch.testing.assert_close(v, old_v, rtol=0, atol=0)


def check(number, ops):
    torch.manual_seed(21)
    if number == 1:
        assert ops.action_chunk(torch.arange(3), 1, 4).tolist() == [1, 2, 2, 2]
        x = torch.tensor([[0.], [5.], [10.]])
        low, high = torch.tensor([0.]), torch.tensor([10.])
        expected = torch.tensor([[-1.], [0.], [1.]])
        torch.testing.assert_close(ops.normalize_values(x, low, high), expected)
        torch.testing.assert_close(ops.unnormalize_values(expected, low, high), x)
        assert torch.isfinite(ops.normalize_values(x, low, low)).all()
    elif number == 2:
        model = reference.Pi0Policy(ops=ops)
        images = torch.rand(2, 2, 3, 8, 8)
        ids = torch.tensor([[1, 8], [1, 0]])
        camera = torch.tensor([[True, True], [True, False]])
        prefix, valid = ops.embed_prefix(model, images, camera, ids, ids != 0)
        assert prefix.shape == (2, 10, 64)
        assert valid.sum(1).tolist() == [10, 5]
        torch.testing.assert_close(prefix[:, :4], model.backbone.encode_vision(images[:, 0], projected=True).tokens)
        embed = ops.time_embedding(torch.tensor([0., 1.]), 32)
        torch.testing.assert_close(embed[0, :16], torch.zeros(16))
        torch.testing.assert_close(embed[0, 16:], torch.ones(16))
        suffix = ops.embed_suffix(model, torch.randn(2, 7), torch.randn(2, 8, 7), torch.tensor([.2, .8]))
        assert suffix.shape == (2, 9, 32)
        suffix.sum().backward()
        assert model.state_proj.weight.grad.abs().sum() > 0
        assert model.action_time[0].weight.grad.abs().sum() > 0
    elif number == 3:
        valid = torch.ones(1, 4, dtype=torch.bool)
        expected = torch.tensor([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=torch.bool)
        assert torch.equal(ops.block_mask(valid, 1), expected)
        valid[0, 0] = False
        mask = ops.block_mask(valid, 1)
        assert not mask[0, 0].any() and not mask[:, :, 0].any()
        assert ops.position_ids(torch.tensor([[True, True, False, True]])).tolist() == [[0, 1, 1, 2]]
        from run_pi0_demo import make_batch
        model = reference.Pi0Policy(ops=ops)
        batch = make_batch(model)
        batch["camera_valid"][:, 1] = False
        noise, times = torch.randn_like(batch["actions"]), torch.tensor([.2, .8])
        initial_loss = model(batch, noise, times)
        changed = dict(batch)
        changed["images"] = batch["images"].clone()
        changed["images"][:, 1] = torch.rand_like(changed["images"][:, 1])*10
        changed["text_ids"] = batch["text_ids"].clone()
        changed["text_ids"][1, -1] = 31  # still masked padding
        torch.testing.assert_close(model(changed, noise, times), initial_loss)
    elif number == 4:
        model = reference.Pi0Policy(ops=ops)
        compare_joint_cached(model)
        p, s = torch.randn(1, 3, 64), torch.randn(1, 9, 32)
        valid = torch.ones(1, 12, dtype=torch.bool)
        positions = ops.position_ids(valid)
        mask = ops.block_mask(valid, 3)
        first_p, first_s = ops.joint_attention(model.backbone.layers[0], model.expert_layers[0], p, s, positions, mask)
        changed = s.clone(); changed[:, 1:] += 10
        other_p, other_s = ops.joint_attention(model.backbone.layers[0], model.expert_layers[0], p, changed, positions, mask)
        torch.testing.assert_close(first_p, other_p)
        torch.testing.assert_close(first_s[:, 0], other_s[:, 0])
        assert not torch.allclose(first_s[:, 1:], other_s[:, 1:])
    elif number == 5:
        times = ops.sample_time(1024, "cpu")
        assert times.shape == (1024,) and ((times >= .001) & (times <= 1)).all()
        assert .52 < times.mean() < .68  # Beta(1.5,1) mean=.6; not uniform noise level
        x, u = ops.flow_pair(torch.full((1, 2, 1), 2.), torch.zeros(1, 2, 1), torch.tensor([.25]))
        torch.testing.assert_close(x, torch.full_like(x, 1.5))
        torch.testing.assert_close(u, torch.full_like(u, -2.))
        torch.testing.assert_close(ops.flow_loss(torch.tensor([1., 3.]), torch.tensor([0., 1.])), torch.tensor(2.5))
    elif number == 6:
        from run_pi0_demo import run_demo
        run_demo(lambda: reference.Pi0Policy(ops=ops))
    elif number == 7:
        torch.testing.assert_close(ops.euler_step(torch.tensor([1.5]), torch.tensor([-2.]), -.25), torch.tensor([2.]))
        model = reference.Pi0Policy(ops=ops)
        compare_joint_cached(model)
        assert ops.execution_chunk(torch.arange(8)[None, :, None]).flatten().tolist() == [0, 1, 2]
        from run_pi0_demo import run_demo
        run_demo(lambda: reference.Pi0Policy(ops=ops))
        # Test-only injection counts prefix encoding, including fresh observations.
        from types import SimpleNamespace
        from run_pi0_demo import make_batch
        calls = []
        def observed_cache(*args):
            calls.append(1)
            return ops.cache_prefix(*args)
        monitored = SimpleNamespace(**{**vars(ops), "cache_prefix": observed_cache})
        model = reference.Pi0Policy(ops=monitored)
        batch = make_batch(model)
        noise = torch.randn_like(batch["actions"])
        first = model.predict_action(batch, noise)
        assert len(calls) == 1, "prefix KV must be computed once per planning round"
        changed = {**batch, "images": 1-batch["images"]}
        second = model.predict_action(changed, noise)
        assert len(calls) == 2 and not torch.allclose(first, second), "new observations need fresh KV"
    else:
        raise ValueError(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", type=int, choices=range(1, 8))
    parser.add_argument("--implementation", choices=("reference", "practice"), default="practice")
    args = parser.parse_args()
    torch.set_num_threads(2)
    failed = False
    for number in ([args.lesson] if args.lesson else range(1, 8)):
        try:
            check(number, selected_ops(number, args.implementation))
            print(f"pi0 lesson {number}: PASS ({args.implementation})")
        except NotImplementedError as error:
            print(f"pi0 lesson {number}: 未完成 {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
