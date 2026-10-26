"""Isolated semantic checks: only selected lesson uses student code; scaffolding is supplied."""
import argparse
import importlib
import torch
from torch import nn
from types import SimpleNamespace
import reference_openvla as reference
from lesson_registry import GROUPS


def selected_ops(number, implementation):
    ops = reference.reference_ops()
    if implementation == "practice":
        module, names = GROUPS[number-1]
        lesson = importlib.import_module(module)
        for name in names:
            setattr(ops, name, getattr(lesson, name))
    return ops


def check(number, ops):
    torch.manual_seed(11)
    if number == 1:
        stats = ops.fit_action_stats(torch.arange(101.).reshape(-1, 1), torch.tensor([True]))
        torch.testing.assert_close(stats["q01"], torch.tensor([1.]))
        torch.testing.assert_close(stats["q99"], torch.tensor([99.]))
        stats = {"q01": torch.tensor([0., 0.]), "q99": torch.tensor([10., 1.]), "mask": torch.tensor([True, False])}
        x = torch.tensor([[0., .2], [5., .6], [10., 1.]])
        expected = torch.tensor([[-1., .2], [0., .6], [1., 1.]])
        torch.testing.assert_close(ops.normalize_actions(x, stats), expected)
        torch.testing.assert_close(ops.unnormalize_actions(expected, stats), x)
        assert torch.isfinite(ops.normalize_actions(x, {**stats, "q99": torch.zeros(2)})).all()
    elif number == 2:
        values = torch.tensor([-9., -1., 0., 1., 9.])
        ids = ops.encode_actions(values, 20, 4)
        assert ids.tolist() == [19, 19, 18, 16, 16]
        decoded = ops.decode_actions(torch.tensor([19, 18, 17, 16, 0, 999]), 20, 4)
        torch.testing.assert_close(decoded, torch.tensor([-2/3, 0., 2/3, 2/3, 2/3, -2/3]), atol=1e-6, rtol=0)
        boundary = torch.arange(256, dtype=torch.float64)*(2/255)-1
        encoded = ops.encode_actions(boundary)
        assert encoded[0] == 319 and encoded[-1] == 64
        assert encoded.unique().numel() == 256
        # Scalar bisect oracle reproduces np.digitize over float64 edges without NumPy dependency.
        from bisect import bisect_right
        values = torch.linspace(-1, 1, 256, dtype=torch.float32)
        expected = torch.tensor([320-bisect_right(boundary.tolist(), float(v)) for v in values])
        torch.testing.assert_close(ops.encode_actions(values), expected)
    elif number == 3:
        a = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
        b = torch.tensor([[[5.], [6.]]], requires_grad=True)
        output = ops.fuse_features(a, b, nn.Identity())
        torch.testing.assert_close(output, torch.tensor([[[1., 2., 5.], [3., 4., 6.]]]))
        output.sum().backward()
        assert a.grad.eq(1).all() and b.grad.eq(1).all()
    elif number == 4:
        labels = ops.action_labels(torch.tensor([1, 8, 64, 65, 2]), 2)
        assert labels.tolist() == [-100, -100, 64, 65, 2]
        assert ops.action_labels(torch.tensor([1, 8, 64, 65, 2]), 2, False).tolist()[-1] == -100
        text, visual = torch.tensor([[1.], [7.], [8.]]), torch.tensor([[90.], [91.]])
        embeds, expanded = ops.pack_single(text, visual, torch.tensor([-100, -100, 8]))
        assert embeds.flatten().tolist() == [1., 90., 91., 7., 8.]
        assert expanded.tolist() == [-100, -100, -100, -100, 8]
        model = reference.OpenVLAPolicy(ops)
        rows = [torch.tensor([1, 8, 9]), torch.tensor([1, 8])]
        _, mask, targets = model.pack(rows, torch.zeros(2, 4, 32))
        assert mask.sum(1).tolist() == [7, 6] and targets.eq(-100).all()
        embeds = torch.randn(2, 7, 32)
        logits = model.language(embeds, mask)
        changed = embeds.clone()
        changed[:, 4:] += 100
        torch.testing.assert_close(model.language(changed, mask)[:, :4], logits[:, :4])
        # Masked padding must not affect any active token, even if its embedding changes.
        changed = embeds.clone()
        changed[1, 6] += 100
        torch.testing.assert_close(model.language(changed, mask)[1, :6], logits[1, :6])
    elif number == 5:
        logits = torch.tensor([[[0., 2.], [2., 0.], [0., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, 1, 0]])
        loss = ops.shifted_loss(logits, labels)
        torch.testing.assert_close(loss, torch.log1p(torch.exp(torch.tensor(-2.))))
        loss.backward()
        assert logits.grad[:, -1].eq(0).all()
        from run_openvla_demo import run_demo
        run_demo(lambda: reference.OpenVLAPolicy(ops))
    elif number == 6:
        # Scripted LM distinguishes each row's final valid position and conditions on previous token.
        class Scripted:
            action_dim, vocab_size = 2, 20
            def encode_images(self, images):
                return images
            def pack(self, rows, visual):
                length = max(map(len, rows))
                x = torch.zeros(len(rows), length, 1)
                mask = torch.zeros(len(rows), length, dtype=torch.bool)
                for i, row in enumerate(rows):
                    x[i, :len(row), 0], mask[i, :len(row)] = row, True
                return x, mask, None
            def language(self, x, mask):
                ids = (x[..., 0].long()+1) % 20
                return torch.nn.functional.one_hot(ids, 20).float()
        fake = Scripted()
        prompts = [torch.tensor([1, 3]), torch.tensor([1, 2, 7])]
        generated = ops.generate_action_ids(fake, torch.zeros(2, 1), prompts)
        assert generated.tolist() == [[4, 5], [8, 9]], generated
        fake.ops = SimpleNamespace(generate_action_ids=lambda *args: torch.tensor([[19, 18]]),
                                   decode_actions=lambda ids, vocab: torch.tensor([[-1., 1.]]),
                                   unnormalize_actions=reference.unnormalize_actions)
        stats = {"q01": torch.tensor([0., 10.]), "q99": torch.tensor([2., 20.]), "mask": torch.tensor([True, True])}
        torch.testing.assert_close(ops.predict_action(fake, None, None, stats), torch.tensor([[0., 20.]]))
    else:
        raise ValueError(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", type=int, choices=range(1, 7))
    parser.add_argument("--implementation", choices=("reference", "practice"), default="practice")
    args = parser.parse_args()
    torch.set_num_threads(1)
    failed = False
    for number in ([args.lesson] if args.lesson else range(1, 7)):
        try:
            check(number, selected_ops(number, args.implementation))
            print(f"lesson {number}: PASS ({args.implementation})")
        except NotImplementedError as error:
            print(f"lesson {number}: 未完成 {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
