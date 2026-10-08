"""CPU checks for the ACT teaching implementation."""
import math
import argparse
import torch

from reference_act import TemporalEnsembler, build_toy_act, sample_action_chunks
from edu_core.training import seed_everything


def main(implementation="reference") -> None:
    if implementation == "practice":
        from practice_act import TemporalEnsembler, build_toy_act, sample_action_chunks, training_step
    else:
        from reference_act import TemporalEnsembler, build_toy_act, sample_action_chunks
        def training_step(model, optimizer, images, qpos, chunks, valid, beta=.01):
            optimizer.zero_grad()
            out=model(images,qpos,chunks,valid,beta=beta)
            out["loss"].backward(); optimizer.step()
            return out
    seed_everything(41)
    trajectories = torch.randn(2, 7, 4)
    all_chunks, all_valid = sample_action_chunks(trajectories, chunk_size=5)
    assert all_chunks.shape == (2, 7, 5, 4) and all_valid.shape == (2, 7, 5)
    assert torch.equal(all_valid[0, -1], torch.tensor([True, False, False, False, False]))
    chunks, valid = all_chunks[:, 3], all_valid[:, 3]
    images, qpos = torch.randn(2, 2, 3, 8, 8), torch.randn(2, 4)
    model = build_toy_act().train()

    torch.manual_seed(5)
    out = model(images, qpos, chunks, valid, beta=0.01)
    assert out["actions"].shape == chunks.shape and torch.isfinite(out["loss"])
    assert out["mean"].shape == out["logvar"].shape == (2, 8)

    # Invalid trailing target values cannot affect posterior or reconstruction.
    padded_changed = chunks.clone(); padded_changed[~valid] = 10_000
    torch.manual_seed(5)
    changed = model(images, qpos, padded_changed, valid, beta=0.01)
    assert torch.equal(out["loss"], changed["loss"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(); out["loss"].backward()
    assert any(p.grad is not None for p in model.style_encoder.parameters())
    assert any(p.grad is not None for p in model.action_decoder.parameters())
    before = model.action_head.weight.detach().clone(); optimizer.step()
    assert not torch.equal(before, model.action_head.weight)

    assert torch.isfinite(training_step(model,optimizer,images,qpos,chunks,valid)["loss"])

    chunk_a, chunk_b = model.predict_chunk(images, qpos), model.predict_chunk(images, qpos)
    assert torch.equal(chunk_a, chunk_b), "ACT inference must use deterministic z=0"

    ensemble = TemporalEnsembler(chunk_size=3, action_dim=1, decay=math.log(2.0))
    assert torch.equal(ensemble.add(torch.tensor([[1.0], [2.0], [3.0]])), torch.tensor([1.0]))
    # At the second query, the old chunk's offset-1 prediction (2) and new
    # chunk's offset-0 prediction (10) describe the same current timestep.
    combined = ensemble.add(torch.tensor([[10.0], [20.0], [30.0]]))
    assert torch.allclose(combined, torch.tensor([(2.0 + 10.0 * 0.5) / 1.5]))
    print(f"ACT CPU demo passed; loss={out['loss'].item():.4f}, chunk={tuple(out['actions'].shape)}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--implementation",choices=("reference","practice"),default="reference")
    main(parser.parse_args().implementation)
