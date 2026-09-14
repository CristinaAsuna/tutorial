"""在随机短视频上跑一次 V-JEPA CPU 训练步。"""
import torch

from edu_core.training import seed_everything
from reference_vjepa import LatentPredictor, MiniVideoViT, VJEPA, sample_spatiotemporal_masks


def main() -> None:
    seed_everything(17)
    videos = torch.rand(2, 3, 8, 32, 32)
    encoder = MiniVideoViT(embed_dim=32, depth=1, num_heads=4)
    model = VJEPA(encoder, LatentPredictor(32, predictor_dim=24, depth=1, num_heads=4))
    target, context = sample_spatiotemporal_masks(2, encoder.base_grid, (1, 2, 2), 2,
                                                   generator=torch.Generator().manual_seed(9))
    out = model(videos, target, context)
    assert out["predictions"].shape == out["targets"].shape == (2, 8, 32)
    assert torch.isfinite(out["loss"])
    optim = torch.optim.AdamW(list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=1e-3)
    teacher_before = next(model.target_encoder.parameters()).detach().clone()
    student_before = next(model.context_encoder.parameters()).detach().clone()
    optim.zero_grad(); out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    optim.step()
    assert not torch.equal(student_before, next(model.context_encoder.parameters()))
    model.update_target(0.9)
    assert not torch.equal(teacher_before, next(model.target_encoder.parameters()))
    assert not model.target_encoder.training and not any(p.requires_grad for p in model.target_encoder.parameters())
    model.eval()
    with torch.no_grad():
        a = model(videos, target, context)["loss"]
        b = model(videos, target, context)["loss"]
    assert torch.equal(a, b)
    print("V-JEPA CPU demo passed", {"loss": round(float(out["loss"].detach()), 5), "grid": encoder.base_grid})


if __name__ == "__main__":
    main()
