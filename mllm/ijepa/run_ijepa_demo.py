"""CPU smoke test for the I-JEPA reference solution."""
import torch
from edu_core.training import seed_everything
from reference_ijepa import IJEPA, sample_block_masks


def main() -> None:
    seed_everything(7)
    model = IJEPA(image_size=32, patch_size=8, dim=48, depth=1, heads=4, predictor_dim=64, predictor_depth=1)
    masks = sample_block_masks((4, 4), num_targets=2, block_size=(2, 2), generator=torch.Generator().manual_seed(3))
    images = torch.randn(2, 3, 32, 32)
    model.train()
    assert not model.target_encoder.training and not any(p.requires_grad for p in model.target_encoder.parameters())
    optimizer = torch.optim.AdamW(list(model.context_encoder.parameters()) + list(model.predictor.parameters()) + list(model.target_pos_proj.parameters()), lr=1e-3)
    before = next(model.target_encoder.parameters()).detach().clone()
    out = model(images, masks)
    assert torch.isfinite(out["loss"]) and not torch.equal(out["context_indices"], out["target_indices"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    optimizer.step(); model.update_target_encoder(0.9)
    assert not torch.equal(before, next(model.target_encoder.parameters()))
    model.eval()
    with torch.no_grad():
        a, b = model(images, masks)["loss"], model(images, masks)["loss"]
    assert torch.equal(a, b)
    print(f"I-JEPA CPU demo passed; loss={out['loss'].item():.4f}, context={len(out['context_indices'])}, targets={len(out['target_indices'])}")


if __name__ == "__main__":
    main()
