"""One fast CPU training step for the reference DINO + iBOT implementation."""
import torch
from edu_core.training import seed_everything

from multicrop import MultiCropAugmentation, make_patch_masks
from reference_dinov2 import DINOHead, DINOiBOTLoss, MiniViT, TeacherStudentDINO


def main() -> None:
    seed_everything(7)
    images = torch.rand(2, 3, 40, 40)
    crops = MultiCropAugmentation(global_size=32, local_size=16)(images)
    masks = make_patch_masks(crops[:2], patch_size=8, mask_ratio=0.5)
    masks += [None] * 4
    backbone = MiniViT(image_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=4)
    model = TeacherStudentDINO(backbone, DINOHead(48, out_dim=32, hidden_dim=64, bottleneck_dim=24))
    criterion = DINOiBOTLoss(32, warmup_steps=2)
    optimizer = torch.optim.AdamW(list(model.student_backbone.parameters()) + list(model.student_head.parameters()), lr=1e-3)

    teacher_before = next(model.teacher_backbone.parameters()).detach().clone()
    student_before = next(model.student_backbone.parameters()).detach().clone()
    teacher_outputs = model.teacher_forward(crops[:2])
    outputs = model.student_forward(crops, masks)
    result = criterion(outputs, teacher_outputs, masks[:2], step=0)
    assert torch.isfinite(result["loss"])
    optimizer.zero_grad()
    result["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.student_backbone.parameters())
    optimizer.step()
    assert not torch.equal(student_before, next(model.student_backbone.parameters()))
    model.update_teacher(momentum=0.9)
    assert not torch.equal(teacher_before, next(model.teacher_backbone.parameters()))
    assert not model.teacher_backbone.training and not any(p.requires_grad for p in model.teacher_parameters())
    assert criterion.center.abs().sum() > 0 and criterion.patch_center.abs().sum() > 0
    print("DINO+iBOT CPU demo passed")
    print({key: round(float(value), 4) for key, value in result.items()})


if __name__ == "__main__":
    main()
