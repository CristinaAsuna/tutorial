"""练习组装：所有论文目标、增广、ViT 与 EMA 经过学生函数。"""
import torch
from torch import nn
import lesson1_vit as l1
import lesson2_multicrop_masking as l2
import lesson3_dino_loss as l3
import lesson4_ibot_teacher_student as l4
from reference_dinov2 import DINOHead, TeacherStudentDINO as BaseModel, DINOiBOTLoss as BaseLoss

MiniViT = l1.ExerciseViT

class TeacherStudentDINO(BaseModel):
    @torch.no_grad()
    def update_teacher(self, momentum):
        l4.ema_update(self.teacher_parameters(), list(self.student_backbone.parameters()) + list(self.student_head.parameters()), momentum)
        self._freeze_teacher()

class DINOiBOTLoss(BaseLoss):
    def forward(self, student_outputs, teacher_outputs, global_masks, step):
        temp = self.current_teacher_temp(step)
        dino = l3.dino_cross_view_loss([s['cls'] for s in student_outputs], [t['cls'] for t in teacher_outputs], self.center, self.student_temp, temp)
        # Weight crop means by their mask counts: the denominator is all masked patches.
        counts = [int(m.sum()) for m in global_masks]
        ibot = sum(l4.ibot_masked_patch_loss(student_outputs[i]['patch'], teacher_outputs[i]['patch'], mask, self.patch_center, self.student_temp, temp) * counts[i] for i, mask in enumerate(global_masks)) / sum(counts)
        with torch.no_grad():
            self.center.copy_(l3.update_center(self.center, [t['cls'] for t in teacher_outputs], self.center_momentum))
            self.patch_center.copy_(l3.update_center(self.patch_center, [t['patch'].flatten(0, 1) for t in teacher_outputs], self.center_momentum))
        loss = dino + self.ibot_weight * ibot
        return {'loss': loss, 'dino_loss': dino.detach(), 'ibot_loss': ibot.detach(), 'teacher_temp': loss.new_tensor(temp)}

def make_views_and_masks(images):
    crops = l2.make_multicrop_views(images)
    masks = [l2.random_patch_mask(c.shape[0], (c.shape[-2]//8)*(c.shape[-1]//8), .5, c.device) for c in crops[:2]]
    return crops, masks + [None] * (len(crops)-2)


def build_toy_dino():
    """返回小型 TeacherStudentDINO；criterion 单独构造 DINOiBOTLoss(32)。"""
    backbone=MiniViT(image_size=32,patch_size=8,embed_dim=48,depth=2,num_heads=4)
    return TeacherStudentDINO(backbone,DINOHead(48,out_dim=32,hidden_dim=64,bottleneck_dim=24))
