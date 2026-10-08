"""全部练习接线和 teacher train/eval 合同回归。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
import lesson1_vit as l1
import lesson2_multicrop_masking as l2
import lesson3_dino_loss as l3
import lesson4_ibot_teacher_student as l4
from multicrop import MultiCropAugmentation
from edu_core.vision import interpolate_2d_pos_embed


def test_model_train_keeps_teacher_frozen_and_eval():
    from reference_dinov2 import MiniViT, DINOHead, TeacherStudentDINO
    model = TeacherStudentDINO(MiniViT(embed_dim=16, depth=1), DINOHead(16, out_dim=8, hidden_dim=16, bottleneck_dim=8))
    model.train()
    assert model.student_backbone.training and model.student_head.training
    assert not model.teacher_backbone.training and not model.teacher_head.training
    assert all(not parameter.requires_grad for parameter in model.teacher_parameters())

def test_practice_calls_all_lessons(monkeypatch):
    seen=set()
    def replace(self,p,m):seen.add(1);return p if m is None else torch.where(m.unsqueeze(-1),self.mask_token.expand_as(p),p)
    def positions(self,p,g):return torch.cat((self.cls_token.expand(p.shape[0],-1,-1),p),1)+interpolate_2d_pos_embed(self.pos_embed,g)
    def views(x):seen.add(2);return MultiCropAugmentation()(x)
    def masks(b,n,r,device):
        m=torch.zeros(b,n,dtype=torch.bool,device=device);m[:,:max(1,round(n*r))]=True;return m
    def dino(s,t,c,st,tt):
        seen.add(3);return sum(-(F.softmax((x.detach()-c)/tt,-1)*F.log_softmax(y/st,-1)).sum(-1).mean() for i,x in enumerate(t) for j,y in enumerate(s) if i!=j)/(2*(len(s)-1))
    def center(c,t,m):return c*m+torch.cat(t).detach().mean(0,keepdim=True)*(1-m)
    def ibot(s,t,m,c,st,tt):seen.add(4);return -(F.softmax((t.detach()-c)/tt,-1)*F.log_softmax(s/st,-1)).sum(-1)[m].mean()
    def ema(t,s,m):
        with torch.no_grad():
            for a,b in zip(t,s):a.mul_(m).add_(b,alpha=1-m)
    monkeypatch.setattr(l1.ExerciseViT,'replace_masked_patches',replace);monkeypatch.setattr(l1.ExerciseViT,'add_cls_and_positions',positions);monkeypatch.setattr(l2,'make_multicrop_views',views);monkeypatch.setattr(l2,'random_patch_mask',masks);monkeypatch.setattr(l3,'dino_cross_view_loss',dino);monkeypatch.setattr(l3,'update_center',center);monkeypatch.setattr(l4,'ibot_masked_patch_loss',ibot);monkeypatch.setattr(l4,'ema_update',ema)
    import run_dinov2_demo
    run_dinov2_demo.main('practice')
    assert seen=={1,2,3,4}
