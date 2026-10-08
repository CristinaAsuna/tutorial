"""保护 full teacher 顺序与练习接线；学生答案始终未被写回。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import check_lessons
import practice_ijepa as practice
import lesson1_block_masks as l1
import lesson2_predictor_packing as l2
import lesson3_ema_teacher as l3
import lesson4_training_probe as l4
from edu_core.training import update_ema,freeze_and_keep_eval
from reference_ijepa import sample_block_masks

def test_full_teacher_before_gather():
    torch.manual_seed(7)
    check_lessons.check_full_teacher()

def test_practice_calls_all_lessons(monkeypatch):
    seen=set()
    def mask(grid,count,block):
        seen.add(1);m=sample_block_masks(grid,num_targets=count,block_size=block);return list(m.targets),m.context
    def pack(c,m,p):seen.add(2);return torch.cat((c,m.expand(c.shape[0],p.shape[1],-1)+p),1)
    def ema(t,s,m):seen.add(3);update_ema(t,s,m);freeze_and_keep_eval(t)
    def regression(pred,target):
        seen.add("loss");return torch.nn.functional.smooth_l1_loss(pred,target.detach())
    monkeypatch.setattr(l4,"latent_regression_loss",regression)
    def train(model,images,masks,opt):
        seen.add(4);opt.zero_grad();loss=model(images,masks)['loss'];loss.backward();opt.step();model.update_target_encoder(.9);return loss
    monkeypatch.setattr(l1,'sample_block_masks',mask);monkeypatch.setattr(l2,'pack_predictor_tokens',pack);monkeypatch.setattr(l3,'update_target',ema);monkeypatch.setattr(l4,'jepa_training_step',train)
    import run_ijepa_demo
    run_ijepa_demo.main('practice')
    assert seen=={1,2,3,4,"loss"}
