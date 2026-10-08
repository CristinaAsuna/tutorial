"""只有测试通过 monkeypatch 提供 oracle；产品 practice 不提供回退。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import lesson1_tubelet_embed as l1
import lesson2_spatiotemporal_masks as l2
import lesson3_latent_predictor as l3
import lesson4_ema_training as l4
from reference_vjepa import sample_spatiotemporal_masks
def patchify_oracle(x,t,p):
    b,c,f,h,w=x.shape
    return x.reshape(b,c,f//t,t,h//p,p,w//p,p).permute(0,2,4,6,3,5,7,1).reshape(b,-1,c*t*p*p)

def test_practice_calls_all_lessons(monkeypatch):
    seen=set()
    def patches(x,t,p):seen.add(1);return patchify_oracle(x,t,p)
    def mask(b,g,block,n):seen.add(2);return sample_spatiotemporal_masks(b,g,block,n)
    def pack(self,c,p):seen.add(3);return torch.cat((self.context_proj(c),self.mask_token.expand(c.shape[0],p.shape[1],-1)+self.target_pos_proj(p)),1)
    def select(self,x,n):return self.out(self.norm(x[:,-n:]))
    def regression(pred,target):
        seen.add("loss");return torch.nn.functional.smooth_l1_loss(pred,target.detach())
    monkeypatch.setattr(l4,"latent_regression_loss",regression)
    def train(model,x,t,c,opt,m):
        seen.add(4);opt.zero_grad();loss=model(x,t,c)['loss'];loss.backward();opt.step();model.update_target(m);return loss
    monkeypatch.setattr(l1,'tubelet_patchify',patches);monkeypatch.setattr(l2,'make_masks',mask);monkeypatch.setattr(l3.Predictor,'pack_tokens',pack);monkeypatch.setattr(l3.Predictor,'select_predictions',select);monkeypatch.setattr(l4,'vjepa_training_step',train)
    import run_vjepa_demo
    run_vjepa_demo.main('practice')
    assert seen=={1,2,3,4,"loss"}
