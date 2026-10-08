"""维护者接线测试：仅测试时注入 oracle；不修改 lesson，不完成练习。"""
from contextlib import ExitStack
from unittest.mock import patch
import torch
import reference_act as ref
import lesson1_action_chunks as l1
import lesson2_cvae_style as l2
import lesson3_transformer_policy as l3
import lesson4_temporal_ensemble as l4
import lesson5_training_and_inference as l5

def rec_loss(p,t,v,m,l,b):
    rec=ref.masked_l1(p,t,v); kl=ref.kl_to_standard_normal(m,l)
    return rec+b*kl,rec,kl

def aggregate(history,decay):
    en=ref.TemporalEnsembler(history[0].shape[0],history[0].shape[1],decay)
    for chunk in history: result=en.add(chunk)
    return result

def train(model,opt,images,qpos,chunks,valid,beta=.01):
    opt.zero_grad(); out=model(images,qpos,chunks,valid,beta=beta)
    out['loss'].backward(); opt.step(); return out

def forward(model,images,qpos,chunks=None,valid=None,*,beta=10.):
    if chunks is None:
        z=torch.zeros(qpos.shape[0],model.z_proj.in_features,device=qpos.device,dtype=qpos.dtype)
        return dict(actions=model._decode(images,qpos,z),loss=None,reconstruction_loss=None,kl_loss=None)
    mean,logvar=model.style_encoder(qpos,chunks,valid)
    z=l5.reparameterize(mean,logvar)
    result=model._decode(images,qpos,z)
    loss,rec,kl=l5.cvae_loss(result,chunks,valid,mean,logvar,beta)
    return dict(actions=result,loss=loss,reconstruction_loss=rec,kl_loss=kl,mean=mean,logvar=logvar)

def main():
    # Patch source before importing adapter because adapter intentionally binds lesson functions.
    replacements=[(l1,'sample_future_chunks',ref.sample_action_chunks),
        (l2.StyleEncoder,'forward',ref.StyleEncoder.forward),
        (l5,'reparameterize',lambda m,l,noise=None:m+torch.exp(.5*l)*(torch.randn_like(m) if noise is None else noise)),
        (l5,'cvae_loss',rec_loss),
        (l3,'encode_observation',ref.ACTPolicy._encode_observation),
        (l3,'decode_action_chunk',lambda mem,q,dec,head:head(dec(q.expand(mem.shape[0],-1,-1),mem))),
        (l4,'ensemble_current_action',aggregate),(l5,'policy_forward',forward),(l5,'training_step',train)]
    with ExitStack() as stack:
        mocks=[stack.enter_context(patch.object(obj,name,side_effect=fn,autospec=True)) for obj,name,fn in replacements]
        from run_act_demo import main as demo
        demo('practice')
        for (_,name,_),mock in zip(replacements,mocks):
            assert mock.call_count>0, f"practice bypassed {name}"
    print('ACT wiring: all student mechanisms and training step reached')

if __name__=='__main__': main()
