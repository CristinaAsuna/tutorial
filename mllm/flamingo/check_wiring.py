"""维护者接线测试：oracle只存在于测试替身，不补答案或改变practice代码。"""
from contextlib import ExitStack
from unittest.mock import patch
import torch
import reference_flamingo as ref
from edu_core.attention import GatedCrossAttentionBlock
import lesson1_perceiver_resampler as l1
import lesson2_media_causal_mask as l2
import lesson3_gated_cross_attention as l3
import lesson4_training_and_generation as l4

def resample(features,latents,proj,norm,attn,blocks):
    b,m,n,_=features.shape; memory=proj(features).reshape(b*m,n,-1)
    x=latents.expand(b*m,-1,-1); x=x+attn(norm(x),memory)
    for block in blocks: x=block(x)
    return x.reshape(b,m,x.shape[1],x.shape[2])

def media(ids,sentinel,num_images,r,valid=None):
    model=ref.build_toy_flamingo()
    # Oracle wrapper uses the explicit R supplied to the student's pure function.
    model.resampler.latents=torch.nn.Parameter(torch.zeros(1,r,32))
    return model.build_media_attention_mask(ids,num_images,valid)

def train(model,opt,ids,images,valid,labels):
    opt.zero_grad(); out=model(ids,images,attention_mask=valid,labels=labels)
    out['loss'].backward(); opt.step(); return out

def main():
    replacements=[(l1,'resample',resample),(l2,'build_media_mask',media),
        (l3.GatedCrossAttention,'forward',GatedCrossAttentionBlock.forward),
        (l4,'decoder_forward',ref.FlamingoDecoder.forward),(l4,'connector_training_step',train)]
    with ExitStack() as stack:
        mocks=[stack.enter_context(patch.object(obj,name,side_effect=fn,autospec=True)) for obj,name,fn in replacements]
        from run_flamingo_demo import main as demo
        demo('practice')
        for (_,name,_),mock in zip(replacements,mocks): assert mock.call_count>0,f"practice bypassed {name}"
    print('Flamingo wiring: all student mechanisms and training step reached')

if __name__=='__main__': main()
