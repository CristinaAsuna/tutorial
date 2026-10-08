"""把所有学生机制接入 reference 外围的 forward/generate；不回退参考机制。"""
from torch import nn
from reference_flamingo import MockVisionEncoder, FlamingoDecoder, FlamingoForConditionalGeneration, IMAGE_TOKEN_INDEX
from lesson1_perceiver_resampler import PerceiverResampler
from lesson2_media_causal_mask import build_media_mask
from lesson3_gated_cross_attention import GatedCrossAttention
from lesson4_training_and_generation import decoder_forward, connector_training_step

class PracticeDecoder(FlamingoDecoder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        heads=kwargs.get('num_heads',4)
        self.gated_cross_attn=nn.ModuleDict({key:GatedCrossAttention(self.dim,heads) for key in self.gated_cross_attn})
    def forward(self,input_ids,visual_memory,media_attention_mask,attention_mask):
        return decoder_forward(self,input_ids,visual_memory,media_attention_mask,attention_mask)

class PracticeFlamingo(FlamingoForConditionalGeneration):
    def build_media_attention_mask(self,input_ids,num_images,attention_mask):
        return build_media_mask(input_ids,IMAGE_TOKEN_INDEX,num_images,self.resampler.latents.shape[1],attention_mask)

def build_toy_flamingo(vocab_size=64):
    return PracticeFlamingo(MockVisionEncoder(),PerceiverResampler(24,32,num_latents=3,num_heads=4,depth=1),
        PracticeDecoder(vocab_size,32,max_positions=64,num_heads=4,depth=4,cross_attention_every=2))
