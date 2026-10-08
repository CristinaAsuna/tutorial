"""关卡 1：固定长度视觉瓶颈。
目标：每张图独立变成 R 个 trainable latents，文本长度不随图像 patch 数增长。
前置：通用 attention 已由 edu_core 提供；只练论文数据流。
形状：features(B,M,N,Dv) -> memory(B*M,N,D) -> latents(B*M,R,D) -> (B,M,R,D)。
手算：B=2,M=2,N=4,R=3 -> 展平 batch=4，每幅图4个patch压成3个latents。
常见错：把 M 并入 patch 轴造成图片相互混合；用 detach 切断 resampler 梯度。
接入：practice 模型的 resampler 就是本类；初始化可复用参考，forward 必须自己写。
"""
from reference_flamingo import PerceiverResampler as ResamplerParameters

def resample(visual_features, latents, vision_proj, cross_norm, cross_attn, blocks):
    # TODO 1.1：检查(B,M,N,Dv)，投影后 reshape(B*M,N,D)。
    # TODO 1.2：latents(1,R,D) expand 到 B*M，cross_norm 后作 Q，memory 作 KV，加 residual。
    # TODO 1.3：逐个通用 TransformerBlock refine latents，恢复(B,M,R,D)。
    raise NotImplementedError("Flamingo TODO 1.1-1.3: resample")

class PerceiverResampler(ResamplerParameters):
    # 父类只初始化 vision_proj/latents/cross_norm/cross_attn/blocks；不调用父类 forward。
    def forward(self, visual_features):
        return resample(visual_features,self.latents,self.vision_proj,self.cross_norm,self.cross_attn,self.blocks)
