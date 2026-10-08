"""关卡 2：交错序列中的图像可见性。
目标：实现本 toy 的 all-seen policy；论文主模型的 immediate previous-image policy 见 README。
前置：cumsum、广播、repeat_interleave；True 表示允许。
形状：ids/mask(B,L)，M幅图每图R个latent -> allow(B,L,M*R)。
手算：[BOS,<image>,a,<image>,b]，R=1 -> [00,10,10,11,11]；padding 行全False。
常见错：sentinel padding 也计数；图像编号从1开始却与0-based arange比较；暴露未来图像。
接入：practice 模型覆盖 build_media_attention_mask，每次 forward/generate 都使用本关。
"""
import torch

def build_media_mask(input_ids, image_token_index, num_images, latents_per_image, attention_mask=None):
    # TODO 2.1：mask 默认全有效，校验 shape、M/R>0；每行有效 sentinel 数必须=M。
    # TODO 2.2：seen=((ids==sentinel)&valid).long().cumsum(1)。
    # TODO 2.3：arange(M)<seen[...,None]，repeat_interleave(R,-1)，与valid[...,None]求交。
    # TODO 2.4：其他有效负token抛ValueError；返回bool。
    raise NotImplementedError("Flamingo TODO 2.1-2.4: build_media_mask")
