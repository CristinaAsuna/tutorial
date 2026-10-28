"""第3关：条件SiT与早期hidden，不重写通用attention。
输入tokens=(B,16,32)、condition=(B,32)，block输出同shape。
modulation输出(B,192)，chunk为6个(B,32)：shift/scale/gate各两组。
手算norm(x)=2,scale=.5,shift=1 -> 4；residual=3,branch=4,gate=.25 -> 4。
完整latent(B,4,8,8)经patch2 -> (B,16,16)再Linear -> (B,16,32)。
第1个block之后project为(B,16,24)，第2个继续预测(B,4,8,8)速度。
常见错：沿batch切hidden；用最终速度对齐；project=False仍调用projector。
检查：python3 -m mllm.repa.check_lessons --lesson 3。
"""
from edu_core.vision import patchify, unpatchify


def conditioned_block(block, tokens, condition):
    # TODO 3.1：block.modulation(condition).chunk(6,-1)，依次shift1/scale1/gate1/shift2/scale2/gate2。
    # TODO 3.2：norm1后按x*(1+scale)+shift调制，attention后乘gate加原残差。
    # TODO 3.3：norm2/MLP同理，所有条件加[:,None]广播。
    raise NotImplementedError('REPA lesson3.conditioned_block TODO 3.1-3.3')


def denoise(model, latent, time, labels, *, project=True):
    # TODO 3.4：patchify(latent,2)，model.patch投影并加model.positions。
    # TODO 3.5：model.time_embed(time[:,None])+model.label_embed(labels)。
    # TODO 3.6：enumerate(model.blocks,1)，每层调用model.ops.conditioned_block。
    #           仅project且index==alignment_depth时计算model.projector(tokens)。
    # TODO 3.7：final_modulation拆shift/scale，调制final_norm(tokens)，model.output。
    # TODO 3.8：unpatchify(...,2,4,grid=(4,4))，返回velocity和projected（关闭时None）。
    raise NotImplementedError('REPA lesson3.denoise TODO 3.4-3.8')
