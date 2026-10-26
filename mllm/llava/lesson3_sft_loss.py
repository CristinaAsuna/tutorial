"""关卡 3：只监督 assistant 回答（含 EOS）。
目标：监督位置由显式角色 mask 决定；前置：布尔运算、torch.where。
形状：ids/assistant_mask/attention_mask 均 (B,L)，输出 labels (B,L)。
手算：ids=[1,-200,7,2,0], assistant_mask=[0,0,1,1,0] -> [-100,-100,7,2,-100]。
常见错：用固定位置代替角色 mask；监督 image sentinel；shift 两次。
接入：practice_llava.build_sft_example；forward 中 next-token shift 已由外围完成。
"""
import torch
from reference_llava import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def assistant_only_labels(input_ids, *, assistant_mask=None, attention_mask=None):
    # TODO 3.1：以显式 assistant_mask 作为监督位置；未提供时所有输入位置默认有效。
    # TODO 3.2：与有效 attention_mask 相交，再排除 -200。
    # TODO 3.3：torch.where(监督位置,input_ids,IGNORE_INDEX)，不要改变输入。
    raise NotImplementedError("LLaVA lesson 3: assistant_only_labels 尚未完成；见 HINTS.md")
