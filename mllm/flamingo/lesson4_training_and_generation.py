"""关卡 4：冻结 LM 的插层执行与训练。
目标：冻结参数仍传递 connector 梯度；按 next-token shift 监督答案。
前置：前三关；因果mask和mock骨干复用，但插层遍历需要自己组装。
形状：visual(B,M,R,D)展平(B,M*R,D)；ids(B,L) -> logits(B,L,V)。
手算：depth=4,every=2 -> block0,block1,gate1,block2,block3,gate3；gate0时仍是纯LM。
常见错：把整个冻结LM放no_grad，切断中间connector；首步要求resampler非零梯度。
接入：practice decoder.forward 和 demo training step 调用本文件；generate通过模型forward反复调用。
"""
import torch

def decoder_forward(decoder,input_ids,visual_memory,media_attention_mask,attention_mask):
    # TODO 4.1：验证 batch/长度/mask；token+position embedding；视觉 flatten(1,2)。
    # TODO 4.2：遍历 blocks，传key_padding_mask=attention_mask,causal=True；若str(index)在gated_cross_attn则调用。
    # TODO 4.3：norm -> lm_head，输出logits；不要包no_grad。
    raise NotImplementedError("Flamingo TODO 4.1-4.3: decoder_forward")

def connector_training_step(model,optimizer,input_ids,pixel_values,attention_mask,labels):
    # TODO 4.4：zero_grad，model(...labels=labels)，loss.backward，optimizer.step，返回输出。
    # 模型外围已负责answer-only labels和next-token shift；首步gate更新后才有resampler有效梯度。
    raise NotImplementedError("Flamingo TODO 4.4: connector_training_step")
