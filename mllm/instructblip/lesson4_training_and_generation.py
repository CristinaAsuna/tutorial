"""关卡 4：训练 Q-Former/投影，保持视觉塔和 LLM 冻结。
前置：梯度传播与 requires_grad；输入 batch 是 model.forward 完整关键字参数字典。
输出 detach 标量 loss；手算 SGD：参数1,梯度2,lr0.1 -> 0.8。
常见错：冻结 LLM 后在 no_grad 中运行它，使视觉 prefix 梯度消失；忘记清梯度。
接入：practice demo 的 optimizer 步骤；生成复用贪心外围但 query 调用学生关卡 1/2。
"""

def training_step(model, optimizer, batch):
    # TODO 4.1：model.train()，zero_grad；视觉塔/LLM 的 eval 由外围 train 覆盖保证。
    # TODO 4.2：forward 获取 loss，backward，step。
    # TODO 4.3：返回 loss.detach()，检查冻结塔无梯度、Q-Former/投影有梯度。
    raise NotImplementedError("InstructBLIP lesson 4: training_step 尚未完成；见 HINTS.md")
