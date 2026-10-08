"""关卡 4：执行一个真实优化步骤。
目标：更新目标参数；前置：zero_grad/backward/step，以及两阶段冻结。
输入：batch 字典含 input_ids,pixel_values,attention_mask,labels；输出：detach 后标量 loss。
手算：梯度 2、lr=0.1、参数 1，SGD 后参数应为 0.8。
常见错：漏 zero_grad 累加梯度；loss.item() 后 backward；冻结 LLM 时用 no_grad 包整个模型。
接入：demo 两个训练阶段都调用本函数；生成复用外围贪心循环，仍经过学生 encode/packing。
"""

def train_one_step(model, optimizer, batch):
    # TODO 4.1：model.train()，optimizer.zero_grad()。
    # TODO 4.2：model(**batch)，取得 loss，backward，再 optimizer.step()。
    # TODO 4.3：返回 loss.detach()，供验收报告。
    raise NotImplementedError("LLaVA lesson 4: train_one_step 尚未完成；见 HINTS.md")
