"""第3关：从逐样本 softmax 走向 batch 内 prototype 平衡分配。
前置：logsumexp；先理解 Q 为 K,N，再看归一化。
形状：logits N,K -> logQ K,N -> targets N,K。
手算：2x2 全零 logits -> 每样本 [0.5,0.5]，每个prototype总量为1。
接入：model.targets，同样用于 CLS 与被 mask 选中的 teacher patches。
常见错误：归一化错轴；省略最终乘N；对 teacher 反传；exp 溢出。
检查：python3 -m mllm.dinov2.check_lessons --lesson 3
"""

def sinkhorn(logits, temperature=.07, iterations=3):
    # TODO 3.1: no_grad 下将 detached logits 转 float64、除温度、转置。
    # TODO 3.2: 用 logsumexp 减去全局 log 总质量。
    # TODO 3.3: 重复 iterations 轮：每行减 logsumexp(row)+log(K)，
    #          每列减 logsumexp(col)+log(N)。三轮后列边际精确，行边际近似。
    # TODO 3.4: exp(logQ+log(N)).T 转回输入 dtype。
    raise NotImplementedError('lesson3.sinkhorn TODO 3.1-3.4')
