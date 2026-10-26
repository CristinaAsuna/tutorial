"""第5关：KoLeo 用最近邻距离鼓励 batch 特征分散。
前置：L2 normalize、最近邻。输入 CLS B,D，B>=2；输出 scalar。
手算：[1,0] 与 [0,1] 的距离 sqrt(2)，loss=-log(sqrt(2))。
接入：每个 global view 单独计算，避免同图的另一 view 成为近邻捷径。
常见错误：未排除自己，距离恒0；用 prototype logits 代替 backbone CLS。
检查：python3 -m mllm.dinov2.check_lessons --lesson 5
"""

def koleo(features, eps=1e-8):
    # TODO 5.1: 验证B>=2，float32 L2归一化；构造 BxB cosine similarity。
    # TODO 5.2: 无梯度选择：对角线=-inf，再 argmax 得最近的其他样本。
    # TODO 5.3: 可反传计算归一化特征与近邻的L2距离，clamp_min(eps)，-log后mean。
    raise NotImplementedError('lesson5.koleo TODO 5.1-5.3')
