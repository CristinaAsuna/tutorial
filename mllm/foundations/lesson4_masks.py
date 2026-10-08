"""基础 4：可见性 mask 和 token 选择。

前置：布尔比较、gather；attention allow-mask 与 target mask 名称要区分。
形状：causal=(Q,K)；tokens=(B,N,D)，indices=(B,M)，输出=(B,M,D)。
例子：三位置 causal 每行分别允许 [0]、[0,1]、[0,1,2]；indices=[2,0] 保持该顺序。
常见错：只筛第一个样本，或排序 indices 后破坏 target 对应关系。
接入：JEPA 的 target/context 选择；语言模型的因果可见性。
检查：python3 check_lessons.py --lesson 4；提示：HINTS.md 的基础 4。
"""
import torch


def causal_allow_mask(length: int):
    # TODO 4.1：产生 query 行号与 key 列号，用 key<=query 比较。
    raise NotImplementedError("基础 4.1：causal_allow_mask")


def select_tokens(tokens, indices):
    # TODO 4.2：indices 在最后补轴并 expand 成 (B,M,D)。
    # TODO 4.3：沿 token 轴 gather，不改变给定索引顺序。
    raise NotImplementedError("基础 4.2–4.3：select_tokens")
