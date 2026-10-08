"""关卡 2：仅 query 直接读取视觉 memory。
目标：保留 query 信息瓶颈；前置：基础 attention、残差和 LayerNorm。
输入 hidden (B,M+L,D)、image (B,N,Dv)，输出同 hidden。
手算：hidden=[q0,q1,t0]，视觉更新 delta=[1,2]，输出 [q0+1,q1+2,t0]。
常见错：让 text cross-attend；遗漏 residual；复制 text 后丢掉梯度。
接入：PracticeLayer 在联合 self-attention 后调用，再执行共享 FFN。
"""
import torch

def query_only_cross_attention(hidden_states, num_queries, image_features, cross_attn, norm):
    # TODO 2.1：检查 0 < M <= 序列长度；取前 M 个 query。
    # TODO 2.2：query + cross_attn(norm(query),image_features)。
    # TODO 2.3：cat 更新 query 与未改动的 hidden[:,M:]，返回完整序列。
    raise NotImplementedError("InstructBLIP lesson 2: query_only_cross_attention 尚未完成；见 HINTS.md")
