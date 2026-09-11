"""
lesson3_qformer.py
==================
【关卡 3】：Q-Former 骨干网络 (Querying Transformer Backbone)

在前两关中，你已经完成了基础 Layer 和 Mask。
现在，我们要将它们拼装成完整的 Q-Former 骨干模型！

本文件需要你实现 2 个核心前向接口:
1. extract_visual_queries: 纯视觉特征抽取 (用于 Stage 2 喂给 LLM)
2. forward: 多模态联合前向传播 (支持 'itc' / 'itm' / 'itg' 模式)

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson3_qformer.py
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from qformer_layer import QFormerLayer
from attention_masks import create_itc_mask, create_itm_mask, create_itg_mask


class QFormer(nn.Module):
    """
    【__init__ 参数】:
      - num_query_tokens: 可学习查询向量数 M (典型值 32)
      - hidden_size: 特征维度 D_q (典型值 768)
      - num_layers: 堆叠层数 (典型值 12)
      - img_feat_dim: 图像特征维度 (如 EVA-CLIP 1408)
      - vocab_size: 词表大小 (如 30522)
      - embed_dim: 对比学习投影维度 (如 256)
    """
    def __init__(
        self,
        num_query_tokens: int = 32,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        img_feat_dim: int = 1408,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        cross_attention_freq: int = 2,
        embed_dim: int = 256
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size

        # 1. 核心可学习参数: (1, M, D_q)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size) * 0.02)

        # 2. 文本 Embedding 体系
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.pos_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.embed_layer_norm = nn.LayerNorm(hidden_size)

        # 3. 堆叠多层 QFormerLayer (隔层配置 Cross-Attention)
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                img_feat_dim=img_feat_dim,
                has_cross_attention=(i % cross_attention_freq == 0)
            )
            for i in range(num_layers)
        ])

        # 4. 任务投影头
        self.vision_proj = nn.Linear(hidden_size, embed_dim)
        self.text_proj = nn.Linear(hidden_size, embed_dim)
        self.itm_head = nn.Linear(hidden_size, 2)

    def extract_visual_queries(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        【接口 1：纯视觉抽取模式 (用于 Stage 2 喂给 LLM)】:

        【输入】: image_embeds: (B, N_img, D_img)
        【输出】: query_output: (B, M, D_q)
        """
        # =========================================================================
        # TODO 3.1: 请实现纯视觉提取逻辑
        # 步骤提示:
        # 1. 获取 B = image_embeds.shape[0], M = self.num_query_tokens
        # 2. 将 self.query_tokens 从 (1, M, D_q) 扩展广播为 (B, M, D_q):
        #    query_states = self.query_tokens.expand(B, -1, -1)
        # 3. 构造全 1 的掩码: attn_mask = torch.ones((B, 1, M, M), device=image_embeds.device)
        # 4. 依次遍历 self.layers 进行前向计算:
        #    hidden_states = layer(hidden_states, num_query_tokens=M, attention_mask=attn_mask, image_embeds=image_embeds)
        # 5. 返回更新后的 hidden_states (B, M, D_q)
        # =========================================================================
        raise NotImplementedError("TODO 3.1 尚未实现！请实现 extract_visual_queries")

    def forward(
        self,
        image_embeds: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "itc"
    ) -> Dict[str, torch.Tensor]:
        """
        【接口 2：多模态联合模式 (用于 Stage 1 预训练)】:

        【输入】:
          - image_embeds: (B, N_img, D_img)
          - input_ids: (B, L)
          - attention_mask: (B, L)
          - mode: 'itc' | 'itm' | 'itg'

        【输出】: 字典包含:
          - "query_output": (B, M, D_q)
          - "text_output": (B, L, D_q)
        """
        # 如果没有文本，自动走纯视觉抽取
        if input_ids is None:
            return {"query_output": self.extract_visual_queries(image_embeds)}

        # =========================================================================
        # TODO 3.2: 请实现联合前向逻辑
        # 步骤提示:
        # 1. 获取 B, L = input_ids.shape, M = self.num_query_tokens
        # 2. 准备 query 嵌入: query_embeds = self.query_tokens.expand(B, -1, -1)
        # 3. 准备 text 嵌入 (Token Embed + Pos Embed + LayerNorm):
        #    positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        #    text_embeds = self.embed_layer_norm(self.word_embeddings(input_ids) + self.pos_embeddings(positions))
        # 4. 拼接 Query 和 Text: hidden_states = torch.cat([query_embeds, text_embeds], dim=1) -> (B, M+L, D_q)
        # 5. 根据 mode 选择掩码:
        #    if mode == 'itc': mask = create_itc_mask(B, M, attention_mask)
        #    elif mode == 'itm': mask = create_itm_mask(B, M, attention_mask)
        #    elif mode == 'itg': mask = create_itg_mask(B, M, attention_mask)
        # 6. 循环遍历 self.layers 计算
        # 7. 拆分前 M 个为 query_output，后 L 个为 text_output，返回字典
        # =========================================================================
        raise NotImplementedError("TODO 3.2 尚未实现！请实现 QFormer.forward")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试关卡 3 ==========")
    B, M, L, D_q = 2, 4, 5, 256
    N_img, D_img = 8, 512
    vocab_size = 1000

    qformer = QFormer(
        num_query_tokens=M,
        hidden_size=D_q,
        num_layers=2,
        num_heads=4,
        img_feat_dim=D_img,
        vocab_size=vocab_size
    )

    image_embeds = torch.randn(B, N_img, D_img)
    input_ids = torch.randint(0, vocab_size, (B, L))
    attn_mask = torch.ones((B, L), dtype=torch.long)

    # 1. 测试纯视觉提取
    v_queries = qformer.extract_visual_queries(image_embeds)
    assert v_queries.shape == (B, M, D_q), f"纯视觉抽取形状不符: {v_queries.shape}"
    print("✅ TODO 3.1 (extract_visual_queries) 通过测试！形状:", v_queries.shape)

    # 2. 测试联合模式
    out_itc = qformer(image_embeds, input_ids, attn_mask, mode="itc")
    assert "query_output" in out_itc and "text_output" in out_itc
    assert out_itc["query_output"].shape == (B, M, D_q)
    assert out_itc["text_output"].shape == (B, L, D_q)
    print("✅ TODO 3.2 (QFormer.forward) 通过测试！")
    print("🎉 恭喜！关卡 3 全部挑战成功！\n")


if __name__ == "__main__":
    run_test()
