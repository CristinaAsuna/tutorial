"""
qformer.py
==========
完整的 Q-Former 骨干网络 (Querying Transformer)

【模块职责】:
1. 维护一组可学习的查询向量: self.query_tokens (1, num_query_tokens, hidden_size)
2. 维护文本 Token Embedding 和 Position Embedding。
3. 堆叠 N 个 QFormerLayer 构成深层网络（在 BLIP-2 中，通常每隔一层插入 Cross-Attention）。
4. 提供两种典型调用模式:
   - 纯视觉查询模式 (用于 Stage 2 喂给 LLM): 仅传入 image_embeds，输出压缩后的视觉 Query 表征。
   - 多模态联合训练模式 (用于 Stage 1 预训练): 传入 image_embeds, input_ids 以及指定模式 ('itc', 'itm', 'itg')。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from attention_masks import create_itc_mask, create_itm_mask, create_itg_mask
from qformer_layer import QFormerLayer


class QFormer(nn.Module):
    def __init__(
        self,
        num_query_tokens: int = 32,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        img_feat_dim: int = 1408,
        vocab_size: int = 30522,       # 对应标准 BERT-base 词表大小
        max_position_embeddings: int = 512,
        cross_attention_freq: int = 2, # 每隔 2 层插入一次 Cross-Attention
        embed_dim: int = 256           # ITC 对比学习低维投影空间
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size

        # 1. 可学习的 Query Token (核心参数): 形状为 (1, M, D_q)
        # 初始化为微小的正态分布或零均值正态分布
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size) * 0.02)

        # 2. 文本 Embedding 体系 (基于 BERT 风格)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.pos_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.embed_layer_norm = nn.LayerNorm(hidden_size)

        # 3. 堆叠多层 QFormerLayer
        self.layers = nn.ModuleList([
            QFormerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                img_feat_dim=img_feat_dim,
                has_cross_attention=(i % cross_attention_freq == 0) # 隔层配置 Cross-Attention
            )
            for i in range(num_layers)
        ])

        # 4. Stage 1 专用多任务输出头 (Heads)
        # 4.1 ITC 投影头 (将 Query 和 Text [CLS] 映射到 256 维单位超球面上做余弦相似度)
        self.vision_proj = nn.Linear(hidden_size, embed_dim)
        self.text_proj = nn.Linear(hidden_size, embed_dim)

        # 4.2 ITM 图文匹配二分类头 (输出匹配与否 logits)
        self.itm_head = nn.Linear(hidden_size, 2)

    def extract_visual_queries(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        【Stage 2 核心接口】: 纯视觉特征提取

        【数据流】:
          输入 image_embeds: (B, N_img, D_img)
          展开 Query Tokens: (1, M, D_q) -> 广播扩展为 (B, M, D_q)
          经过 N 层 QFormerLayer，通过 Cross-Attention 不断抽取图像语义
          输出 query_output: (B, M, D_q)
        """
        B = image_embeds.shape[0]
        # 扩展 Query Tokens 匹配 batch 大小: (B, M, D_q)
        query_states = self.query_tokens.expand(B, -1, -1)

        # 由于没有文本输入，Query 内部全互见，Mask 为全 1: (B, 1, M, M)
        M = self.num_query_tokens
        attn_mask = torch.ones((B, 1, M, M), device=image_embeds.device)

        hidden_states = query_states
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                num_query_tokens=M,
                attention_mask=attn_mask,
                image_embeds=image_embeds
            )

        return hidden_states  # (B, M, D_q)

    def forward(
        self,
        image_embeds: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "itc"
    ) -> Dict[str, torch.Tensor]:
        """
        【Stage 1 训练核心前向传播】

        支持三种模式 (mode):
          - 'itc': 图文对比 (Query 与 Text 隔离)
          - 'itm': 图文匹配 (Query 与 Text 深度互联)
          - 'itg': 因果文本生成 (Text 单向看 Query, Text 内部因果掩码)
        """
        # 如果没有文本输入，退化为纯视觉查询模式
        if input_ids is None:
            query_output = self.extract_visual_queries(image_embeds)
            return {"query_output": query_output}

        B, L = input_ids.shape
        M = self.num_query_tokens

        # 1. 构造 Query 嵌入向量: (B, M, D_q)
        query_embeds = self.query_tokens.expand(B, -1, -1)

        # 2. 构造 Text 嵌入向量: (B, L, D_q)
        # Token Embedding + Position Embedding
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        text_embeds = self.word_embeddings(input_ids) + self.pos_embeddings(positions)
        text_embeds = self.embed_layer_norm(text_embeds)

        # 3. 沿序列长度维度拼合序列:
        # [Query (M 个), Text (L 个)] -> (B, M + L, D_q)
        hidden_states = torch.cat([query_embeds, text_embeds], dim=1)

        # 4. 根据 mode 选择对应的 Attention Mask: (B, 1, M + L, M + L)
        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=input_ids.device, dtype=torch.long)

        if mode == "itc":
            attn_mask = create_itc_mask(B, M, attention_mask)
        elif mode == "itm":
            attn_mask = create_itm_mask(B, M, attention_mask)
        elif mode == "itg":
            attn_mask = create_itg_mask(B, M, attention_mask)
        else:
            raise ValueError(f"未知模式: {mode}")

        # 5. 经过所有 Transformer 层
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                num_query_tokens=M,
                attention_mask=attn_mask,
                image_embeds=image_embeds
            )

        # 6. 切分输出
        query_output = hidden_states[:, :M, :]  # (B, M, D_q)
        text_output = hidden_states[:, M:, :]   # (B, L, D_q)

        return {
            "query_output": query_output,
            "text_output": text_output,
            "hidden_states": hidden_states
        }
