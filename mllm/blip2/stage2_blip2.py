"""
stage2_blip2.py
===============
BLIP-2 第二阶段：视觉-语言生成式对齐 (Stage 2: Vision-to-Language Generation)

【架构组成与职责分工】:
1. Frozen Visual Encoder (如 EVA-CLIP / ViT-G): 提取原始图像 patch 特征，全参数冻结。
2. Q-Former (Stage 1 已学好抽取能力的桥梁): 负责将 257 个图像 patch 特征提炼压缩为 32 个 Query Tokens。
3. Linear Projection Layer: 将 32 个 Query 的特征维度 (如 768) 投影到 LLM 词嵌入维度 (如 2048 / 4096)。
4. Frozen LLM (如 OPT, LLaMA, Flan-T5): 冻结所有权重，利用投影后的视觉 Query 作为软提示词 (Soft Prompt / Prefix)
   进行自回归文本生成。

【Tensor 维度接力全过程】:
  1. 图像输入:          images           (B, 3, 224, 224)
  2. ViT 编码:          image_embeds     (B, N_img=257, D_img=1408)  [冻结]
  3. Q-Former 压缩抽取: query_output     (B, M=32, D_q=768)          [可学习/微调]
  4. 线性对齐层:        projected_query  (B, M=32, D_llm=2048)       [可学习]
  5. 文本 Prompt 嵌入:  text_embeds      (B, L_prompt, D_llm)        [冻结 Embedding]
  6. 拼接送入 LLM:      llm_inputs       (B, M + L_prompt, D_llm)
  7. LLM 自回归预测/生成输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from edu_core.training import freeze_and_keep_eval
from typing import Optional, Dict
from qformer import QFormer


class MockVisionEncoder(nn.Module):
    """用于本地教学和快速验证的轻量级 ViT 视觉编码器模拟类"""
    def __init__(self, img_feat_dim: int = 1408):
        super().__init__()
        self.conv = nn.Conv2d(3, img_feat_dim, kernel_size=16, stride=16) # 将 224x224 图像切为 14x14=196 个 patch

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, 3, 224, 224)
        x = self.conv(pixel_values) # (B, D_img, 14, 14)
        x = x.flatten(2).transpose(1, 2) # (B, 196, D_img)
        return x


class MockLLM(nn.Module):
    """用于本地教学和快速验证的轻量级因果语言模型 (Decoder-Only LLM) 模拟类"""
    def __init__(self, vocab_size: int = 32000, llm_dim: int = 2048):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, llm_dim)
        # 使用 Transformer EncoderLayer + 因果下三角 mask 模拟真实 LLM 的 Cross-Token 依赖
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_dim,
            nhead=4,
            dim_feedforward=llm_dim * 2,
            batch_first=True
        )
        self.lm_head = nn.Linear(llm_dim, vocab_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run causal decoding; attention_mask uses 1 for a valid token."""
        seq_len = inputs_embeds.shape[1]
        # Bool masks share the same convention as src_key_padding_mask: True means blocked.
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=inputs_embeds.device), diagonal=1
        )
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != inputs_embeds.shape[:2]:
                raise ValueError("attention_mask must have shape (batch_size, sequence_length).")
            key_padding_mask = ~attention_mask.bool()
        h = self.decoder_layer(
            inputs_embeds,
            src_mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        logits = self.lm_head(h) # (B, Seq_Len, vocab_size)
        return logits


class Blip2ForConditionalGeneration(nn.Module):
    """
    BLIP-2 Stage 2 完整顶层模型
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        qformer: QFormer,
        llm: nn.Module,
        qformer_dim: int = 768,
        llm_dim: int = 2048
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.qformer = qformer
        self.llm = llm

        # 【核心投影层】: 连接 Q-Former 与 LLM 的可训练桥梁之一。
        self.llm_proj = nn.Linear(qformer_dim, llm_dim)

        # 按照论文设计，必须冻结 ViT 和 LLM
        self.freeze_modules()

    def freeze_modules(self):
        """冻结视觉编码器和 LLM，并固定其推理态；Q-Former 和投影层可训练。"""
        freeze_and_keep_eval(self.vision_encoder)
        freeze_and_keep_eval(self.llm)

    def train(self, mode: bool = True):
        """Keep frozen backbones in eval mode even when the wrapper is trained."""
        super().train(mode)
        self.vision_encoder.eval()
        self.llm.eval()
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,     # (B, 3, H, W)
        prompt_input_ids: torch.Tensor, # (B, L_prompt) 问题或前缀 token
        answer_input_ids: torch.Tensor, # (B, L_answer) 目标回答 token (用于算 Loss)
        prompt_attention_mask: Optional[torch.Tensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2 训练前向过程: 计算因果语言模型损失 (Cross Entropy)
        """
        B = pixel_values.shape[0]
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids)
        if answer_attention_mask is None:
            answer_attention_mask = torch.ones_like(answer_input_ids)
        if prompt_attention_mask.shape != prompt_input_ids.shape:
            raise ValueError("prompt_attention_mask must match prompt_input_ids.")
        if answer_attention_mask.shape != answer_input_ids.shape:
            raise ValueError("answer_attention_mask must match answer_input_ids.")

        # 1. 冻结提取视觉特征 (不需要计算图梯度)
        with torch.no_grad():
            image_embeds = self.vision_encoder(pixel_values) # (B, N_img, D_img)

        # 2. Q-Former 提取 32 个视觉 Query 向量
        # query_output: (B, M, D_q) 其中 M=32
        query_output = self.qformer.extract_visual_queries(image_embeds)

        # 3. 线性投影到 LLM 的特征空间
        # projected_query: (B, M, D_llm)
        projected_query = self.llm_proj(query_output)

        # 4. 每个样本先去除 prompt padding，再与答案拼接。不能直接保留 prompt
        # padding：否则“预测 answer 第一个 token”的上一个位置会是 pad。
        M = projected_query.shape[1]
        L_a = answer_input_ids.shape[1]
        sequence_embeds, sequence_labels, sequence_masks = [], [], []
        for batch_index in range(B):
            valid_prompt_ids = prompt_input_ids[batch_index][prompt_attention_mask[batch_index].bool()]
            valid_prompt_embeds = self.llm.embed_tokens(valid_prompt_ids)
            answer_embeds = self.llm.embed_tokens(answer_input_ids[batch_index])
            sequence_embeds.append(torch.cat([projected_query[batch_index], valid_prompt_embeds, answer_embeds], dim=0))

            prefix_labels = torch.full(
                (M + valid_prompt_ids.shape[0],), -100, dtype=torch.long, device=pixel_values.device
            )
            answer_labels = answer_input_ids[batch_index].masked_fill(
                ~answer_attention_mask[batch_index].bool(), -100
            )
            sequence_labels.append(torch.cat([prefix_labels, answer_labels], dim=0))
            sequence_masks.append(torch.ones(M + valid_prompt_ids.shape[0] + L_a, dtype=torch.long, device=pixel_values.device))

        # Batch-pad only at the very end, where it cannot become a causal predecessor.
        inputs_embeds = nn.utils.rnn.pad_sequence(sequence_embeds, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(sequence_labels, batch_first=True, padding_value=-100)
        llm_attention_mask = nn.utils.rnn.pad_sequence(sequence_masks, batch_first=True, padding_value=0)

        # 5. LLM 前向预测
        # logits: (B, Total_Len, vocab_size)
        logits = self.llm(inputs_embeds, attention_mask=llm_attention_mask)

        # 6. 计算自回归下一个 Token 预测的交叉熵损失
        # 预测位置 t 的 token 是 labels 的位置 t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return {"loss": loss, "logits": logits, "labels": labels}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stage 2 推理生成过程: 贪心生成 (Greedy Search) 文本回答
        """
        B = pixel_values.shape[0]
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids)
        if prompt_attention_mask.shape != prompt_input_ids.shape:
            raise ValueError("prompt_attention_mask must match prompt_input_ids.")
        if (prompt_attention_mask.sum(dim=1) == 0).any():
            raise ValueError("Each prompt must contain at least one valid token.")
        # The compact batched loop appends new tokens after the common padded length.
        # Therefore generation accepts standard right padding (1...1, 0...0), not holes/left padding.
        prompt_mask = prompt_attention_mask.bool()
        if prompt_mask.shape[1] > 1 and (prompt_mask[:, 1:] > prompt_mask[:, :-1]).any():
            raise ValueError("generate only supports right-padded prompt_attention_mask values.")

        # 1. 抽取视觉特征与投影
        image_embeds = self.vision_encoder(pixel_values)
        query_output = self.qformer.extract_visual_queries(image_embeds)
        projected_query = self.llm_proj(query_output) # (B, M, D_llm)

        # 2. 文本 Prompt 嵌入
        prompt_embeds = self.llm.embed_tokens(prompt_input_ids) # (B, L_prompt, D_llm)

        # 3. 初始输入: 视觉 Query + Prompt
        curr_embeds = torch.cat([projected_query, prompt_embeds], dim=1)
        visual_attention_mask = torch.ones(
            (B, projected_query.shape[1]), dtype=torch.long, device=pixel_values.device
        )
        curr_attention_mask = torch.cat([visual_attention_mask, prompt_attention_mask], dim=1)
        generated_ids = []

        # 4. 贪心逐 Token 循环自回归生成
        for _ in range(max_new_tokens):
            logits = self.llm(curr_embeds, attention_mask=curr_attention_mask)
            if generated_ids:
                next_logits = logits[:, -1, :]
            else:
                # The first prediction must come from each sample's final valid prompt token.
                final_prompt_positions = (
                    projected_query.shape[1] + prompt_attention_mask.long().sum(dim=1) - 1
                )
                next_logits = logits[torch.arange(B, device=logits.device), final_prompt_positions]
            next_token_id = next_logits.argmax(dim=-1, keepdim=True) # (B, 1)
            generated_ids.append(next_token_id)

            # 将新生成的 token 转为 embedding，追加到当前输入
            next_embed = self.llm.embed_tokens(next_token_id)
            curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)
            curr_attention_mask = torch.cat(
                [curr_attention_mask, torch.ones((B, 1), dtype=torch.long, device=pixel_values.device)], dim=1
            )

        return torch.cat(generated_ids, dim=1) # (B, max_new_tokens)
