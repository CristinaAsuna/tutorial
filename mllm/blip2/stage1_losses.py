"""
stage1_losses.py
================
Stage 1 三大损失函数与难例挖掘 (Hard Negative Mining)

1. ITC (Image-Text Contrastive Loss):
   - 基于余弦相似度与 InfoNCE 损失。
   - 核心问题: 图像有 M 个 Query 向量，而文本只有一个 [CLS] 向量，如何算相似度？
   - BLIP-2 方案: 计算所有 M 个 Query 与文本 [CLS] 的点积，取【最大值 (Max Pooling)】作为图文对的相似度。

2. ITM (Image-Text Matching Loss) & 难例挖掘:
   - 二分类任务：判定图像和文本是否真正匹配 (Label 1 vs 0)。
   - 难例挖掘 (Hard Negative Mining): 利用 ITC 刚算出来的相似度矩阵，在同一个 Batch 内部找到
     非配对却“长得最像”的文本/图像作为负样本，逼迫模型学习细粒度区分度。

3. ITG (Image-Grounded Text Generation Loss):
   - 自回归下一个 Token 预测损失 (Causal LM Loss)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_itc_loss(
    query_output: torch.Tensor,      # (B, M, D_q) 来自 ITC mode
    text_output: torch.Tensor,       # (B, L, D_q) 来自 ITC mode
    vision_proj: nn.Linear,          # (D_q -> embed_dim)
    text_proj: nn.Linear,            # (D_q -> embed_dim)
    temperature: float = 0.07
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算图文对比损失 (ITC)

    Tensor 演变推导:
      1. query_feat: (B, M, D_q) -> (B, M, D_embed) -> L2 归一化
      2. text_feat: 提取 [CLS] 位置 (B, 0, D_q) -> (B, D_embed) -> L2 归一化
      3. 相似度矩阵:
         sim_matrix[i, j, m] = query_feat[i, m] · text_feat[j]
         形状: (B, B, M)
      4. 最大相似度池化:
         sim_i2t[i, j] = max_m (sim_matrix[i, j, m])
         形状: (B, B)
    """
    B, M, _ = query_output.shape

    # 1. 投影与 L2 归一化
    query_feat = F.normalize(vision_proj(query_output), dim=-1)           # (B, M, D_embed)
    # 调用方必须保证 input_ids 的第 0 位是 tokenizer 插入的 [CLS] token。
    text_feat = F.normalize(text_proj(text_output[:, 0, :]), dim=-1)      # (B, D_embed) 取 [CLS] token

    # 2. 批量点积计算跨样本相似度 (B_image, B_text, M_queries)
    # einsum 说明: 'b' 表示图像 batch，'m' 表示 query 索引，'c' 表示文本 batch，'d' 表示特征维度
    sim_matrix = torch.einsum("bmd, cd -> bcm", query_feat, text_feat)    # (B, B, M)

    # 3. 取每个图像的 M 个 Query 中与该文本最匹配的那个点积作为相似度
    sim_i2t = sim_matrix.max(dim=-1).values / temperature                  # (B, B)
    sim_t2i = sim_matrix.permute(1, 0, 2).max(dim=-1).values / temperature # (B, B)

    # 4. 对角线上的元素即为正样本标签: [0, 1, 2, ..., B-1]
    labels = torch.arange(B, device=query_output.device)

    # 5. 双向交叉熵求和
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    loss_itc = (loss_i2t + loss_t2i) / 2.0

    return loss_itc, sim_i2t, sim_t2i


def sample_hard_negatives(sim_i2t: torch.Tensor, sim_t2i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【难例挖掘 (Hard Negative Mining)】:
    从相似度矩阵中，针对每一个样本选出“得分最高但不是正样本”的负样本索引。

    这是便于观察的确定性教学简化：BLIP-2 的训练实现通常会按去除
    正样本后的相似度分布进行概率采样。两种方法都要求 batch 至少有
    两个样本；否则不存在有效负例。

    输入:
      sim_i2t: (B, B) 图像到文本的相似度矩阵
      sim_t2i: (B, B) 文本到图像的相似度矩阵
    输出:
      neg_text_indices:  (B,) 针对每张图像挑出的最难文本负例
      neg_image_indices: (B,) 针对每个文本挑出的最难图像负例
    """
    B = sim_i2t.shape[0]
    if B < 2:
        raise ValueError("Hard-negative mining requires batch_size >= 2.")

    # 将正样本对角线位置赋为极小负数，排除正样本自身
    mask = torch.eye(B, device=sim_i2t.device).bool()

    sim_i2t_neg = sim_i2t.clone()
    sim_i2t_neg[mask] = -1e9
    # 对每张图像，挑出除正样本外相似度最高的文本负样本
    neg_text_indices = sim_i2t_neg.argmax(dim=-1)  # (B,)

    sim_t2i_neg = sim_t2i.clone()
    sim_t2i_neg[mask] = -1e9
    # 对每个文本，挑出除正样本外相似度最高的图像负样本
    neg_image_indices = sim_t2i_neg.argmax(dim=-1) # (B,)

    return neg_text_indices, neg_image_indices


def compute_itm_loss(
    qformer: nn.Module,
    image_embeds: torch.Tensor,      # (B, N_img, D_img)
    input_ids: torch.Tensor,         # (B, L)
    attention_mask: torch.Tensor,    # (B, L)
    neg_text_indices: torch.Tensor,  # (B,)
    neg_image_indices: torch.Tensor  # (B,)
) -> torch.Tensor:
    """
    计算图文匹配损失 (ITM)
    组装正样本对 + 难例负样本对，送入 Q-Former (mode='itm') 深度交互，最后做二分类。
    """
    B = image_embeds.shape[0]

    # 1. 组装输入对:
    #   - 正样本 (Image_i, Text_i): 共 B 对
    #   - 负样本 1 (Image_i, Text_neg): 共 B 对 (图像搭配难例负文本)
    #   - 负样本 2 (Image_neg, Text_i): 共 B 对 (难例负图像搭配文本)

    # 图像拼接: (3B, N_img, D_img)
    all_image_embeds = torch.cat([
        image_embeds,
        image_embeds,
        image_embeds[neg_image_indices]
    ], dim=0)

    # 文本拼接: (3B, L)
    all_input_ids = torch.cat([
        input_ids,
        input_ids[neg_text_indices],
        input_ids
    ], dim=0)

    all_attn_mask = torch.cat([
        attention_mask,
        attention_mask[neg_text_indices],
        attention_mask
    ], dim=0)

    # 2. 运行 Q-Former 的 ITM 模式（双向全可见注意力）
    outputs = qformer(
        image_embeds=all_image_embeds,
        input_ids=all_input_ids,
        attention_mask=all_attn_mask,
        mode="itm"
    )

    # query_output: (3B, M, D_q)
    query_output = outputs["query_output"]

    # 3. 通过 itm_head 预测 logits:
    # 预测形状: (3B, M, 2) -> 对 M 个 Query 求平均 -> (3B, 2)
    itm_logits = qformer.itm_head(query_output).mean(dim=1)

    # 4. 构造标签: 前 B 个为 1 (匹配)，后 2B 个为 0 (不匹配)
    itm_labels = torch.cat([
        torch.ones(B, dtype=torch.long, device=image_embeds.device),
        torch.zeros(2 * B, dtype=torch.long, device=image_embeds.device)
    ], dim=0)

    # 5. 计算二分类交叉熵损失
    loss_itm = F.cross_entropy(itm_logits, itm_labels)
    return loss_itm


def compute_itg_loss(
    text_output: torch.Tensor,       # (B, L, D_q) 来自 ITG mode 的文本特征
    input_ids: torch.Tensor,         # (B, L) 文本原始 token id
    lm_head: nn.Linear,              # (D_q -> vocab_size) 语言模型生成头
    pad_token_id: int = 0,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    计算图文生成损失 (ITG - Image-Grounded Text Generation)
    自回归因果语言模型损失，基于前序 token 与视觉 query 预测下一个 token。

    Tensor 演变:
      logits: (B, L-1, vocab_size)
      targets: (B, L-1) 偏移一位
    """
    # 预测下一个 Token: 用位置 t 的输出预测 t+1 位置的 input_id
    logits = lm_head(text_output[:, :-1, :])      # (B, L-1, vocab_size)
    targets = input_ids[:, 1:]                    # (B, L-1)

    # 优先使用显式 attention mask，而不是假设所有 tokenizer 都以 0 作为 pad。
    # targets 的位置 t 对应输入 mask 的位置 t + 1。
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        targets = targets.masked_fill(~attention_mask[:, 1:].bool(), -100)
        ignore_index = -100
    else:
        # 保留旧接口的教学回退：未提供 mask 时才使用 pad_token_id。
        ignore_index = pad_token_id

    # 展平计算交叉熵，忽略 padding target
    loss_itg = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index
    )
    return loss_itg
