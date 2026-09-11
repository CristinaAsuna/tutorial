"""
lesson4_stage1_losses.py
========================
【关卡 4】：Stage 1 预训练三大损失与难例挖掘 (Hard Negative Mining)

本文件需要你实现 BLIP-2 Stage 1 的核心训练目标:
1. compute_itc_loss: 图文对比损失 (Query 与 Text [CLS] 的 Max Pooling 匹配)
2. sample_hard_negatives: 难例挖掘算法 (在 Batch 内寻找最容易混淆的负样本)
3. compute_itm_loss: 深度图文匹配二分类损失 (3B 样本组合)
4. compute_itg_loss: 图像条件文本因果生成损失

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson4_stage1_losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from qformer import QFormer


# ==============================================================================
# 目标 1: ITC (Image-Text Contrastive) Loss
# ==============================================================================
def compute_itc_loss(
    query_output: torch.Tensor,      # (B, M, D_q)
    text_output: torch.Tensor,       # (B, L, D_q)
    vision_proj: nn.Linear,          # (D_q -> embed_dim)
    text_proj: nn.Linear,            # (D_q -> embed_dim)
    temperature: float = 0.07
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    【数学推导】:
      1. query_feat: 对 vision_proj(query_output) 做 F.normalize(..., dim=-1) -> (B, M, D)
      2. text_feat: 提取 text_output[:, 0, :] ([CLS] 位置) 做归一化 -> (B, D)
      3. 相似度矩阵: sim_matrix[i, j, m] = query_feat[i, m] · text_feat[j] -> 形状 (B, B, M)
      4. 取所有 Query 中的最大相似度: sim_i2t = max_m(sim_matrix) / temp -> (B, B)
      5. 对角线索引为正样本目标: labels = torch.arange(B)
      6. 双向交叉熵平均: (CE(sim_i2t, labels) + CE(sim_t2i, labels)) / 2
    """
    # =========================================================================
    # TODO 4.1: 请实现 ITC 损失计算
    # 步骤提示:
    # 1. B, M = query_output.shape[:2]
    # 2. query_feat = F.normalize(vision_proj(query_output), dim=-1)
    # 3. text_feat = F.normalize(text_proj(text_output[:, 0, :]), dim=-1)
    # 4. sim_matrix = torch.einsum("bmd, cd -> bcm", query_feat, text_feat)  # (B, B, M)
    # 5. sim_i2t = sim_matrix.max(dim=-1).values / temperature              # (B, B)
    #    sim_t2i = sim_matrix.permute(1, 0, 2).max(dim=-1).values / temperature
    # 6. labels = torch.arange(B, device=query_output.device)
    # 7. loss = (F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_t2i, labels)) / 2.0
    # 8. 返回 loss, sim_i2t, sim_t2i
    # =========================================================================
    raise NotImplementedError("TODO 4.1 尚未实现！请实现 compute_itc_loss")


# ==============================================================================
# 目标 2: 难例挖掘 (Hard Negative Mining)
# ==============================================================================
def sample_hard_negatives(sim_i2t: torch.Tensor, sim_t2i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【算法逻辑】:
      正样本在相似度矩阵对角线上 (i == j)。
      对于第 i 张图像，要在所有 j != i 的文本中，找到相似度最高的那个文本索引 (最难负样本)！
    """
    # =========================================================================
    # TODO 4.2: 请实现难例挖掘
    # 步骤提示:
    # 1. B = sim_i2t.shape[0]
    # 2. 构造对角线掩码: mask = torch.eye(B, device=sim_i2t.device).bool()
    # 3. 复制矩阵并把对角线赋值为 -1e9 (排除正样本):
    #    s_i2t = sim_i2t.clone()
    #    s_i2t[mask] = -1e9
    #    neg_text_indices = s_i2t.argmax(dim=-1)   # 每张图最像的负文本索引 (B,)
    # 4. 同理处理 sim_t2i 得到 neg_image_indices (B,)
    # 5. 返回 neg_text_indices, neg_image_indices
    # =========================================================================
    raise NotImplementedError("TODO 4.2 尚未实现！请实现 sample_hard_negatives")


# ==============================================================================
# 目标 3: ITM (Image-Text Matching) Loss
# ==============================================================================
def compute_itm_loss(
    qformer: nn.Module,
    image_embeds: torch.Tensor,      # (B, N_img, D_img)
    input_ids: torch.Tensor,         # (B, L)
    attention_mask: torch.Tensor,    # (B, L)
    neg_text_indices: torch.Tensor,  # (B,)
    neg_image_indices: torch.Tensor  # (B,)
) -> torch.Tensor:
    """
    【算法逻辑】:
      组装 3 组样本对 (共 3B 个样本):
        1. 正样本对: (Image, Text) - 标签 1
        2. 难例负样本 1: (Image, Text_neg) - 标签 0
        3. 难例负样本 2: (Image_neg, Text) - 标签 0
      送入 Q-Former (mode='itm')，取输出的 Query 特征过 itm_head 做二分类。
    """
    # =========================================================================
    # TODO 4.3: 请实现 ITM 损失计算
    # 步骤提示:
    # 1. B = image_embeds.shape[0]
    # 2. 拼接 3B 个图像特征: [image_embeds, image_embeds, image_embeds[neg_image_indices]]
    # 3. 拼接 3B 个文本 input_ids: [input_ids, input_ids[neg_text_indices], input_ids]
    # 4. 拼接 3B 个 attention_mask: [attention_mask, attention_mask[neg_text_indices], attention_mask]
    # 5. 送入 qformer(image_embeds=all_img, input_ids=all_ids, attention_mask=all_mask, mode='itm')
    # 6. itm_logits = qformer.itm_head(outputs['query_output']).mean(dim=1) -> 形状 (3B, 2)
    # 7. labels: 前 B 个为 1，后 2B 个为 0
    # 8. 返回 F.cross_entropy(itm_logits, labels)
    # =========================================================================
    raise NotImplementedError("TODO 4.3 尚未实现！请实现 compute_itm_loss")


# ==============================================================================
# 目标 4: ITG (Image-Grounded Text Generation) Loss
# ==============================================================================
def compute_itg_loss(
    text_output: torch.Tensor,       # (B, L, D_q) 来自 mode='itg'
    input_ids: torch.Tensor,         # (B, L) 原始 token ids
    lm_head: nn.Linear,              # (D_q -> vocab_size)
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    【自回归 LM 损失推导】:
      用时刻 t 的特征预测时刻 t+1 的 token。
      - logits: lm_head(text_output[:, :-1, :]) -> (B, L-1, vocab_size)
      - targets: input_ids[:, 1:]               -> (B, L-1)
    """
    # =========================================================================
    # TODO 4.4: 请实现 ITG 因果生成损失
    # 步骤提示:
    # 1. logits = lm_head(text_output[:, :-1, :])
    # 2. targets = input_ids[:, 1:]
    # 3. F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=pad_token_id)
    # =========================================================================
    raise NotImplementedError("TODO 4.4 尚未实现！请实现 compute_itg_loss")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试关卡 4 ==========")
    B, M, L, D_q = 4, 4, 6, 128
    vocab_size = 500

    qformer = QFormer(num_query_tokens=M, hidden_size=D_q, num_layers=2, num_heads=4, img_feat_dim=256, vocab_size=vocab_size, embed_dim=64)
    query_out = torch.randn(B, M, D_q)
    text_out = torch.randn(B, L, D_q)

    # 1. 测试 ITC
    loss_itc, sim_i2t, sim_t2i = compute_itc_loss(query_out, text_out, qformer.vision_proj, qformer.text_proj)
    assert sim_i2t.shape == (B, B), "相似度矩阵形状错误"
    assert loss_itc.item() > 0, "ITC Loss 必须大于 0"
    print("✅ TODO 4.1 (compute_itc_loss) 通过测试！")

    # 2. 测试难例挖掘
    neg_text_idx, neg_img_idx = sample_hard_negatives(sim_i2t, sim_t2i)
    assert neg_text_idx.shape == (B,) and neg_img_idx.shape == (B,)
    # 保证没有挑中自己
    for i in range(B):
        assert neg_text_idx[i] != i, "难例挖掘不能选择样本自身！"
    print("✅ TODO 4.2 (sample_hard_negatives) 通过测试！")

    # 3. 测试 ITM
    dummy_img = torch.randn(B, 10, 256)
    dummy_ids = torch.randint(0, vocab_size, (B, L))
    dummy_mask = torch.ones((B, L), dtype=torch.long)
    loss_itm = compute_itm_loss(qformer, dummy_img, dummy_ids, dummy_mask, neg_text_idx, neg_img_idx)
    assert loss_itm.item() > 0
    print("✅ TODO 4.3 (compute_itm_loss) 通过测试！")

    # 4. 测试 ITG
    lm_head = nn.Linear(D_q, vocab_size)
    loss_itg = compute_itg_loss(text_out, dummy_ids, lm_head)
    assert loss_itg.item() > 0
    print("✅ TODO 4.4 (compute_itg_loss) 通过测试！")
    print("🎉 恭喜！关卡 4 全部挑战成功！\n")


if __name__ == "__main__":
    run_test()
