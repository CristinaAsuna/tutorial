"""
lesson5_stage2_blip2.py
=======================
【关卡 5】：Stage 2 生成式对齐与大模型生成 (Vision-to-Language Generation)

这是 BLIP-2 的收官决战！
我们将把冻结的图像编码器 (ViT)、训练好的 Q-Former、可学习的投影层 (Linear) 与冻结的大模型 (LLM) 拼接在一起！

本文件需要你实现 3 个关键部分:
1. freeze_modules: 参数冻结策略 (确保 ViT 和 LLM 不参与反向传播梯度更新)
2. forward: 训练时软前缀 (Soft Prompt) 拼接、-100 标签掩码构造与因果 Loss
3. generate: 推理时的贪心自回归文本生成循环

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson5_stage2_blip2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from qformer import QFormer
from stage2_blip2 import MockVisionEncoder, MockLLM


class Blip2ForConditionalGeneration(nn.Module):
    """
    BLIP-2 顶层模型

    【__init__ 参数】:
      - vision_encoder: 冻结的视觉特征提取器
      - qformer: 负责语义抽取的 Q-Former
      - llm: 冻结的大语言模型
      - qformer_dim: Q-Former 维度 D_q (如 768)
      - llm_dim: LLM 词嵌入维度 D_llm (如 2048)
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

        # 核心桥梁层: 唯一需要训练的线性映射层
        self.llm_proj = nn.Linear(qformer_dim, llm_dim)

        self.freeze_modules()

    def freeze_modules(self):
        """
        【参数冻结】:
          按 BLIP-2 论文设定，vision_encoder 和 llm 必须全参数冻结！
        """
        # =========================================================================
        # TODO 5.1: 请冻结 vision_encoder 和 llm 的所有参数
        # 提示: 遍历 self.vision_encoder.parameters() 和 self.llm.parameters()，
        #       将 param.requires_grad 设为 False
        # =========================================================================
        raise NotImplementedError("TODO 5.1 尚未实现！请实现 freeze_modules")

    def forward(
        self,
        pixel_values: torch.Tensor,     # (B, 3, H, W)
        prompt_input_ids: torch.Tensor, # (B, L_p) 问题文本 token ids
        answer_input_ids: torch.Tensor  # (B, L_a) 回答文本 token ids (用于求 Loss)
    ) -> Dict[str, torch.Tensor]:
        """
        【训练前向传播与 Loss 计算】:

        序列组装示意图:
          Embedding: [ Visual Queries (M个) | Prompt (L_p个) | Answer (L_a个) ]
          Labels:    [      -100 (忽略)     |   -100 (忽略)  | Answer (L_a个) ]
        """
        # =========================================================================
        # TODO 5.2: 请实现 Stage 2 训练前向过程
        # 步骤提示:
        # 1. 在 torch.no_grad() 保护下，提取视觉特征:
        #    image_embeds = self.vision_encoder(pixel_values)  # (B, N_img, D_img)
        # 2. Q-Former 提取 32 个视觉 Query:
        #    query_output = self.qformer.extract_visual_queries(image_embeds) # (B, M, D_q)
        # 3. 线性投影到 LLM 空间:
        #    projected_query = self.llm_proj(query_output)     # (B, M, D_llm)
        # 4. 准备文本嵌入:
        #    text_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1) # (B, L_p + L_a)
        #    text_embeds = self.llm.embed_tokens(text_ids)    # (B, L_p + L_a, D_llm)
        # 5. 拼接全体作为 LLM 输入:
        #    inputs_embeds = torch.cat([projected_query, text_embeds], dim=1)  # (B, M + L_p + L_a, D_llm)
        # 6. 构造 Labels:
        #    前 M + L_p 个位置填 -100: ignore = torch.full((B, M + L_p), -100, device=..., dtype=torch.long)
        #    labels = torch.cat([ignore, answer_input_ids], dim=1)
        # 7. logits = self.llm(inputs_embeds)
        # 8. 自回归错位一位计算 Cross Entropy (ignore_index=-100):
        #    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100)
        # 9. 返回 {"loss": loss, "logits": logits}
        # =========================================================================
        raise NotImplementedError("TODO 5.2 尚未实现！请实现 Blip2ForConditionalGeneration.forward")

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        max_new_tokens: int = 10
    ) -> torch.Tensor:
        """
        【推理生成：贪心自回归搜索】:
        """
        # =========================================================================
        # TODO 5.3: 请实现推理生成循环
        # 步骤提示:
        # 1. 抽取视觉特征 -> Q-Former 抽取 -> self.llm_proj 投影为 projected_query (B, M, D_llm)
        # 2. 文本 Prompt 嵌入: prompt_embeds = self.llm.embed_tokens(prompt_input_ids)
        # 3. 初始输入: curr_embeds = torch.cat([projected_query, prompt_embeds], dim=1)
        # 4. 循环 max_new_tokens 次:
        #    a. logits = self.llm(curr_embeds)
        #    b. next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True) # (B, 1)
        #    c. 记录 generated_ids.append(next_id)
        #    d. next_embed = self.llm.embed_tokens(next_id)
        #    e. curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)
        # 5. 拼接所有生成的 token id 并返回: torch.cat(generated_ids, dim=1) -> (B, max_new_tokens)
        # =========================================================================
        raise NotImplementedError("TODO 5.3 尚未实现！请实现 Blip2ForConditionalGeneration.generate")


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试关卡 5 ==========")
    B, M, D_q, D_img, D_llm, vocab_size = 2, 4, 128, 256, 128, 500

    vision_encoder = MockVisionEncoder(img_feat_dim=D_img)
    qformer = QFormer(num_query_tokens=M, hidden_size=D_q, num_layers=2, num_heads=4, img_feat_dim=D_img, vocab_size=vocab_size)
    llm = MockLLM(vocab_size=vocab_size, llm_dim=D_llm)

    model = Blip2ForConditionalGeneration(vision_encoder, qformer, llm, qformer_dim=D_q, llm_dim=D_llm)

    # 1. 验证冻结
    assert all(not p.requires_grad for p in model.vision_encoder.parameters()), "ViT 必须全冻结！"
    assert all(not p.requires_grad for p in model.llm.parameters()), "LLM 必须全冻结！"
    assert model.llm_proj.weight.requires_grad, "Linear 投影层必须可训练！"
    print("✅ TODO 5.1 (freeze_modules) 通过测试！")

    # 2. 验证前向传播与 Loss 反向传播
    img = torch.randn(B, 3, 224, 224)
    prompt = torch.randint(0, vocab_size, (B, 3))
    answer = torch.randint(0, vocab_size, (B, 4))
    outputs = model(img, prompt, answer)
    assert "loss" in outputs and outputs["loss"].item() > 0
    outputs["loss"].backward()
    assert model.llm_proj.weight.grad is not None, "投影层必须能收到梯度更新！"
    print("✅ TODO 5.2 (前向与 Loss 反向传播) 通过测试！")

    # 3. 验证生成
    gen_ids = model.generate(img, prompt, max_new_tokens=5)
    assert gen_ids.shape == (B, 5), f"生成序列形状不符: {gen_ids.shape}"
    print("✅ TODO 5.3 (generate 贪心生成) 通过测试！")
    print("🏆 恭喜通关！你已经完整从零实现了整个 BLIP-2 多模态大模型！\n")


if __name__ == "__main__":
    run_test()
