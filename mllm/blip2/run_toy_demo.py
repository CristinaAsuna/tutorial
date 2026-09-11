"""
run_toy_demo.py
===============
端到端运行验证脚本：带你一步步看清每个模块的数据流动与 Tensor Shape！
无需下载任何庞大权重，使用合成假数据（Toy Tensors）即刻跑通。
"""

import torch
import torch.nn as nn
from attention_masks import create_itc_mask, create_itm_mask, create_itg_mask
from qformer_layer import QFormerLayer
from qformer import QFormer
from stage1_losses import compute_itc_loss, sample_hard_negatives, compute_itm_loss, compute_itg_loss
from stage2_blip2 import MockVisionEncoder, MockLLM, Blip2ForConditionalGeneration


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_attention_masks():
    section("1. 验证三大注意力掩码 (Attention Masks)")
    B, M, L = 2, 4, 3
    dummy_text_mask = torch.ones((B, L), dtype=torch.long)
    # 把最后一个 token 设为 pad
    dummy_text_mask[:, -1] = 0

    itc_mask = create_itc_mask(B, M, dummy_text_mask)
    itm_mask = create_itm_mask(B, M, dummy_text_mask)
    itg_mask = create_itg_mask(B, M, dummy_text_mask)

    print(f"Batch={B}, Query={M}, Text={L} (包含 1 个 Pad Token)")
    print(f"[ITC Mask 形状]: {itc_mask.shape} (预期: [{B}, 1, {M+L}, {M+L}])")
    print(f"[ITM Mask 形状]: {itm_mask.shape} (预期: [{B}, 1, {M+L}, {M+L}])")
    print(f"[ITG Mask 形状]: {itg_mask.shape} (预期: [{B}, 1, {M+L}, {M+L}])")

    # 语义断言：不能只检查 shape；必须验证三种可见性规则。
    assert not itc_mask[:, :, :M, M:].any() and not itc_mask[:, :, M:, :M].any()
    assert not itm_mask[:, :, :, -1].any(), "ITM 不应把 pad 作为 Key/Value"
    assert not itg_mask[:, :, :M, M:].any(), "ITG Query 不可偷看文本"
    assert not itg_mask[:, :, M, M + 1:].any(), "ITG 首个文本 token 不可看未来文本"

    print("\n观察 ITG 掩码首个样本的注意力可见性矩阵 (1=可见, 0=遮蔽):")
    print(itg_mask[0, 0].cpu().numpy())


def test_qformer_layer():
    section("2. 验证单个 QFormerLayer (含 Query 专享 Cross-Attention)")
    B, M, L = 2, 4, 3
    D_q = 768
    N_img, D_img = 16, 1408

    layer = QFormerLayer(hidden_size=D_q, num_heads=8, img_feat_dim=D_img)

    # 模拟 [Query, Text] 拼接后的 hidden_states
    x = torch.randn(B, M + L, D_q)
    image_embeds = torch.randn(B, N_img, D_img)
    mask = torch.ones(B, 1, M + L, M + L)

    out = layer(hidden_states=x, num_query_tokens=M, attention_mask=mask, image_embeds=image_embeds)
    print(f"输入 hidden_states 形状: {x.shape}")
    print(f"输入 image_embeds 形状:  {image_embeds.shape}")
    print(f"输出 hidden_states 形状: {out.shape} (严格保持相同维度)")


def test_qformer_and_stage1_losses():
    section("3. 验证完整 Q-Former 与 Stage 1 三大对齐损失 (ITC / ITM / ITG)")
    B, M, L = 4, 8, 6
    D_q = 256  # 测试用较小维度加速
    N_img, D_img = 10, 512
    vocab_size = 1000

    qformer = QFormer(
        num_query_tokens=M,
        hidden_size=D_q,
        num_layers=2,
        num_heads=4,
        img_feat_dim=D_img,
        vocab_size=vocab_size,
        embed_dim=128
    )

    image_embeds = torch.randn(B, N_img, D_img)
    input_ids = torch.randint(2, vocab_size, (B, L))
    input_ids[:, 0] = 1  # 教学约定：第 0 位是 [CLS]。
    attention_mask = torch.ones((B, L), dtype=torch.long)

    # 3.1 纯视觉查询模式 (Stage 2 使用)
    visual_queries = qformer.extract_visual_queries(image_embeds)
    print(f"[纯视觉提取] 输出 Query 形状: {visual_queries.shape} (预期: [{B}, {M}, {D_q}])")

    # 3.2 ITC 对比学习
    itc_out = qformer(image_embeds, input_ids, attention_mask, mode="itc")
    loss_itc, sim_i2t, sim_t2i = compute_itc_loss(
        query_output=itc_out["query_output"],
        text_output=itc_out["text_output"],
        vision_proj=qformer.vision_proj,
        text_proj=qformer.text_proj
    )
    print(f"[ITC 损失计算成功]: Loss = {loss_itc.item():.4f}, 相似度矩阵形状 = {sim_i2t.shape}")

    # 3.3 难例挖掘与 ITM 损失
    neg_text_idx, neg_img_idx = sample_hard_negatives(sim_i2t, sim_t2i)
    loss_itm = compute_itm_loss(
        qformer=qformer,
        image_embeds=image_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        neg_text_indices=neg_text_idx,
        neg_image_indices=neg_img_idx
    )
    print(f"[ITM 难例损失计算成功]: Loss = {loss_itm.item():.4f} (样本量扩展为 3xBatch)")

    # 3.4 ITG 文本因果生成损失
    itg_out = qformer(image_embeds, input_ids, attention_mask, mode="itg")
    lm_head = nn.Linear(D_q, vocab_size)
    loss_itg = compute_itg_loss(itg_out["text_output"], input_ids, lm_head, attention_mask=attention_mask)
    print(f"[ITG 文本因果损失成功]: Loss = {loss_itg.item():.4f}")

    try:
        sample_hard_negatives(sim_i2t[:1, :1], sim_t2i[:1, :1])
        raise AssertionError("B=1 必须没有合法 hard negative")
    except ValueError:
        pass


def test_stage2_pipeline():
    section("4. 验证 Stage 2: 冻结骨干 + Linear 投影 + LLM 训练与推理生成")
    B, M = 2, 8
    D_q = 256
    D_img = 512
    D_llm = 512
    vocab_size = 1000

    # 模拟模块
    vision_encoder = MockVisionEncoder(img_feat_dim=D_img)
    qformer = QFormer(num_query_tokens=M, hidden_size=D_q, num_layers=2, num_heads=4, img_feat_dim=D_img)
    llm = MockLLM(vocab_size=vocab_size, llm_dim=D_llm)

    model = Blip2ForConditionalGeneration(
        vision_encoder=vision_encoder,
        qformer=qformer,
        llm=llm,
        qformer_dim=D_q,
        llm_dim=D_llm
    )

    # 验证权重冻结状态以及 train() 不会重新打开冻结模块的 dropout/BatchNorm。
    model.train()
    vi_frozen = all(not p.requires_grad for p in model.vision_encoder.parameters())
    llm_frozen = all(not p.requires_grad for p in model.llm.parameters())
    proj_trainable = model.llm_proj.weight.requires_grad
    assert not model.vision_encoder.training and not model.llm.training
    print(f"权重冻结验证: ViT 冻结={vi_frozen}, LLM 冻结={llm_frozen}, Linear 投影层可训练={proj_trainable}")

    # 模拟数据
    pixel_values = torch.randn(B, 3, 224, 224)
    prompt_ids = torch.randint(0, vocab_size, (B, 5))   # 5 个 prompt token
    answer_ids = torch.randint(0, vocab_size, (B, 4))   # 4 个 answer token

    # 前向传播与反向传播
    outputs = model(pixel_values, prompt_ids, answer_ids)
    loss = outputs["loss"]
    loss.backward()
    print(f"[Stage 2 前向与反向传播成功]: Loss = {loss.item():.4f}")
    print(f"Linear 投影层梯度正常存在: norm = {model.llm_proj.weight.grad.norm().item():.4f}")
    assert model.qformer.query_tokens.grad is not None, "Q-Former 必须仍能获得梯度"

    # padding 不作为 LLM 的可见 token，也不作为监督目标；改变 pad id 不应影响 loss。
    model.eval()
    padded_prompt = prompt_ids.clone()
    padded_answer = answer_ids.clone()
    prompt_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    answer_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    padded_prompt[prompt_mask == 0] = 3
    padded_answer[answer_mask == 0] = 5
    padded_out = model(pixel_values, padded_prompt, padded_answer, prompt_mask, answer_mask)
    changed_prompt, changed_answer = padded_prompt.clone(), padded_answer.clone()
    changed_prompt[prompt_mask == 0] = 9
    changed_answer[answer_mask == 0] = 11
    changed_out = model(pixel_values, changed_prompt, changed_answer, prompt_mask, answer_mask)
    for row in range(B):
        answer_start = M + int(prompt_mask[row].sum())
        assert torch.all(padded_out["labels"][row, answer_start:answer_start + 4][answer_mask[row] == 0] == -100)
    assert torch.allclose(padded_out["loss"], changed_out["loss"], atol=1e-6)
    print("[Stage 2 Padding Mask 成功]: pad 不可见且不计入损失")

    # 推理生成测试
    generated_ids = model.generate(pixel_values, prompt_ids, max_new_tokens=6)
    print(f"[Stage 2 贪心生成成功]: 生成 Token ID 形状 = {generated_ids.shape} (预期: [{B}, 6])")
    generated_again = model.generate(pixel_values, prompt_ids, max_new_tokens=6)
    assert torch.equal(generated_ids, generated_again), "eval 状态下贪心生成应可复现"


if __name__ == "__main__":
    test_attention_masks()
    test_qformer_layer()
    test_qformer_and_stage1_losses()
    test_stage2_pipeline()
    section("🎉 所有 BLIP-2 关键组件单元测试全部通过！")
