# 从零构建 BLIP-2 (Coding BLIP-2 From Scratch)

> 共享基础包：先执行 `python3 -m pip install -e "../../general_utils/edu_core[dev]"`。顶层 mock 实现服务机制教学；真实视觉塔、LLM 与数据流程见 `recipe/`，不应与 toy demo 混同。

欢迎开启 BLIP-2 的手写实战！BLIP-2 的精髓在于 **“用极低的参数量（Q-Former），架起冻结的视觉大模型（ViT）与冻结的语言大模型（LLM）之间的语义桥梁”**。

---

## 目录结构规划

我们将整个 BLIP-2 拆解为以下几个递进的模块：

```text
blip2/
├── lesson1_*.py ... lesson5_*.py  # 保留 TODO 的顺序练习题
├── practice_blip2.py              # 独立的 Multi-Head Attention 热身题
├── attention_masks.py              # 参考解：ITC / ITM / ITG 注意力掩码
├── qformer_layer.py, qformer.py    # 参考解：Q-Former 基础层与骨干
├── stage1_losses.py                # 参考解：ITC、ITM、ITG 损失
├── stage2_blip2.py                 # 参考解：冻结骨干、Q-Former、投影与 LLM
└── run_toy_demo.py                 # 仅验证参考解的端到端 toy demo
```

---

## 核心张量维度约定（Tensor Dimension Cheat-Sheet）

在编写代码时，请时刻牢记以下关键变量与维度标记：

| 变量名 | 含义 | 典型取值 |
| :--- | :--- | :--- |
| `B` (Batch Size) | 批次大小 | 4 或 8 |
| `M` (`num_query_tokens`) | 可学习的查询向量数量 | **32** |
| `L` (`seq_len_text`) | 文本序列长度 (Prompt/Caption) | 16 ~ 64 |
| `N_img` (`num_patches`) | 图像 Patch 数量 (如 ViT 14x14+1) | **257** |
| `D_q` (`hidden_size`) | Q-Former 内部特征隐藏层维度 | **768** (BERT-base 大小) |
| `D_img` (`img_feat_dim`)| 视觉编码器输出特征维度 | **1408** (EVA-CLIP) 或 768/1024 |
| `D_llm` (`llm_dim`) | LLM 输入的 Token Embedding 维度 | **2048** (OPT-2.7B) 或 4096 (Llama-7B) |

---

## 学习路线图

1. **第一步（Attention Mask）**：搞清楚为什么 Q-Former 输入 `[Query, Text]` 拼合序列时，需要三种不同的 Mask。
2. **第二步（QFormer Layer）**：手写 Cross-Attention 与带 Mask 的 Self-Attention，特别注意：**只有 Query 走 Cross-Attention 查图像，Text 不查图像**！
3. **第三步（QFormer 整体组装）**：初始化 `self.query_tokens`，拼接 Query 和 Text 的 Embedding，送入 N 层 Layer。
4. **第四步（Stage 1 损失函数）**：实现图文对比 (ITC)、难例挖掘图文匹配 (ITM) 和自回归因果文本生成 (ITG)。
5. **第五步（Stage 2 对齐 LLM）**：用一层 `Linear(D_q, D_llm)` 将 32 个 Query 映射到 LLM 空间，前缀拼接 Prompt 送进 LLM 生成。

---

## 教学简化与使用约定

- 本项目是**机制教学实现**：`MockVisionEncoder` 和 `MockLLM` 只用于观察数据流，不等同于可直接替换真实 EVA-CLIP、OPT 或 LLaMA 的训练代码。
- Stage 2 冻结 ViT 和 LLM 的参数及训练态；**Q-Former 与 `llm_proj` 仍然可训练**，不是只有投影层可训练。
- Stage 2 的 `forward` 接受可选 `prompt_attention_mask` 与 `answer_attention_mask`；其中 `1` 表示有效 token、`0` 表示 padding。padding 不参与注意力或损失。`generate` 的 batch prompt 使用标准右侧 padding。
- ITC 将 `text_output[:, 0]` 视为文本 `[CLS]` 表征，因此调用方必须保证 `input_ids[:, 0]` 是 tokenizer 插入的 `[CLS]`。
- hard-negative 挖掘为便于复现的确定性 argmax 版本；真实 BLIP-2 常按相似度概率分布采样，且两者都要求 batch size 至少为 2。
- `lesson*.py` 预期在未完成时抛出 `NotImplementedError`；完成后可与无前缀的参考解逐项对照。运行 `run_toy_demo.py` 验证参考解，而不是练习题文件。
