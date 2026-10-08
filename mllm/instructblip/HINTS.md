# 分层提示

先完成关卡局部检查，再打开下一层；三级是伪代码提示，完整答案留在 reference。

## 关卡 1：联合 query/instruction

<details><summary>一级：方向</summary>

query 是参数，batch expand 应共享梯度。

</details>

<details><summary>二级：步骤</summary>

cat 轴是 dim=1；instruction mask 前插 M 个 True。

</details>

<details><summary>三级：伪代码</summary>

B=instruction_embeds.size(0)；query_tokens.expand(B,-1,-1)，返回 cat(tokens) 与 cat(valid)。

</details>

## 关卡 2：query-only cross-attention

<details><summary>一级：方向</summary>

先区分整段 self-attention 与 query 段 cross-attention。

</details>

<details><summary>二级：步骤</summary>

前 M 个位置加视觉 residual；剩余 text 必须逐元素保持原样。

</details>

<details><summary>三级：伪代码</summary>

query=hidden[:,:M]；updated=query+cross_attn(norm(query),image)，cat(updated,hidden[:,M:])。

</details>

## 关卡 3：LLM prefix 与双 tokenizer

<details><summary>一级：方向</summary>

指令只进 Q-Former；prompt/answer 使用 LLM embedding。

</details>

<details><summary>二级：步骤</summary>

每行分别用 mask 取有效 prompt/answer，不能把 padding 留在中间。

</details>

<details><summary>三级：伪代码</summary>

依次实现 project_visual_queries（调用投影）、concatenate_prefix_row（沿 token 轴 cat）、prefix_answer_labels（前 M+P 个 -100，之后有效 answer ids）。外层去 padding、embedding、循环与 pad_sequence 已提供。

</details>

## 关卡 4：冻结专家训练

<details><summary>一级：方向</summary>

冻结 LLM 参数仍应让 loss 反传到输入 prefix。

</details>

<details><summary>二级：步骤</summary>

不要给 LLM forward 加 no_grad；仅视觉编码器是 no_grad。

</details>

<details><summary>三级：伪代码</summary>

train/zero_grad/forward/loss.backward/step/return detached loss；确认只有 Q-Former/投影梯度。

</details>
