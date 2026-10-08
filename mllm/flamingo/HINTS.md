# 分层提示

先看第一级；仍卡住再展开第二级。第三级提供可用算式或伪代码，完整答案仍在 reference。验收脚本中的 expected 是语义示例，不需要先读参考模型。

## 1：resampler

<details><summary>一级：想清楚什么</summary>

每幅图独立压缩，R来自learned latents，和N无关。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

projection后合并B/M；learned latents expand(B*M,R,D)；查memory后加residual，再逐层refine。

</details>

<details><summary>三级：算式与伪代码</summary>

`memory=vision_proj(features).reshape(B*M,N,D)`；`x=latents.expand(B*M,-1,-1)`；`x+=cross_attn(cross_norm(x),memory)`；循环blocks；reshape(B,M,R,D)。这是toy的简化压缩器。

</details>

## 2：media mask

<details><summary>一级：想清楚什么</summary>

本关实现all-seen：第二张图出现后允许第1、第2张图；第一张图前完全不查视觉。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

先只计有效sentinel的累计数量，再与0-based image编号比较，最后把每图许可扩为R份。

</details>

<details><summary>三级：算式与伪代码</summary>

`seen=((ids==sentinel)&valid).long().cumsum(1)`；`allowed=arange(M)[None,None,:]<seen[...,None]`；`allowed.repeat_interleave(R,-1)&valid[...,None]`。先校验每行sentinel数=M；原论文immediate policy使用等于seen-1。

</details>

## 3：gate

<details><summary>一级：想清楚什么</summary>

初始两门为0使connector严格恒等；gate一旦变化，才向视觉支路传播有效梯度。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

先cross_attn(norm1(x),memory)，再tanh(attn_gate) residual；更新后的x计算FFN residual。

</details>

<details><summary>三级：算式与伪代码</summary>

`x=x+attn_gate.tanh()*cross_attn(norm1(x),context,attention_mask=allow,key_padding_mask=context_padding_mask)`；返回`x+ff_gate.tanh()*mlp(norm2(x))`。全遮蔽query由共享attention返回0，不需自己softmax。

</details>

## 4：LM插层与训练

<details><summary>一级：想清楚什么</summary>

冻结参数和no_grad并不等价：语言层参数不更新，但中间连接器仍需要链式梯度。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

加token与position embeddings；遍历block，传causal和有效文本mask；按字符串key查connector。

</details>

<details><summary>三级：算式与伪代码</summary>

`for i,block in enumerate(blocks): x=block(x,key_padding_mask=valid,causal=True); if str(i) in connectors: x=connectors[str(i)](x,memory,attention_mask=media)`；最终lm_head(norm(x))。训练顺序zero_grad/forward/backward/step，返回out；模型外围负责shift。

</details>
