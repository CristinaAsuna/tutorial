# 分层提示

先看第一级；仍卡住再展开第二级。第三级提供可用算式或伪代码，完整答案仍在 reference。验收脚本中的 expected 是语义示例，不需要先读参考模型。

## 1：future chunk

<details><summary>一级：想清楚什么</summary>

每个窗口由“起始时刻+相对offset”决定；末尾无效位置仍需一个合法索引。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

先建(T,K)索引矩阵，再用它一次索引 actions；valid 不依赖动作值。

</details>

<details><summary>三级：算式与伪代码</summary>

`index=arange(T)[:,None]+arange(K)[None,:]`；`chunks=actions[:,index.clamp_max(T-1)]`，`valid=(index<T).expand(B,-1,-1)`。

</details>

## 2：posterior token packing

<details><summary>一级：想清楚什么</summary>

CLS摘要要看到当前qpos和所有有效未来动作；无效动作不能改变posterior。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

CLS expand B；qpos投影后unsqueeze(1)；action投影保留K轴；前两个mask位为True。

</details>

<details><summary>三级：算式与伪代码</summary>

拼 `[CLS,qpos,actions]`；`output=encoder(tokens+pos,src_key_padding_mask=~valid)`；`mean(output[:,0])`和`logvar(output[:,0])`。

</details>

## 2：重参数化与loss

<details><summary>一级：想清楚什么</summary>

标准差是exp(logvar/2)，KL按latent维求和；重建只平均有效标量。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

先扩展mask到(B,K,A)，布尔索引prediction/target；计算每个样本的KL再batch平均。

</details>

<details><summary>三级：算式与伪代码</summary>

`z=mean+exp(.5*logvar)*noise`；`KL=.5*(mean.square()+logvar.exp()-1-logvar).sum(-1).mean()`；返回`(L1+beta*KL,L1,KL)`。无有效mask时抛ValueError。

</details>

## 3：observation memory

<details><summary>一级：想清楚什么</summary>

视觉patch需要知道空间位置与来自哪台相机；qpos/z各贡献一个token。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

Conv前合并B和Cam，Conv后flatten(2).transpose(1,2)；恢复(B,Cam,N,D)再加两种embedding。

</details>

<details><summary>三级：算式与伪代码</summary>

`features+=spatial_pos[:,None]+camera_embed(arange(Cam))[None,:,None]`；恢复(B,Cam*N,D)，拼`qpos_proj(qpos)[:,None]`及`z_proj(z)[:,None]`；送memory_encoder。

</details>

## 3：action queries

<details><summary>一级：想清楚什么</summary>

K个query是输出位置，不是已知的未来动作；全部并行解码。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

query expand B；decoder的tgt是queries，memory是观测；动作头将D映射A。

</details>

<details><summary>三级：算式与伪代码</summary>

`action_head(decoder(action_queries.expand(B,-1,-1),observation_memory))`。不加因果mask，不做语言token shift。

</details>

## 4：temporal ensemble

<details><summary>一级：想清楚什么</summary>

历史每个chunk的不同offset指向同一当前时刻。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

oldest->newest，n个历史时chunk_i的offset为n-1-i；权重i=0对应oldest。

</details>

<details><summary>三级：算式与伪代码</summary>

`proposals=stack([chunk[n-1-i] for i,chunk in enumerate(history)])`；`weights=exp(-decay*arange(n,dtype=proposals.dtype,device=proposals.device))`，加权后除sum。历史容量由adapter限制为K。

</details>

## 5：训练/推理组装

<details><summary>一级：想清楚什么</summary>

是否传入demonstration决定是否推断posterior；推理恒z=0。

</details>

<details><summary>二级：拆成哪些张量操作</summary>

训练按style_encoder、reparameterize、_decode、cvae_loss顺序；不要detach这些张量。

</details>

<details><summary>三级：算式与伪代码</summary>

无target返回零latent预测和None losses；训练返回`actions,loss,reconstruction_loss,kl_loss,mean,logvar`。训练步`zero_grad(); out=model(...); out["loss"].backward(); optimizer.step(); return out`。

</details>
