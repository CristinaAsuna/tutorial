# 分层提示

先尝试手算，再逐层展开。所有关卡的参数初始化和批量工程已提供；只填写关键计算。

## 第 1 关：action_normalization

### `fit_action_stats`

只从训练动作拟合统计量。actions=(N,A)，mask=(A,)；N=101 等距数 0..100 的 q01=1、q99=99。

<details><summary>提示 1：思路</summary>

沿样本轴分别计算 .01 和 .99 分位数。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 沿样本轴分别计算 .01 和 .99 分位数。
2. 返回 q01/q99/mask 字典；mask 转 bool。

</details>

<details><summary>提示 3：伪代码</summary>

```text
low <- quantile(actions, 0.01, axis=N)
high <- quantile(actions, 0.99, axis=N)
return {q01:low, q99:high, mask:boolean(mask)}
```

</details>

### `normalize_actions`

物理动作转 [-1,1]。输入 (...,A)，输出同形；low=0 high=10 时 0,5,10 -> -1,0,1。

<details><summary>提示 1：思路</summary>

逐维缩放；分母至少 1e-6，常量维不产生 NaN。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 逐维缩放；分母至少 1e-6，常量维不产生 NaN。
2. 裁剪缩放值，mask=False 的维度保留原值。

</details>

<details><summary>提示 3：伪代码</summary>

```text
span <- max(q99-q01, 1e-6)
scaled <- clip(2*(actions-q01)/span-1, -1,1)
return where(mask, scaled, actions)
```

</details>

### `unnormalize_actions`

归一化动作恢复物理单位；-1,0,1 -> 0,5,10。

<details><summary>提示 1：思路</summary>

反解仿射变换。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 反解仿射变换。
2. 仅恢复 mask=True 的维度；夹爪维度保持原值。

</details>

<details><summary>提示 3：伪代码</summary>

```text
physical <- (actions+1)/2*(q99-q01)+q01
return where(mask, physical, actions)
```

</details>

## 第 2 关：action_tokenizer

### `encode_actions`

动作 (...,A) -> long IDs (...,A)。小例 n_bins=4,V=20：[-1,0,1] -> [19,18,16]。

<details><summary>提示 1：思路</summary>

在 float64 中构造官方边界并裁剪动作。float32 边界会改变靠近内部边界的动作 ID，先分清“数值相近”和“落在同一区间”。

</details>

<details><summary>提示 2：张量操作</summary>

1. 用 float64 `arange*(2/(n_bins-1))-1` 构造 n_bins 个边界，动作裁剪后转 float64 比较。
2. 用等价于 numpy.digitize(right=False) 的 bucketize，注意 torch 的 right 参数。
3. 返回 V-index，范围为 [V-n_bins,V-1]。

</details>

<details><summary>提示 3：伪代码</summary>

```text
edges <- float64_range(n_bins)*(2/(n_bins-1))-1
index <- count(edges <= clipped_action)
return vocab_size-index
```

</details>

### `decode_actions`

token IDs -> bin centers。四边界 [-1,-1/3,1/3,1] 的中心为 [-2/3,0,2/3]。

<details><summary>提示 1：思路</summary>

先计算相邻边界的中点。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 先计算相邻边界的中点。
2. 将 V-id-1 裁剪至 [0,n_bins-2] 再索引；最后两个边界索引可共享中心。

</details>

<details><summary>提示 3：伪代码</summary>

```text
centers <- (edges[1:]+edges[:-1])/2
index <- clip(vocab_size-ids-1, 0,n_bins-2)
return centers[index]
```

</details>

## 第 3 关：fused_vision

### `fuse_features`

两路 (B,N,D1)/(B,N,D2) 沿特征轴拼接。[[1,2]] + [[3]] -> [[1,2,3]]。

<details><summary>提示 1：思路</summary>

检查 N 是相同网格，不是两个不同序列。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 检查 N 是相同网格，不是两个不同序列。
2. 沿最后一轴拼接，再调用已初始化的 projector。

</details>

<details><summary>提示 3：伪代码</summary>

```text
joined <- concatenate(first,second,axis=feature)
return projector(joined)
```

</details>

## 第 4 关：multimodal_sequence

### `pack_single`

单行插入视觉 token：文本 [BOS,7,8]、视觉 [v0,v1] -> [BOS,v0,v1,7,8]；视觉 labels=-100。

<details><summary>提示 1：思路</summary>

拼接文本首 token、视觉 embedding、剩余文本 embedding。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 拼接文本首 token、视觉 embedding、剩余文本 embedding。
2. 相同位置插入 ignore labels，返回 (embeds, labels)；不改变 labels 的监督值。

</details>

<details><summary>提示 3：伪代码</summary>

```text
embeds <- concat(text[:1],visual,text[1:])
targets <- concat(labels[:1],repeat(IGNORE,N),labels[1:])
return embeds,targets
```

</details>

### `action_labels`

ids=BOS+指令+A 个动作+EOS，只监督动作和可选 EOS。例如 [1,8,64,65,2],A=2 -> [-100,-100,64,65,2]。

<details><summary>提示 1：思路</summary>

clone ids，不原地更改输入。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. clone ids，不原地更改输入。
2. 屏蔽最后 A+1 个位置之前的所有标签。
3. predict_stop=False 时屏蔽最后 EOS。

</details>

<details><summary>提示 3：伪代码</summary>

```text
targets <- copy(ids)
targets[before last A+1 positions] <- IGNORE
if not predict_stop: targets[last] <- IGNORE
return targets
```

</details>

## 第 5 关：training

### `shifted_loss`

logits=(B,L,V)、labels=(B,L)；位置 i 的预测对齐位置 i+1 的标签，不再做第二次 shift。

<details><summary>提示 1：思路</summary>

取 logits 去末位、labels 去首位。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 取 logits 去末位、labels 去首位。
2. 展平 B 与 L，交叉熵 ignore_index=-100。

</details>

<details><summary>提示 3：伪代码</summary>

```text
pred <- flatten_batch_time(logits[:,all_except_last,:])
target <- flatten(labels[:,all_except_first])
return cross_entropy(pred,target,ignore=-100)
```

</details>

### `training_step`

一次完整训练更新。model(images,prompts,actions,stats) 返回标量 loss。

<details><summary>提示 1：思路</summary>

进入 train，清零梯度。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 进入 train，清零梯度。
2. 调用 model 并反向传播；优化器 step。
3. 返回 detach loss，不能重新计算参考 loss。

</details>

<details><summary>提示 3：伪代码</summary>

```text
model.train(); optimizer.zero_grad()
loss <- model(images,prompts,actions,stats)
backward(loss); optimizer.step()
return detach(loss)
```

</details>

## 第 6 关：autoregressive_control

### `generate_action_ids`

从不同长度 prompts 逐个生成 A 维动作。B=2，prompt 长度 2/3；第一次分别读各自最后有效 token 的 logits。

<details><summary>提示 1：思路</summary>

初始化行副本；视觉只编码一次。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 初始化行副本；视觉只编码一次。
2. 循环 A 次，调用 model.pack 与 model.language。
3. 按有效长度取得最后 logits，全词表 argmax；追加行尾。
4. 按动作维堆叠生成结果；不额外强制 action-only 词表。

</details>

<details><summary>提示 3：伪代码</summary>

```text
rows <- copies(prompts); visual <- model.encode_images(images)
repeat A times:
  embeds,mask,_ <- model.pack(rows,visual)
  logits <- model.language(embeds,mask)
  next[b] <- argmax(logits[b,sum(mask[b])-1,:])
  rows[b] <- append(rows[b],next[b]); collect next
return stack(collected,axis=action)
```

</details>

### `predict_action`

ID (B,A) -> 归一化动作 -> 物理动作。非法文本 ID 按官方 decoder 的裁剪规则处理。

<details><summary>提示 1：思路</summary>

通过 model.ops 调用生成函数。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 通过 model.ops 调用生成函数。
2. 通过 model.ops 解码并按 stats 反归一化；不得直接调用参考函数。

</details>

<details><summary>提示 3：伪代码</summary>

```text
ids <- model.ops.generate_action_ids(model,images,prompts)
u <- model.ops.decode_actions(ids,model.vocab_size)
return model.ops.unnormalize_actions(u,stats)
```

</details>
