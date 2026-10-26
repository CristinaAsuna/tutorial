# π0 三级提示

先做手算，再依次展开思路、操作和伪代码；不必先阅读参考答案。

## 第 1 关：observations

### `action_chunk`

给定单条 episode 的 (T,A) 动作，返回从当前 now 开始的 (H,A)。[0,1,2],now=1,H=4 -> [1,2,2,2]。

<details><summary>提示1：思路</summary>

构造 now 到 now+H 的索引。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 构造 now 到 now+H 的索引。
2. 上界截断到 episode 末帧。
3. 按索引取动作，不跨 episode。

</details>

<details><summary>提示3：伪代码</summary>

```text
index <- clamp(range(now,now+H),max=T-1)
return actions[index]
```

</details>

### `normalize_values`

物理量 (...,A) -> [-1,1]；low=0,high=10 时 0,5,10 -> -1,0,1。统计来自训练集。

<details><summary>提示1：思路</summary>

分母至少 1e-6。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 分母至少 1e-6。
2. 逐维仿射缩放。
3. 截断离群值，保留广播轴。

</details>

<details><summary>提示3：伪代码</summary>

```text
return clip(2*(values-low)/max(high-low,1e-6)-1,-1,1)
```

</details>

### `unnormalize_values`

归一化动作恢复物理单位；-1,0,1 -> 0,5,10。输入输出形状相同。

<details><summary>提示1：思路</summary>

反解仿射式。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 反解仿射式。
2. 不要对最终物理输出再次乘机器人控制频率。

</details>

<details><summary>提示3：伪代码</summary>

```text
return (values+1)/2*(high-low)+low
```

</details>

## 第 2 关：prefix_suffix

### `embed_prefix`

images=(B,M,3,H,W)，camera_valid=(B,M)，text=(B,L)，输出 (B,M*N+L,Dv) 和有效位。两相机各4个tokens + 3文字 -> 11。

<details><summary>提示1：思路</summary>

合并 B/M 调用 backbone.encode_vision(projected=True)。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 合并 B/M 调用 backbone.encode_vision(projected=True)。
2. 恢复相机次序，并展开相机有效位结合视觉mask。
3. 拼视觉与 backbone.embed_text，拼对应有效mask。

</details>

<details><summary>提示3：伪代码</summary>

```text
vision <- encode(flatten(images,B*M),projected=true)
valid <- camera_valid AND per_image_valid
return concat(visual_tokens,text_embeds),concat(visual_valid,text_valid)
```

</details>

### `time_embedding`

time=(B,)，dim 为偶数。period=.004*(4/.004)^fraction；time=0 时 sin 半边为0，cos 半边为1。

<details><summary>提示1：思路</summary>

构造 dim/2 个 0到1 的 fraction。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 构造 dim/2 个 0到1 的 fraction。
2. 计算跨 .004 到4的几何周期。
3. 按 2*pi*time/period 拼 sin 和 cos。

</details>

<details><summary>提示3：伪代码</summary>

```text
period <- .004*(1000)^linspace(0,1,D/2)
phase <- 2*pi*time/period
return concat(sin(phase),cos(phase))
```

</details>

### `embed_suffix`

state=(B,7)，noisy=(B,8,7)，输出 (B,9,De)：一个state token、八个动作token。

<details><summary>提示1：思路</summary>

使用 model.state_proj 生成单个state token。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 使用 model.state_proj 生成单个state token。
2. 动作经过 action_in，时间经 model.ops.time_embedding 并沿H扩展。
3. 拼动作/时间特征，送 action_time，前置state token。

</details>

<details><summary>提示3：伪代码</summary>

```text
state_token <- state_proj(state)[:,None]
a <- action_in(noisy); t <- expand(ops.time_embedding(time,De),H)
return concat_tokens(state_token,action_time(concat_features(a,t)))
```

</details>

## 第 3 关：block_attention

### `block_mask`

valid=(B,P+1+H)，分组 prefix=0,state=1,actions=2。P=1,H=2 时允许矩阵为 [[1,0,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]]。

<details><summary>提示1：思路</summary>

构造三组编号。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 构造三组编号。
2. key组编号 <= query组编号时允许。
3. 同时屏蔽无效query与key，True=允许。

</details>

<details><summary>提示3：伪代码</summary>

```text
group <- [0]*P+[1]+[2]*H
allow[q,k] <- group[k]<=group[q]
return allow AND query_valid AND key_valid
```

</details>

### `position_ids`

valid=[1,1,0,1] -> positions=[0,1,1,2]，padding不增加实际位置。

<details><summary>提示1：思路</summary>

valid转long做cumsum。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. valid转long做cumsum。
2. 减1，左侧padding处负位置截断到0。

</details>

<details><summary>提示3：伪代码</summary>

```text
return max(cumsum(valid)-1,0)
```

</details>

## 第 4 关：dual_stream_transformer

### `joint_attention`

prefix=(B,P,Dv)，suffix=(B,1+H,De)，Dv可以不同于De，但attention head几何相同。

<details><summary>提示1：思路</summary>

分别用两套 layer.project 得到含RoPE的Q/K和V。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 分别用两套 layer.project 得到含RoPE的Q/K和V。
2. 沿token轴拼Q/K/V，用shared projected_attention和allow。
3. 按P切分attention结果，再调用各自layer.finish。

</details>

<details><summary>提示3：伪代码</summary>

```text
pq,pk,pv <- prefix_layer.project(prefix,pos[:P])
sq,sk,sv <- expert_layer.project(suffix,pos[P:])
z <- attention(concat(pq,sq),concat(pk,sk),concat(pv,sv),allow)
return prefix_layer.finish(prefix,z[:P]),expert_layer.finish(suffix,z[P:])
```

</details>

### `joint_forward`

多层双流：prefix在VLM层，suffix在expert层。输出只取动作位 (B,H,7)，不对state算flow loss。

<details><summary>提示1：思路</summary>

连接prefix与suffix有效位，调用ops.block_mask/position_ids。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 连接prefix与suffix有效位，调用ops.block_mask/position_ids。
2. 逐层调用ops.joint_attention。
3. 去掉suffix首state位，expert_norm/action_out。

</details>

<details><summary>提示3：伪代码</summary>

```text
allow,pos <- ops masks/positions
for each pair of layers: prefix,suffix <- ops.joint_attention(...)
return action_out(expert_norm(suffix[:,1:]))
```

</details>

### `suffix_attention`

缓存 prefix K/V=(B,Hkv,P,Dh) 已含RoPE。suffix只计算自己的Q/K/V。

<details><summary>提示1：思路</summary>

对suffix调用expert_layer.project。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 对suffix调用expert_layer.project。
2. 拼prefix和suffix的K/V，Q只有suffix。
3. attention结果送expert_layer.finish，不写回prefix缓存。

</details>

<details><summary>提示3：伪代码</summary>

```text
q,k,v <- expert.project(suffix,positions)
z <- attention(q,concat(prefix_k,k),concat(prefix_v,v),allow)
return expert.finish(suffix,z)
```

</details>

### `cached_forward`

每个expert层读同层的prefix缓存，state位置为有效prefix长度，后面动作连续编号。

<details><summary>提示1：思路</summary>

构造suffix内state/actions分组mask与prefix有效mask。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 构造suffix内state/actions分组mask与prefix有效mask。
2. suffix位置加cache.valid_mask.sum。
3. 每层调用ops.suffix_attention。
4. 去掉state后投影动作速度。

</details>

<details><summary>提示3：伪代码</summary>

```text
allow <- concat_keys(prefix_valid,suffix_block_mask)
positions <- prefix_valid_count+ops.position_ids(suffix_valid)
for layer,kv: suffix <- ops.suffix_attention(layer,suffix,positions,kv,allow)
return action_out(expert_norm(suffix[:,1:]))
```

</details>

## 第 5 关：flow_matching

### `sample_time`

openpi时间约定：1为噪声，0为动作。time=.999*Beta(1.5,1)+.001，输出(B,)。

<details><summary>提示1：思路</summary>

在指定device构造Beta分布。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 在指定device构造Beta分布。
2. 采样batch_size个时间。
3. 缩放到(.001,1)，不使用环境时间t。

</details>

<details><summary>提示3：伪代码</summary>

```text
return .999*Beta(1.5,1).sample(B)+.001
```

</details>

### `flow_pair`

x_t=t*eps+(1-t)*a，target=eps-a；a=2,eps=0,t=.25 -> x=1.5,target=-2。

<details><summary>提示1：思路</summary>

time扩成(B,1,1)。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. time扩成(B,1,1)。
2. 线性混合动作与噪声。
3. 返回插值和速度目标。

</details>

<details><summary>提示3：伪代码</summary>

```text
t <- reshape(time,B,1,1)
return t*noise+(1-t)*actions,noise-actions
```

</details>

### `flow_loss`

prediction/target=(B,H,A)，MSE为所有动作元素的均值；[1,3]对[0,1] -> 2.5。

<details><summary>提示1：思路</summary>

逐元素差平方。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 逐元素差平方。
2. 全部轴求均值，不对state token或文字logits求此loss。

</details>

<details><summary>提示3：伪代码</summary>

```text
return mean(square(prediction-target))
```

</details>

## 第 6 关：training

### `training_step`

一次更新：zero_grad -> model(batch) -> backward -> step。tiny默认端到端梯度，冻结预训练模式只更新expert。

<details><summary>提示1：思路</summary>

调用model.train并清梯度。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 调用model.train并清梯度。
2. 调用model，传入显式noise/time用于可重复检查。
3. 反向传播并optimizer.step。
4. 返回detach loss。

</details>

<details><summary>提示3：伪代码</summary>

```text
model.train(); optimizer.zero_grad()
loss <- model(batch,noise,time)
backward(loss); optimizer.step()
return detach(loss)
```

</details>

## 第 7 关：cached_control

### `cache_prefix`

编码图像/文字前缀，返回PrefixEncoding：每层已旋转K和V、有效mask和positions。

<details><summary>提示1：思路</summary>

构造双向prefix allow mask和ops.position_ids。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 构造双向prefix allow mask和ops.position_ids。
2. 调用backbone.language(use_cache=True,output_logits=False)。
3. 用PrefixEncoding打包，不detach训练中的可训练缓存。

</details>

<details><summary>提示3：伪代码</summary>

```text
allow <- valid_query AND valid_key
pos <- ops.position_ids(valid)
out <- backbone.language(prefix,valid,allow,pos,use_cache=true)
return PrefixEncoding(out.key_values,valid,pos)
```

</details>

### `euler_step`

x_next=x+dt*v；x=1.5,v=-2,dt=-.25 -> 2。dt为负，因为从噪声1积分到动作0。

<details><summary>提示1：思路</summary>

按输入step_size更新。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 按输入step_size更新。
2. 不偷偷改速度符号或裁剪每一步动作。

</details>

<details><summary>提示3：伪代码</summary>

```text
return actions+step_size*velocity
```

</details>

### `sample_actions`

初始noise=(B,8,7)，每轮观测只建立一次prefix缓存，10次采样不增长缓存。

<details><summary>提示1：思路</summary>

通过ops.embed_prefix/cache_prefix编码一次，归一化state。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 通过ops.embed_prefix/cache_prefix编码一次，归一化state。
2. 每步时间1-i/steps，构造suffix，调用ops.cached_forward。
3. 用ops.euler_step以-1/steps积分。
4. 最终调用ops.unnormalize_values。

</details>

<details><summary>提示3：伪代码</summary>

```text
prefix,valid <- ops.embed_prefix(observation); cache <- ops.cache_prefix(...)
x <- copy(initial_noise); state <- ops.normalize_values(state)
for i in range(steps):
  suffix <- ops.embed_suffix(state,x,time=1-i/steps)
  x <- ops.euler_step(x,ops.cached_forward(cache,suffix),-1/steps)
return ops.unnormalize_values(x)
```

</details>

### `execution_chunk`

pi0预测从当前时刻开始，取前Ta个动作。预测[0..7],Ta=3 -> [0,1,2]，不同于DP的To-1偏移。

<details><summary>提示1：思路</summary>

沿时间轴取前steps步，保留B/A轴。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 沿时间轴取前steps步，保留B/A轴。

</details>

<details><summary>提示3：伪代码</summary>

```text
return prediction[:,:steps]
```

</details>

### `receding_control`

每执行一个chunk才重新observe；两轮为 observe,execute,observe,execute。

<details><summary>提示1：思路</summary>

逐轮observe得到新batch。先确认输出的每一轴代表什么。

</details>

<details><summary>提示2：张量操作</summary>

1. 逐轮observe得到新batch。
2. 调用model.predict_action并传该轮noise。
3. 通过ops.execution_chunk提取动作，execute后收集。

</details>

<details><summary>提示3：伪代码</summary>

```text
for noise in noises:
  batch <- observe(); predicted <- model.predict_action(batch,noise)
  chunk <- ops.execution_chunk(predicted); execute(chunk); collect(chunk)
return collected
```

</details>
