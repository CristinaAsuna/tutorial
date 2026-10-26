# 分层提示

先尝试手算，再逐层展开。所有关卡的参数初始化和批量工程已提供；只填写关键计算。

## 第 1 关：trajectory_windows

### `trajectory_window`

单 episode，now=0,To=2,Tp=4 时观测索引 [0,0]、动作索引 [0,0,1,2]。

<details><summary>提示 1：思路</summary>

窗口起点 now-To+1。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 窗口起点 now-To+1。
2. 观测取 To 步，动作取 Tp 步；分别 clamp 到同一 episode 有效索引。
3. 重复首尾帧，返回 obs/action；不是 ACT 的未来窗口起点。

</details>

<details><summary>提示 3：伪代码</summary>

```text
start <- now-To+1
oi <- clip(range(start,now+1),0,episode_length-1)
ai <- clip(range(start,start+Tp),0,episode_length-1)
return observations[oi],actions[ai]
```

</details>

### `fit_bounds`

训练集 (N,A) -> 每维 low/high；[[0,4],[2,8]] -> low=[0,4],high=[2,8]。

<details><summary>提示 1：思路</summary>

沿样本轴取最小/最大值。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 沿样本轴取最小/最大值。
2. 返回 low/high 字典，不能混入评估轨迹。

</details>

<details><summary>提示 3：伪代码</summary>

```text
return {low:min(data,axis=N), high:max(data,axis=N)}
```

</details>

### `normalize`

(...,A) 缩放到训练边界 [-1,1]；0,5,10 -> -1,0,1。

<details><summary>提示 1：思路</summary>

分母 clamp_min(1e-6)。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 分母 clamp_min(1e-6)。
2. 仿射缩放，不额外裁剪评估输入。

</details>

<details><summary>提示 3：伪代码</summary>

```text
span <- max(high-low,1e-6)
return 2*(data-low)/span-1
```

</details>

### `unnormalize`

归一化动作 -> 物理动作；low=0 high=10，0 -> 5。

<details><summary>提示 1：思路</summary>

反解归一化公式，保留广播轴。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 反解归一化公式，保留广播轴。

</details>

<details><summary>提示 3：伪代码</summary>

```text
return (data+1)/2*(high-low)+low
```

</details>

## 第 2 关：forward_diffusion

### `cosine_schedule`

K -> betas=(K,), alpha_bar=(K,)。f(t)=cos²((t+.008)/1.008*pi/2)；beta_i=1-f((i+1)/K)/f(i/K)。

<details><summary>提示 1：思路</summary>

float64 构造 K+1 个累计余弦点并除首值。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. float64 构造 K+1 个累计余弦点并除首值。
2. 相邻比值得 beta，最大 .999，转 float32。
3. 对 1-beta 做 cumprod 得实际 alpha_bar。

</details>

<details><summary>提示 3：伪代码</summary>

```text
u <- equally_spaced(0,1,K+1)
f <- cos((u+.008)/1.008*pi/2)^2; f <- f/f[0]
beta <- min(1-f[1:]/f[:-1],.999)
return beta,cumulative_product(1-beta)
```

</details>

### `add_noise`

a_t=sqrt(alpha_bar_t)*a0+sqrt(1-alpha_bar_t)*eps。a0=2,eps=0,alpha_bar=.25 -> 1。

<details><summary>提示 1：思路</summary>

按每个 batch 的 timesteps gather 累计系数。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 按每个 batch 的 timesteps gather 累计系数。
2. reshape (B,1,1) 广播到 (B,Tp,A)。
3. 组合 clean 与 noise。

</details>

<details><summary>提示 3：伪代码</summary>

```text
a <- reshape(alpha_bar[timesteps], B,1,1)
return sqrt(a)*clean + sqrt(1-a)*noise
```

</details>

## 第 3 关：observation_condition

### `encode_observations`

images=(B,To,3,H,W)，states=(B,To,4)，输出 (B,To,16)。例如 B=2,To=2 时 encoder 收到 4 张图。

<details><summary>提示 1：思路</summary>

合并 B/To 轴做视觉编码。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 合并 B/To 轴做视觉编码。
2. 恢复时序轴，沿特征轴拼 states；推理循环外调用。

</details>

<details><summary>提示 3：伪代码</summary>

```text
features <- encoder(reshape(images,B*To,C,H,W))
features <- reshape(features,B,To,Fvision)
return concat(features,states,axis=feature)
```

</details>

## 第 4 关：conditional_unet

### `timestep_embedding`

时间步 (B,) -> (B,D)，D 为偶数且 >=4；t=0 时 sin 半边全0、cos 半边全1。

<details><summary>提示 1：思路</summary>

构造半维指数频率，范围 1..1/10000。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 构造半维指数频率，范围 1..1/10000。
2. 时间步外积频率，拼 sin/cos。

</details>

<details><summary>提示 3：伪代码</summary>

```text
freq[j] <- exp(-log(10000)*j/(D/2-1))
angles[b,j] <- timesteps[b]*freq[j]
return concat(sin(angles),cos(angles),axis=feature)
```

</details>

### `film_modulate`

x=(B,C,T)，scale_bias=(B,2C)。scale=2,bias=3,x=1 -> 5，使用 scale*x+bias。

<details><summary>提示 1：思路</summary>

沿通道拆 scale/bias。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 沿通道拆 scale/bias。
2. 末尾加时间广播轴；不是 (1+scale)*x。

</details>

<details><summary>提示 3：伪代码</summary>

```text
scale,bias <- split(scale_bias,2,axis=channel)
return scale[:,:,None]*features+bias[:,:,None]
```

</details>

### `residual_forward`

初始化在 ConditionalResidual 中。conv1 -> FiLM -> conv2 + skip；输入 (B,Cin,T)。

<details><summary>提示 1：思路</summary>

运行 block.conv1。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 运行 block.conv1。
2. 调用 block.ops.film_modulate 和 block.condition。
3. 运行 conv2，加投影后的残差。

</details>

<details><summary>提示 3：伪代码</summary>

```text
hidden <- block.conv1(x)
hidden <- block.ops.film_modulate(hidden,block.condition(condition))
return block.conv2(hidden)+block.skip(x)
```

</details>

### `unet_down`

输入 (B,2,8) -> skip (B,16,8)、down (B,32,4)。

<details><summary>提示 1：思路</summary>

条件残差块提取 skip。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 条件残差块提取 skip。
2. downsample 缩短时间轴；返回 (down,skip)。

</details>

<details><summary>提示 3：伪代码</summary>

```text
skip <- model.down_block(x,cond)
return model.downsample(skip),skip
```

</details>

### `unet_up`

输入 (B,32,4)、skip (B,16,8) -> (B,16,8)。

<details><summary>提示 1：思路</summary>

upsample 恢复时间轴。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. upsample 恢复时间轴。
2. 沿通道拼 skip，再进条件残差块。

</details>

<details><summary>提示 3：伪代码</summary>

```text
up <- model.upsample(x)
return model.up_block(concat(up,skip,axis=channel),cond)
```

</details>

### `unet_forward`

(B,Tp,A) -> 噪声预测同形；cond=(B,48)=16维时间+2*16维观测。

<details><summary>提示 1：思路</summary>

通过 model.ops.timestep_embedding 与 time_mlp 构造时间条件，拼 obs.flatten(1)。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 通过 model.ops.timestep_embedding 与 time_mlp 构造时间条件，拼 obs.flatten(1)。
2. 转成 Conv1d 的 B,A,T；调用 ops.unet_down。
3. middle 后调用 ops.unet_up。
4. output 投影并转回 B,T,A。

</details>

<details><summary>提示 3：伪代码</summary>

```text
time <- model.time_mlp(model.ops.timestep_embedding(timesteps,16))
cond <- concat(time,flatten(obs,from_axis=1),axis=feature)
x,skip <- model.ops.unet_down(model,transpose_time_channel(sample),cond)
x <- model.middle(x,cond)
x <- model.ops.unet_up(model,x,skip,cond)
return transpose_time_channel(model.output(x))
```

</details>

## 第 5 关：noise_training

### `noise_loss`

预测 eps 与真实 eps 的 MSE；[1,3] 对 [0,1] -> (1+4)/2=2.5。

<details><summary>提示 1：思路</summary>

对每个元素平方误差，全部轴求均值；重复填充窗口同样参与 loss。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 对每个元素平方误差，全部轴求均值；重复填充窗口同样参与 loss。

</details>

<details><summary>提示 3：伪代码</summary>

```text
difference <- prediction-noise
return mean(square(difference))
```

</details>

### `training_step`

loss -> backward -> optimizer -> EMA。ema 参数 frozen；默认 decay=.95。

<details><summary>提示 1：思路</summary>

train 与 zero_grad；model.compute_loss 接收固定 noise/timesteps。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. train 与 zero_grad；model.compute_loss 接收固定 noise/timesteps。
2. backward 后 step。
3. 调用共享 update_ema(ema,model,decay)；返回 detach loss。

</details>

<details><summary>提示 3：伪代码</summary>

```text
model.train(); optimizer.zero_grad()
loss <- model.compute_loss(batch,noise,timesteps)
backward(loss); optimizer.step()
update_ema(ema,model,decay)
return detach(loss)
```

</details>

## 第 6 关：reverse_sampling

### `ddpm_step`

t 是单个整数，noise 与 sample 同形。令 a=alpha_bar[t],p=alpha_bar[t-1]，t=0 时 p=1。

<details><summary>提示 1：思路</summary>

从 eps 反推 x0=(sample-sqrt(1-a)*eps)/sqrt(a)，裁剪 [-1,1]。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 从 eps 反推 x0=(sample-sqrt(1-a)*eps)/sqrt(a)，裁剪 [-1,1]。
2. 均值 sqrt(p)*beta/(1-a)*x0 + sqrt(1-beta)*(1-p)/(1-a)*sample。
3. 方差 beta*(1-p)/(1-a)，仅 t>0 加噪。

</details>

<details><summary>提示 3：伪代码</summary>

```text
a <- alpha_bar[t]; p <- alpha_bar[t-1] if t>0 else 1
x0 <- clip((sample-sqrt(1-a)*prediction)/sqrt(a),-1,1)
mean <- sqrt(p)*beta[t]/(1-a)*x0 + sqrt(1-beta[t])*(1-p)/(1-a)*sample
if t==0: return mean
variance <- beta[t]*(1-p)/(1-a)
return mean+sqrt(max(variance,1e-20))*noise
```

</details>

### `ddim_step`

eta=0 的确定性跳步；previous_t=-1 时返回最终预测 x0。

<details><summary>提示 1：思路</summary>

还原并裁剪 x0。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 还原并裁剪 x0。
2. previous_t<0 时累计系数取1。
3. sqrt(a_prev)*x0 + sqrt(1-a_prev)*prediction。

</details>

<details><summary>提示 3：伪代码</summary>

```text
a <- alpha_bar[t]; p <- alpha_bar[previous_t] if previous_t>=0 else 1
x0 <- clip((sample-sqrt(1-a)*prediction)/sqrt(a),-1,1)
return sqrt(p)*x0+sqrt(1-p)*prediction
```

</details>

### `sample_actions`

返回归一化 (B,Tp,A)；images/states 仅在循环前编码一次。

<details><summary>提示 1：思路</summary>

用 model.ops.encode_observations 缓存条件；clone 初始噪声。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 用 model.ops.encode_observations 缓存条件；clone 初始噪声。
2. DDPM 从 K-1 到0；每步构造 (B,) 时间，denoiser 预测后调用 ops.ddpm_step。
3. DDIM 校验 ddim_steps，使用 linspace(K-1,0) 的递减网格和 ops.ddim_step。
4. 未知 sampler 报 ValueError；不能重新编码图像。

</details>

<details><summary>提示 3：伪代码</summary>

```text
obs <- model.ops.encode_observations(encoder,images,states)
x <- copy(initial_noise)
DDPM: times <- [K-1,...,0]
DDIM: times <- descending_linspace(K-1,0,ddim_steps)
for each time t:
  eps <- model.denoiser(x,repeat(t,B),obs)
  DDPM: x <- ops.ddpm_step(x,eps,t,beta,alpha_bar,step_noises[t])
  DDIM: x <- ops.ddim_step(x,eps,t,next_time_or_minus_one,alpha_bar)
return x
```

</details>

## 第 7 关：receding_control

### `execution_chunk`

预测 [a0,a1,...,a7]，To=2,Ta=3 -> [a1,a2,a3]。

<details><summary>提示 1：思路</summary>

从 To-1 起沿时间轴取 Ta 步，保留 B/A 轴。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 从 To-1 起沿时间轴取 Ta 步，保留 B/A 轴。

</details>

<details><summary>提示 3：伪代码</summary>

```text
start <- obs_horizon-1
return prediction[:,start:start+action_horizon,:]
```

</details>

### `receding_rollout`

observe()->(images,states)，execute(chunk) 更新环境及历史；每轮执行 Ta 步才重规划。

<details><summary>提示 1：思路</summary>

每轮获取最新观测。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 每轮获取最新观测。
2. model.predict_action 接收对应轮次的固定噪声。
3. execute(result[action]) 并收集 chunk，不做 ACT temporal ensemble。

</details>

<details><summary>提示 3：伪代码</summary>

```text
for i in [0,...,rounds-1]:
  images,states <- observe()
  result <- model.predict_action(images,states,initial_noises[i],step_noises[i])
  execute(result[action]); collect result[action]
return collected
```

</details>

## 第 8 关：transformer_extension

### `transformer_masks`

True=允许注意；动作因果三角。memory 列为 [diffusion_time,o0,o1]，action0 可见 [True,True,False]。

<details><summary>提示 1：思路</summary>

生成 (Tp,Tp) 下三角 allow mask。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 生成 (Tp,Tp) 下三角 allow mask。
2. cross mask[i,j]=(i>=j-1)，扩散时间列总可见。

</details>

<details><summary>提示 3：伪代码</summary>

```text
self_allow[i,j] <- (j <= i)
memory_allow[i,j] <- (j-1 <= i)
return self_allow,memory_allow
```

</details>

### `transformer_tokens`

sample=(B,8,2)->(B,8,32)，obs=(B,2,16)->memory=(B,3,32)。

<details><summary>提示 1：思路</summary>

投影动作并加 action_pos。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 投影动作并加 action_pos。
2. 时间 embedding 经 time_mlp 后加 token 轴。
3. 拼时间与 obs_in(obs)，加 memory_pos。

</details>

<details><summary>提示 3：伪代码</summary>

```text
x <- action_in(sample)+action_pos
time <- time_mlp(ops.timestep_embedding(timesteps,16))[:,None,:]
memory <- concat(time,obs_in(obs),axis=token)+memory_pos
return x,memory
```

</details>

### `transformer_forward`

两层因果 self-attention + 时间对齐 cross-attention，输出 (B,8,2)。

<details><summary>提示 1：思路</summary>

调用 ops.transformer_tokens 和 ops.transformer_masks。先写下每个轴及所需输出，不改变外围接口。

</details>

<details><summary>提示 2：张量操作</summary>

1. 调用 ops.transformer_tokens 和 ops.transformer_masks。
2. 每层 pre-LN self attention 残差，再带 cross_mask 的 cross attention 残差。
3. MLP 残差后最终 norm/output；True 表示允许，不要反转。

</details>

<details><summary>提示 3：伪代码</summary>

```text
x,memory <- ops.transformer_tokens(model,sample,timesteps,obs)
self_mask,cross_mask <- ops.transformer_masks(Tp,To,device)
for each initialized block:
  x <- x+self_attention(norm1(x),allow=self_mask)
  x <- x+cross_attention(norm2(x),memory,allow=cross_mask)
  x <- x+MLP(norm3(x))
return output(norm(x))
```

</details>
