# RAE逐函数三级提示

先手算、补TODO、局部检查；按需逐层展开，避免先阅读完整参考解。

## 第1关：encoding

### `normalize_tokens`

形状：tokens=(B,S+N,D) → (B,N,D)，沿D求均值/方差，不跨B/N。 手算：[CLS,REG,[1,3]] → [-1,1]（忽略eps）。

<details><summary>一级：思路</summary>

去掉特殊token，保留空间对应的patch；逐token无affine LN。

</details>
<details><summary>二级：张量操作</summary>

先切token轴，再在最后维做layer_norm；不要对图像batch做标准化。

</details>
<details><summary>三级：伪代码</summary>

```text
patches=tokens[:,special_tokens:]; return layer_norm(patches,(D,),weight=None,bias=None)
```

</details>

## 第2关：decoder

### `decode_tokens`

形状：latent=(B,16,64) → decoder → pixels=(B,16,48) → (B,3,16,16)。 手算：4×4网格每个token对应4×4×3=48个像素值；第0 token回到左上角。

<details><summary>一级：思路</summary>

ViT decoder把表示恢复到像素；encoder和decoder的patch语义不同。

</details>
<details><summary>二级：张量操作</summary>

先decoder_blocks再decoder_out，调用edu_core unpatchify；局部像素顺序p,p,C。

</details>
<details><summary>三级：伪代码</summary>

```text
pixels=model.decoder_out(model.decoder_blocks(latent)); return unpatchify(pixels,4,3,grid=(4,4))
```

</details>

## 第3关：augmentation

### `augment_latent`

形状：latent/epsilon=(B,N,D)，amplitude_normal=(B,)，sigma=(B,1,1)。 手算：z=2,tau=.2,normal=-2,epsilon=3 → 3.2；sigma=.4可大于tau。

<details><summary>一级：思路</summary>

为decoder训练增加离散latent附近的扰动，提高对生成误差的适应。

</details>
<details><summary>二级：张量操作</summary>

缺省随机源分别randn(B)和randn_like；abs幅度再广播；eval调用者跳过增广。

</details>
<details><summary>三级：伪代码</summary>

```text
sigma=tau*abs(normal)[:,None,None]; return latent+sigma*epsilon
```

</details>

## 第4关：reconstruction

### `reconstruction_loss`

形状：两个(B,3,16,16) → 零维标量。 手算：prediction=[1,3],target=[0,1] → 1.5。

<details><summary>一级：思路</summary>

像素L1：该toy不冒充论文完整LPIPS/GAN目标。

</details>
<details><summary>二级：张量操作</summary>

先逐元素差绝对值，再全部元素mean。

</details>
<details><summary>三级：伪代码</summary>

```text
return abs(prediction-images).mean()
```

</details>

### `stage1_step`

形状：images=(B,3,16,16) → reconstruction → scalar loss。 手算：encoder=2不变；decoder参数按p←p-lr*grad更新。

<details><summary>一级：思路</summary>

第一阶段只更新decoder，encoder保持frozen/eval。

</details>
<details><summary>二级：张量操作</summary>

set_stage(1)后zero_grad，rae前向，调用ops.reconstruction_loss，再backward/step。

</details>
<details><summary>三级：伪代码</summary>

```text
course.set_stage(1); zero_grad; loss=ops.reconstruction_loss(course.rae(images),images); backward; step; return loss.detach()
```

</details>

## 第5关：flow

### `shift_time`

形状：time=(B,)，m=N*D，alpha=sqrt(m/n)，输出(B,)。 手算：m/n=4,t=.5 → 2/3；m/n=1 → .5。toy m=1024<n=4096，因此.5→1/3。

<details><summary>一级：思路</summary>

按latent总元素数调整噪声时间；不是只看通道数。

</details>
<details><summary>二级：张量操作</summary>

正维度、时间[0,1]；用alpha*t/(1+(alpha-1)*t)，端点仍0和1。

</details>
<details><summary>三级：伪代码</summary>

```text
alpha=sqrt(m/n); return alpha*time/(1+(alpha-1)*time)
```

</details>

### `flow_pair`

形状：latent/noise=(B,N,D)，time=(B,)需扩成(B,1,1)。 手算：z=2,noise=0,t=.25 → x=1.5，velocity=-2。

<details><summary>一级：思路</summary>

时间0是数据，时间1是噪声；监督是插值对时间的导数。

</details>
<details><summary>二级：张量操作</summary>

广播t；x=(1-t)z+t*noise；target=noise-z，不能反号。

</details>
<details><summary>三级：伪代码</summary>

```text
t=time[:,None,None]; return (1-t)*latent+t*noise,noise-latent
```

</details>

### `velocity_loss`

形状：prediction/target=(B,N,D) → scalar。 手算：prediction=[1,3],target=[0,1] → (1+4)/2=2.5。

<details><summary>一级：思路</summary>

对速度场回归，不是让网络直接预测干净图像。

</details>
<details><summary>二级：张量操作</summary>

target detach，平方差在全部元素取mean。

</details>
<details><summary>三级：伪代码</summary>

```text
return square(prediction-target.detach()).mean()
```

</details>

## 第6关：ddt

### `ddt_forward`

形状：noisy=(B,16,64)，base=(B,16,32)，condition/head=(B,16,96)，velocity=(B,16,64)。 手算：identity projector、t=2、两token的s=[1,3]时，condition=SiLU([3,5])≈[2.858,4.967]；先pool则变成重复SiLU(4)，丢失位置差异。

<details><summary>一级：思路</summary>

窄base供逐token条件，宽head独立读取原始噪声latent，绕过窄投影。

</details>
<details><summary>二级：张量操作</summary>

time_embed与base_embed初始化已给出；运行base；condition_projector(SiLU(t[:,None]+s))；head从noisy_embed(noisy)重新开始。

</details>
<details><summary>三级：伪代码</summary>

```text
t=time_embed(time[:,None]); s=base_embed(noisy)+pos; for block: s=block(s,SiLU(t)); cond=condition_projector(SiLU(t[:,None]+s)); x=noisy_embed(noisy); for head: x=head(x,cond); shift,scale=final_modulation(cond).chunk(2,-1); return out(final_norm(x)*(1+scale)+shift)
```

</details>

## 第7关：training

### `stage2_step`

形状：images→latent(B,16,64)→shifted time→flow pair→predicted velocity→scalar。 手算：EMA旧2、student更新后6、m=.75 → 新EMA=3。

<details><summary>一级：思路</summary>

第二阶段冻结整个RAE，训练生成器；optimizer之后更新EMA。

</details>
<details><summary>二级：张量操作</summary>

set_stage(2)，zero_grad；编码干净latent；ops.shift_time/flow_pair/velocity_loss；backward/step；edu_core update_ema。

</details>
<details><summary>三级：伪代码</summary>

```text
set_stage(2); zero_grad; z=ops.grid_to_tokens(rae.encode_grid(images)); t=ops.shift_time(t,N*D,4096); x,u=ops.flow_pair(z,t,noise); loss=ops.velocity_loss(generator(x,t),u); backward; step; update_ema(ema,generator,m); return loss.detach()
```

</details>

## 第8关：sampling

### `euler_sample`

形状：initial_noise=(B,16,64) → 相同shape latent。 手算：x=0,v=-2，偏移网格1→1/3→0，两步x: 0→4/3→2。

<details><summary>一级：思路</summary>

从噪声时间1沿负步长积分到数据时间0，不再加入随机噪声。

</details>
<details><summary>二级：张量操作</summary>

验证steps>0；no_grad中clone初始噪声，grid=ops.shift_time(linspace(1,0,steps+1),N*D,4096)，用相邻grid差作dt。

</details>
<details><summary>三级：伪代码</summary>

```text
with no_grad: grid=ops.shift_time(linspace(1,0,steps+1),N*D,4096); x=initial_noise.clone(); for i: x+=(grid[i+1]-grid[i])*model(x,grid[i].expand(B)); return x
```

</details>

### `sample_images`

形状：noise=(B,16,64) → images=(B,3,16,16)。 手算：16个4×4像素块拼成16×16图像，不把64通道latent直接当RGB。

<details><summary>一级：思路</summary>

完整生成：EMA速度场→latent→冻结decoder→图像。

</details>
<details><summary>二级：张量操作</summary>

set_stage(2)；调用ops.euler_sample(course.ema,...)后course.rae.decode，保留学生采样路由。

</details>
<details><summary>三级：伪代码</summary>

```text
with no_grad: set_stage(2); z=ops.euler_sample(course.ema,initial_noise,steps); return course.rae.decode(z)
```

</details>

## 第1关补充：`tokens_to_grid`

输入(B,N,D)，N=H*W，输出(B,D,H,W)。编号token=[0,1,2,3]按2×2应成为[[0,1],[2,3]]。

<details><summary>一级：思路</summary>

同一批patch的向量变成空间网格，不是学习投影。宽度变化最快。

</details>
<details><summary>二级：操作</summary>

1. 验证N=H*W。2. transpose(1,2)得到B,D,N。3. reshape成B,D,H,W。

</details>
<details><summary>三级：伪代码</summary>

```text
B,N,D=tokens.shape
assert N==H*W
return tokens.transpose(1,2).reshape(B,D,H,W)
```

</details>

## 第1关补充：`grid_to_tokens`

输入(B,D,H,W)，输出(B,H*W,D)。[[0,1],[2,3]]恢复为[0,1,2,3]。

<details><summary>一级：思路</summary>

反解上一函数，让每个token的D个channels重新相邻；不能直接reshape B,N,D。

</details>
<details><summary>二级：操作</summary>

1. flatten(2)合并H/W，得B,D,N。2. transpose(1,2)交换token与feature轴。

</details>
<details><summary>三级：伪代码</summary>

```text
return grid.flatten(2).transpose(1,2)
```

</details>
