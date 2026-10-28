# REPA三级提示

每个helper先写输入输出shape和手算值，再逐级展开；第三层给实际伪代码，不要求先读参考解。

## 1 `clean_targets`

<details><summary>一级：两条路读什么？</summary>
两者都读同一clean RGB。codec给生成latent，teacher给监督patch；不能把noise传给teacher。
</details>
<details><summary>二级：梯度如何截断？</summary>
检查(B,3,16,16)，在torch.no_grad范围调用codec.encode与teacher；两输出都无requires_grad。不要对student投影调用detach。
</details>
<details><summary>三级：伪代码</summary>

```text
assert input shape contract, otherwise ValueError
with no_grad:
    clean <- system.codec.encode(images)
    targets <- system.teacher(images)
return clean, targets
```
</details>

## 2 `flow_pair`

<details><summary>一级：先看两端点</summary>
t=0保留clean，t=1得到noise；对t求导得到noise-clean。
</details>
<details><summary>二级：广播</summary>
time[:,None,None,None]对应(B,1,1,1)；检查batch与时间范围。z=2,eps=0,t=.25应得1.5和-2。
</details>
<details><summary>三级：伪代码</summary>

```text
validate matching clean/noise and time(B,) in [0,1]
t <- time[:,None,None,None]
return (1-t)*clean+t*noise, noise-clean
```
</details>

## 3a `conditioned_block`

<details><summary>一级：gate控制哪条残差？</summary>
两个gate分别乘attention/MLP分支，原始tokens不会被gate清零。zero-init时block是恒等。
</details>
<details><summary>二级：形状</summary>
modulation(condition)在最后轴chunk(6)，每块(B,D)，加[:,None]后调制(B,N,D)。次序shift1,scale1,gate1,shift2,scale2,gate2。
</details>
<details><summary>三级：伪代码</summary>

```text
s1,a1,g1,s2,a2,g2 <- modulation(c).chunk(6,last_axis)
x <- x + g1[:,None]*attention(norm1(x)*(1+a1[:,None])+s1[:,None])
return x + g2[:,None]*mlp(norm2(x)*(1+a2[:,None])+s2[:,None])
```
</details>

## 3b `denoise`

<details><summary>一级：hidden取在哪里？</summary>
在第alignment_depth个block之后投影，后续block仍继续预测速度。project=False使投影完全不执行。
</details>
<details><summary>二级：轴变换</summary>
patchify(latent,2)得到(B,16,16)，Linear到32维；time MLP与类别embedding相加。最终每token的16个数重组为4通道2×2局部patch。
</details>
<details><summary>三级：伪代码</summary>

```text
x <- patch(patchify(latent,2)) + positions
c <- time_embed(time[:,None]) + label_embed(labels)
projected <- None
for i,block in enumerate(blocks,start=1):
    x <- model.ops.conditioned_block(block,x,c)
    if project and i==alignment_depth: projected <- projector(x)
shift,scale <- final_modulation(c).chunk(2,last_axis)
pixels <- output(final_norm(x)*(1+scale[:,None])+shift[:,None])
return unpatchify(pixels,2,channels=4,grid=(4,4)),projected
```
</details>

## 4a `alignment_loss`

<details><summary>一级：为什么有负号？</summary>
优化器最小化loss，因此要最大化相似度就最小化负cos。同向为-1，反向为+1。
</details>
<details><summary>二级：保留patch配对</summary>
只在D轴normalize与sum，不先在N轴pool。targets.detach后归一化，最后对B,N均值。
</details>
<details><summary>三级：伪代码</summary>

```text
require equal 3D shapes
a <- normalize(projected,dim=-1)
b <- normalize(targets.detach(),dim=-1)
return -sum(a*b,dim=-1).mean()
```
</details>

## 4b `joint_objective`

<details><summary>一级：两个监督有什么区别？</summary>
velocity在latent所有元素上MSE，alignment是已计算的有符号标量；weight=0退化为纯velocity。
</details>
<details><summary>二级：返回合同</summary>
返回三个标量loss、velocity_loss、alignment_loss；此处不要再次调用teacher或重复normalize。
</details>
<details><summary>三级：伪代码</summary>

```text
mse <- mean(square(prediction-velocity))
return dict(loss=mse+weight*alignment, velocity_loss=mse, alignment_loss=alignment)
```
</details>

## 5a `ema_update`

<details><summary>一级：更新谁？</summary>
EMA是denoiser的滑动平均，不是用于clean目标的external teacher。
</details>
<details><summary>二级：公式</summary>
ema=m*ema+(1-m)*student，必须no_grad。共享update_ema已完成参数遍历和momentum校验。
</details>
<details><summary>三级：伪代码</summary>

```text
update_ema(ema, model, momentum)
```
</details>

## 5b `training_step`

<details><summary>一级：先后顺序</summary>
EMA必须看optimizer更新后的student；forward计算图不能被no_grad包住。
</details>
<details><summary>二级：训练态和返回</summary>
system.train已自动维持teacher/codec/EMA的eval。zero_grad后调用system，backward总loss并step，最后返回detach日志。
</details>
<details><summary>三级：伪代码</summary>

```text
system.train(); optimizer.zero_grad(set_to_none=True)
result <- system(images,labels,noise=noise,time=time)
result['loss'].backward(); optimizer.step()
system.ops.ema_update(system.ema_denoiser,system.denoiser,system.config.ema_momentum)
return {key: value.detach() for key,value in result.items()}
```
</details>

## 6a `euler_sample`

<details><summary>一级：时间走哪个方向？</summary>
初始noise对应1，干净latent对应0；负步长配合noise-clean速度，不能同时反转两者。
</details>
<details><summary>二级：不读取干净图像</summary>
clone noise，steps必须正整数；每步构造(B,)时间，调用model(project=False)，只用返回的velocity。
</details>
<details><summary>三级：伪代码</summary>

```text
validate steps > 0 and integer
x <- initial_noise.clone()
for i in range(steps):
    t <- x.new_full((B,),1-i/steps)
    v,_ <- model(x,t,labels,project=False)
    x <- x-v/steps
return x
```
</details>

## 6b `decode_latents`

<details><summary>一级：输出在哪个空间？</summary>
Euler输出(B,4,8,8)的codec latent，还不是RGB；decoder映射回(B,3,16,16)。
</details>
<details><summary>二级：保持接口简单</summary>
调用codec.decode，不再encode，不使用teacher，不把显示裁剪混进模型输出。
</details>
<details><summary>三级：伪代码</summary>

```text
return codec.decode(latents)
```
</details>
