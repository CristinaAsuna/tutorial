好。现在我们把路线接上：

你已经在学：

* 基础版 diffusion U-Net
* 更优雅的 conditional 版：`scale-shift`、`side_emb`

下一步就是：

> **把“全局条件向量 emb”升级成“文本条件 context”**
> 也就是写出 **text-to-image 版本的 denoising U-Net**

最核心的变化只有一句话：

* **time / side 这种条件**：通常是一个全局向量，注入到 ResBlock
* **text 条件**：是一串 token，通常通过 **cross-attention** 注入到 U-Net

所以 text-to-image 版不是推翻前面的 U-Net，而是：

> **保留 time embedding + ResBlock 主干，再在部分 block 里加入 cross-attention。**

---

# 一、先建立整体图

最经典的 text-to-image diffusion（比如 Stable Diffusion 风格）大致是：

$$
x_t,\ t,\ \text{text prompt}
\longrightarrow
\text{text encoder}
\longrightarrow
\text{text tokens}
\longrightarrow
\text{U-Net with cross-attention}
\longrightarrow
\epsilon_\theta(x_t,t,\text{text})
$$

这里：

* `x_t`: noisy latent/image
* `t`: timestep
* `text`: 文本 prompt
* `text encoder`: 比如 CLIP text encoder
* `context`: 文本 token 特征，shape 通常是 `[B, N, D]`

其中最关键的是：

> **ResBlock 还是处理图像 feature**
>
> **CrossAttention 负责让图像 feature 去“看”文本 token**

---

# 二、先分清两类条件

## 1. timestep / class / side

这类条件通常是一个向量：

$$
emb \in \mathbb{R}^{B\times D}
$$

所以适合：

* 加到 ResBlock
* 做 scale-shift norm

---

## 2. text prompt

这类条件通常是一串 token：

$$
context \in \mathbb{R}^{B\times N\times D}
$$

例如：

* `B`: batch
* `N`: token 数
* `D`: token 维度

它不是单个向量，所以不能简单 `emb + context`。

因此要用：

> **cross-attention**

---

# 三、text-to-image U-Net 的核心改动

如果你已经有 BetterUNet，那么 text-to-image 版只需要加三样东西：

1. `TextConditionBlock` 的概念：能接收 `context`
2. `CrossAttention`
3. 一个能同时传 `emb` 和 `context` 的 `TimestepContextSequential`

---

# 四、先从最关键的 Cross-Attention 学起

你先记住一句话：

> **Self-attention：Q,K,V 都来自图像**
>
> **Cross-attention：Q 来自图像，K/V 来自文本**

数学上：

$$
Q = W_q h,\quad K=W_k c,\quad V=W_v c
$$

其中：

* (h): 图像特征
* (c): 文本 token

这样图像每个位置都能根据文本内容更新自己。

---

# 五、先写一个最小 CrossAttention

我们先写一个最小可读版本，不追求最优性能。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
```

---

## 5.1 基础小模块

```python
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```

```python
def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device

    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
```

---

## 5.2 能接收 `emb/context` 的 block 接口

```python
class TimestepContextBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb, context=None):
        raise NotImplementedError
```

---

## 5.3 更优雅的顺序容器

```python
class TimestepContextSequential(nn.Sequential, TimestepContextBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepContextBlock):
                x = layer(x, emb, context)
            else:
                x = layer(x)
        return x
```

这和你前面学的 `EmbedSequential` 一样，只是现在多了一个 `context`。

---

# 六、先保留你已经会的 ResBlock

先写一个 text-to-image 也能用的 ResBlock。

```python
class ResBlock(TimestepContextBlock):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_ch: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.out_ch = out_ch

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32 if in_ch >= 32 else 1, in_ch),
            SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_ch, 2 * out_ch if use_scale_shift_norm else out_ch),
        )

        self.out_norm = nn.GroupNorm(32 if out_ch >= 32 else 1, out_ch)
        self.out_rest = nn.Sequential(
            SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb, context=None):
        h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = self.out_rest(self.out_norm(h + emb_out))

        return self.skip(x) + h
```

注意：

* `ResBlock` 还是只真正使用 `emb`
* `context` 先不在这里用
* 这和真实 Stable Diffusion 的思想是一致的：**文本主要通过 attention 注入**

---

# 七、写 CrossAttention

这是最核心的新增部分。

```python
class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """
        x: [B, N, Cq]
        context: [B, M, Cc]
        """
        b, n, _ = x.shape
        _, m, _ = context.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # [B, heads, N, head_dim]
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale   # [B, heads, N, M]
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(b, n, self.num_heads * self.head_dim)

        return self.to_out(out)
```

---

# 八、把 CrossAttention 包成 2D block

因为 U-Net 特征是 `[B, C, H, W]`，而 attention 更方便处理 `[B, N, C]`。

所以我们要：

1. 把 feature map flatten 成 token
2. 做 cross-attention
3. 再 reshape 回去

```python
class CrossAttentionBlock(TimestepContextBlock):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.norm = nn.GroupNorm(32 if channels >= 32 else 1, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)

        self.attn = CrossAttention(
            query_dim=channels,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, emb, context=None):
        if context is None:
            raise ValueError("context is required for CrossAttentionBlock")

        b, c, h, w = x.shape
        residual = x

        x = self.proj_in(self.norm(x))            # [B, C, H, W]
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]

        x = self.attn(x, context)                 # [B, HW, C]

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)

        return residual + x
```

这里的核心是：

* query = 图像 token
* key/value = 文本 token

---

# 九、现在我们就能写 text-conditioned block 了

一个很常见的结构是：

```text
ResBlock -> CrossAttention
```

所以可以写：

```python
class TextConditionedBlock(TimestepContextSequential):
    pass
```

其实不用额外定义类，直接这样用就行：

```python
block = TimestepContextSequential(
    ResBlock(...),
    CrossAttentionBlock(...),
)
```

---

# 十、再补两个采样模块

```python
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
```

```python
class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
```

---

# 十一、写一个最小 text-to-image U-Net

我给你写一个“足够标准但不复杂”的版本。

```python
class TextToImageUNet(nn.Module):
    def __init__(
        self,
        in_ch=4,
        base_ch=128,
        out_ch=4,
        context_dim=768,
    ):
        super().__init__()
        emb_ch = base_ch * 4
        self.base_ch = base_ch

        # timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, emb_ch),
            SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        # input
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # down
        self.down1 = TimestepContextSequential(
            ResBlock(base_ch, base_ch, emb_ch),
        )

        self.down2 = TimestepContextSequential(
            ResBlock(base_ch, base_ch * 2, emb_ch),
            CrossAttentionBlock(base_ch * 2, context_dim=context_dim),
        )
        self.downsample1 = Downsample(base_ch * 2)

        self.down3 = TimestepContextSequential(
            ResBlock(base_ch * 2, base_ch * 4, emb_ch),
            CrossAttentionBlock(base_ch * 4, context_dim=context_dim),
        )
        self.downsample2 = Downsample(base_ch * 4)

        # middle
        self.mid = TimestepContextSequential(
            ResBlock(base_ch * 4, base_ch * 4, emb_ch),
            CrossAttentionBlock(base_ch * 4, context_dim=context_dim),
            ResBlock(base_ch * 4, base_ch * 4, emb_ch),
        )

        # up
        self.upsample1 = Upsample(base_ch * 4)
        self.up1 = TimestepContextSequential(
            ResBlock(base_ch * 4 + base_ch * 4, base_ch * 2, emb_ch),
            CrossAttentionBlock(base_ch * 2, context_dim=context_dim),
        )

        self.upsample2 = Upsample(base_ch * 2)
        self.up2 = TimestepContextSequential(
            ResBlock(base_ch * 2 + base_ch * 2, base_ch, emb_ch),
            CrossAttentionBlock(base_ch, context_dim=context_dim),
        )

        self.up3 = TimestepContextSequential(
            ResBlock(base_ch + base_ch, base_ch, emb_ch),
        )

        self.out = nn.Sequential(
            nn.GroupNorm(32 if base_ch >= 32 else 1, base_ch),
            SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t, context):
        """
        x: [B, C, H, W]
        t: [B]
        context: [B, N, D]
        """
        emb = timestep_embedding(t, self.base_ch)
        emb = self.time_mlp(emb)

        x = self.in_conv(x)

        # down
        h1 = self.down1(x, emb, context)
        h2 = self.down2(h1, emb, context)
        x = self.downsample1(h2)

        h3 = self.down3(x, emb, context)
        x = self.downsample2(h3)

        # middle
        x = self.mid(x, emb, context)

        # up
        x = self.upsample1(x)
        x = torch.cat([x, h3], dim=1)
        x = self.up1(x, emb, context)

        x = self.upsample2(x)
        x = torch.cat([x, h2], dim=1)
        x = self.up2(x, emb, context)

        x = torch.cat([x, h1], dim=1)
        x = self.up3(x, emb, context)

        return self.out(x)
```

---

# 十二、这个版本和你前面学过的 BetterUNet 的关系

你可以这样看：

## 之前的 BetterUNet

输入：

```python
forward(x, t, side=None)
```

条件来源：

* `t`
* `side`

它们都被压成一个全局向量 `emb`

---

## 现在的 TextToImageUNet

输入：

```python
forward(x, t, context)
```

条件来源：

* `t -> emb`
* `text -> context`

其中：

* `emb` 进 ResBlock
* `context` 进 CrossAttention

这就是 text-to-image 版本的本质。

---

# 十三、真实 Stable Diffusion 风格还会再多什么

你现在这个版本已经抓住本质了，但真实工程里还会更完整一些：

## 1. Transformer block 更完整

真实实现常常不是单个 cross-attention block，而是：

```text
LayerNorm
Self-Attention
LayerNorm
Cross-Attention
LayerNorm
FeedForward
```

也就是一个 spatial transformer / basic transformer block。

---

## 2. 文本编码器不在 U-Net 里

真实情况一般是：

```python
text_tokens = text_encoder(prompt_ids)
eps = unet(x_t, t, context=text_tokens)
```

所以 U-Net 只接收已经编码好的 `context`。

---

## 3. latent diffusion

像 Stable Diffusion，输入不是 RGB 图，而是 VAE latent：

* `in_ch=4`
* `out_ch=4`

这就是为什么上面我默认 `in_ch=4, out_ch=4`

---

# 十四、你真正要掌握的心法

以后你看任何 text-to-image diffusion U-Net，都先问这 4 个问题：

### 1. time embedding 怎么进？

答：进 ResBlock

### 2. text embedding 怎么进？

答：进 cross-attention

### 3. attention 放哪？

答：通常放在 down / mid / up 的若干 block 中

### 4. 最终预测什么？

答：噪声、v、或别的 parameterization，但 shape 与输入 latent 相同

---

# 十五、如果你想把 `side_emb` 和 text 一起保留怎么办

完全可以。

做法是：

$$
emb = time_emb + side_emb
$$

然后 `context` 仍然单独进 cross-attention。

所以 forward 就会变成：

```python
def forward(self, x, t, context, side=None):
    emb = time_emb(t)
    if side is not None:
        emb = emb + side_emb(side)
    ...
```

这也是很常见的多条件做法。

---

# 十六、最短总结

**基础 diffusion U-Net**：

* `x_t, t -> eps`

**更优雅条件版**：

* `x_t, t, side -> eps`
* side 和 t 合成全局 `emb`

**text-to-image 版**：

* `x_t, t, text_context -> eps`
* `t -> emb` 注入 ResBlock
* `text_context -> cross-attention` 注入 U-Net

所以 text-to-image 的本质不是“把 text 直接拼进输入”，而是：

> **让图像 feature 在每一层通过 cross-attention 去读取文本 token。**

---

下一步最适合的是：我直接继续带你**把这个最小 TextToImageUNet 再升级成更像 Stable Diffusion 的版本**，也就是加上
**Self-Attention + Cross-Attention + FeedForward 的 TransformerBlock**。
