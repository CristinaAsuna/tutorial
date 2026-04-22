好。我们现在把你手里的 **最小 TextToImageUNet** 再往前推一步，升级成更像 **Stable Diffusion** 的版本。

你要抓住的核心变化其实只有一句话：

> 之前我们是 `ResBlock + CrossAttentionBlock`
> 现在我们要换成更标准的
> **`ResBlock + SpatialTransformer`**
>
> 而 `SpatialTransformer` 里面包含：
>
> * Self-Attention
> * Cross-Attention
> * FeedForward

这就是你在很多 Stable Diffusion 代码里看到的那套结构。

---

# 1. 先说整体结构变成什么样

之前你可以把一个条件 block 理解成：

```text
ResBlock
↓
CrossAttention
```

现在更像 Stable Diffusion 的 block 是：

```text
ResBlock
↓
SpatialTransformer
    ├─ Self-Attention
    ├─ Cross-Attention
    └─ FeedForward
```

所以整个 U-Net 还是：

* down path
* middle block
* up path

只是某些分辨率上的 block 不再是简单 attention，而是一个更完整的 transformer 模块。

---

# 2. 为什么要这样升级

因为简单的 `CrossAttentionBlock` 只做了：

* 图像 token 看文本 token

但真实 Stable Diffusion 里希望图像 token 的建模更完整：

第一步，**Self-Attention**
让图像内部不同位置先互相通信。

第二步，**Cross-Attention**
让图像去读文本。

第三步，**FeedForward**
做非线性特征变换。

所以一个更完整的 token 更新过程是：

$$
x \leftarrow x + \text{SelfAttn}(x)



x \leftarrow x + \text{CrossAttn}(x, context)



x \leftarrow x + \text{FFN}(x)
$$

这就是 transformer block 的标准套路。

---

# 3. 我们先写最底层模块

还是从小模块往上拼。

---

## 3.1 GEGLU / FeedForward

Stable Diffusion 系里常用 GEGLU 或普通 FFN。
为了你容易理解，我先写一个清晰版，用 GEGLU。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
```

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

### GEGLU

```python
class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
```

### FeedForward

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)
```

---

# 4. 写 Attention

我们写一个既能 self-attention，也能 cross-attention 的版本。

核心逻辑：

* 如果 `context is None`，那就是 self-attention
* 如果给了 `context`，那就是 cross-attention

```python
class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int | None = None, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = num_heads * head_dim
        context_dim = context_dim if context_dim is not None else query_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        """
        x: [B, N, C]
        context: [B, M, C_ctx] or None
        """
        if context is None:
            context = x

        b, n, _ = x.shape
        _, m, _ = context.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,D]
        k = k.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,M,D]
        v = v.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,M,D]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,N,M]
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B,H,N,D]
        out = out.transpose(1, 2).contiguous().view(b, n, self.num_heads * self.head_dim)

        return self.to_out(out)
```

---

# 5. 写 TransformerBlock

这是这一步的核心。

一个 block 里面：

1. LayerNorm
2. Self-Attention
3. LayerNorm
4. Cross-Attention
5. LayerNorm
6. FeedForward

并且每一步都 residual。

```python
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 64, context_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(
            query_dim=dim,
            context_dim=None,   # self-attn
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,  # cross-attn
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, context=None):
        x = x + self.attn1(self.norm1(x), context=None)
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x
```

这就是最小但已经很像 Stable Diffusion 的 transformer block。

---

# 6. 写 SpatialTransformer

因为 U-Net feature 是 2D feature map `[B,C,H,W]`，
但 transformer 要处理 token `[B,N,C]`。

所以 `SpatialTransformer` 的工作是：

1. 先把 feature map 用 `1x1 conv` 投影
2. reshape 成 token
3. 过若干个 `BasicTransformerBlock`
4. reshape 回 feature map
5. 再用 `1x1 conv` 投影回去
6. residual 加回原输入

```python
class SpatialTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        head_dim: int,
        depth: int = 1,
        context_dim: int | None = None,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.norm = nn.GroupNorm(32 if channels >= 32 else 1, channels)
        self.proj_in = nn.Conv2d(channels, inner_dim, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                context_dim=context_dim,
            )
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(inner_dim, channels, kernel_size=1)

    def forward(self, x, context=None):
        """
        x: [B, C, H, W]
        context: [B, M, D]
        """
        b, c, h, w = x.shape
        residual = x

        x = self.proj_in(self.norm(x))          # [B, inner_dim, H, W]
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1)  # [B, HW, inner_dim]

        for block in self.transformer_blocks:
            x = block(x, context=context)

        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)

        return x + residual
```

---

# 7. 保留你已经会的 ResBlock

因为 Stable Diffusion 不是“纯 transformer U-Net”，而是：

* CNN ResBlock 负责局部视觉建模
* SpatialTransformer 负责全局和文本交互

所以 ResBlock 还是保留。

```python
class TimestepContextBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb, context=None):
        raise NotImplementedError
```

```python
class TimestepContextSequential(nn.Sequential, TimestepContextBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepContextBlock):
                x = layer(x, emb, context)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context=context)
            else:
                x = layer(x)
        return x
```

注意这里我让 `SpatialTransformer` 单独处理，因为它不直接吃 `emb`。

---

## ResBlock

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

---

# 8. 继续保留 upsample / downsample

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

# 9. 组装成更像 Stable Diffusion 的 U-Net

这份不是完整 1:1 复刻，但结构思想已经非常接近。

```python
class StableStyleUNet(nn.Module):
    def __init__(
        self,
        in_ch=4,
        base_ch=128,
        out_ch=4,
        context_dim=768,
        transformer_depth=1,
        num_heads=8,
        head_dim=64,
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

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # down path
        self.down1 = TimestepContextSequential(
            ResBlock(base_ch, base_ch, emb_ch),
        )

        self.down2 = TimestepContextSequential(
            ResBlock(base_ch, base_ch * 2, emb_ch),
            SpatialTransformer(
                channels=base_ch * 2,
                num_heads=num_heads,
                head_dim=head_dim,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
        )
        self.downsample1 = Downsample(base_ch * 2)

        self.down3 = TimestepContextSequential(
            ResBlock(base_ch * 2, base_ch * 4, emb_ch),
            SpatialTransformer(
                channels=base_ch * 4,
                num_heads=num_heads,
                head_dim=head_dim,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
        )
        self.downsample2 = Downsample(base_ch * 4)

        # middle
        self.mid = TimestepContextSequential(
            ResBlock(base_ch * 4, base_ch * 4, emb_ch),
            SpatialTransformer(
                channels=base_ch * 4,
                num_heads=num_heads,
                head_dim=head_dim,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResBlock(base_ch * 4, base_ch * 4, emb_ch),
        )

        # up path
        self.upsample1 = Upsample(base_ch * 4)
        self.up1 = TimestepContextSequential(
            ResBlock(base_ch * 4 + base_ch * 4, base_ch * 2, emb_ch),
            SpatialTransformer(
                channels=base_ch * 2,
                num_heads=num_heads,
                head_dim=head_dim,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
        )

        self.upsample2 = Upsample(base_ch * 2)
        self.up2 = TimestepContextSequential(
            ResBlock(base_ch * 2 + base_ch * 2, base_ch, emb_ch),
            SpatialTransformer(
                channels=base_ch,
                num_heads=num_heads,
                head_dim=head_dim,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
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
        x: [B, 4, H, W]      latent input
        t: [B]               timestep
        context: [B, N, D]   text tokens from text encoder
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

# 10. 这和我们之前的最小 TextToImageUNet 到底差在哪

之前版本里，你的文本注入是：

```text
ResBlock
↓
CrossAttentionBlock
```

现在变成：

```text
ResBlock
↓
SpatialTransformer
    ├─ SelfAttention
    ├─ CrossAttention
    └─ FeedForward
```

也就是说，文本注入从“一个简单 attention 层”升级成了“完整 transformer 子模块”。

这就是更像 Stable Diffusion 的地方。

---

# 11. 再解释一下各层分工

你要建立这个清晰认知：

## ResBlock 做什么

* 卷积局部建模
* 接收 timestep embedding
* 调整 feature statistics
* 负责局部纹理、局部结构

## SpatialTransformer 做什么

* 把 feature map 变成 token
* 做全局建模
* 让图像 token 互相交流
* 让图像 token 读取文本 token

## Self-Attention 的意义

图像内部先对齐自己，理解全局空间关系。

## Cross-Attention 的意义

让文本决定图像哪些位置关注哪些词。

## FeedForward 的意义

做 token 级的非线性变换，提升表达能力。

---

# 12. 这已经是 Stable Diffusion 了吗

还不是完整复刻，但已经非常接近“结构思想”。

真实 Stable Diffusion 还会有这些更细的细节：

* 更标准的 transformer 实现
* 可能多个 transformer block stacked
* memory efficient attention
* zero-init tricks
* VAE latent interface
* CLIP text encoder 的具体 context 处理
* 更复杂的通道/分辨率配置

但从“理解架构”角度，你现在这版已经够用了。

---

# 13. 你现在应该真正掌握的结构图

把它记成这个：

```text
x_t
↓
in_conv
↓
[ResBlock]
↓
[ResBlock + SpatialTransformer]
↓
downsample
↓
[ResBlock + SpatialTransformer]
↓
downsample
↓
[ResBlock + SpatialTransformer + ResBlock]
↓
upsample
↓
[concat skip + ResBlock + SpatialTransformer]
↓
upsample
↓
[concat skip + ResBlock + SpatialTransformer]
↓
[concat skip + ResBlock]
↓
out_conv
↓
eps_pred
```

其中 `SpatialTransformer` 内部又是：

```text
LayerNorm
→ Self-Attention
→ residual
→ LayerNorm
→ Cross-Attention
→ residual
→ LayerNorm
→ FeedForward
→ residual
```

---

# 14. 你下一步最该练什么

我建议你现在做两件事：

第一件事：
自己不看答案，把这三个类重新写一遍：

* `CrossAttention`
* `BasicTransformerBlock`
* `SpatialTransformer`

第二件事：
把你之前的最小 `TextToImageUNet` 里的 `CrossAttentionBlock` 全部替换成 `SpatialTransformer`。

你一旦自己能替换成功，这套结构就真正吃透了。

---

# 15. 最后一句压缩总结

**最小 TextToImageUNet** 是“卷积 U-Net + 简单 cross-attention”。
**更像 Stable Diffusion 的版本** 是“卷积 U-Net + SpatialTransformer”，而 `SpatialTransformer` 里包含 **Self-Attention + Cross-Attention + FeedForward**，因此文本条件与图像特征的交互更完整，也更接近真实的 Stable Diffusion 架构。

下一条我最建议做的是：我直接带你**把这份 `StableStyleUNet` 逐行标注成“每一层 tensor shape 怎么变”**。


好，我们就把这份 `StableStyleUNet` **按 tensor shape 一层一层标出来**。

我会先固定一个具体配置，否则 shape 不好讲清楚。

---

# 1. 先固定一个例子

我们设：

* batch size = `B`
* latent 输入 `x` 的 shape:

[
x \in \mathbb{R}^{B \times 4 \times 64 \times 64}
]

这很像 Stable Diffusion 的 latent 空间输入。

再设：

* `base_ch = 128`
* `emb_ch = 4 * base_ch = 512`
* `context_dim = 768`
* 文本 token 数 = `N = 77`

所以：

$$
context \in \mathbb{R}^{B \times 77 \times 768}
$$

时间步：

$$
t \in \mathbb{R}^{B}
$$

---

# 2. 先贴一下我们要分析的 forward 主线

```python
def forward(self, x, t, context):
    emb = timestep_embedding(t, self.base_ch)
    emb = self.time_mlp(emb)

    x = self.in_conv(x)

    h1 = self.down1(x, emb, context)
    h2 = self.down2(h1, emb, context)
    x = self.downsample1(h2)

    h3 = self.down3(x, emb, context)
    x = self.downsample2(h3)

    x = self.mid(x, emb, context)

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

下面我们逐行标。

---

# 3. timestep embedding 这两行

## 3.1 输入 t

```python
t
```

shape:

[
[B]
]

---

## 3.2 `timestep_embedding(t, self.base_ch)`

这里 `self.base_ch = 128`

所以：

```python
emb = timestep_embedding(t, 128)
```

输出：

$$
emb \in \mathbb{R}^{B \times 128}
$$

---

## 3.3 `self.time_mlp(emb)`

```python
emb = self.time_mlp(emb)
```

`time_mlp` 是：

* `Linear(128 -> 512)`
* `SiLU`
* `Linear(512 -> 512)`

所以输出：

$$
emb \in \mathbb{R}^{B \times 512}
$$

这就是后面所有 `ResBlock` 用的时间条件向量。

---

# 4. 输入卷积

## 4.1 原始输入

```python
x
```

shape:

[
[B, 4, 64, 64]
]

---

## 4.2 `self.in_conv(x)`

```python
x = self.in_conv(x)
```

`in_conv = Conv2d(4 -> 128, 3x3, padding=1)`

所以：

[
[B, 4, 64, 64]
\to
[B, 128, 64, 64]
]

---

# 5. down1

```python
h1 = self.down1(x, emb, context)
```

`down1` 只有一个 `ResBlock(base_ch, base_ch)`，也就是：

* 输入通道 128
* 输出通道 128

所以：

[
[B, 128, 64, 64]
\to
[B, 128, 64, 64]
]

因此：

$$
h1 \in \mathbb{R}^{B \times 128 \times 64 \times 64}
$$

---

# 6. down2

```python
h2 = self.down2(h1, emb, context)
```

`down2` 是：

```text
ResBlock(128 -> 256)
SpatialTransformer(channels=256)
```

---

## 6.1 先过 ResBlock

[
[B, 128, 64, 64]
\to
[B, 256, 64, 64]
]

---

## 6.2 再过 SpatialTransformer

`SpatialTransformer` 不改变外部 shape，输入输出通道和空间分辨率都不变。

所以：

[
[B, 256, 64, 64]
\to
[B, 256, 64, 64]
]

因此：

$$
h2 \in \mathbb{R}^{B \times 256 \times 64 \times 64}
$$

---

# 7. 进入 `downsample1`

```python
x = self.downsample1(h2)
```

`Downsample` 是 stride=2 的卷积，不改通道，只减半空间尺寸：

[
[B, 256, 64, 64]
\to
[B, 256, 32, 32]
]

---

# 8. down3

```python
h3 = self.down3(x, emb, context)
```

`down3` 是：

```text
ResBlock(256 -> 512)
SpatialTransformer(channels=512)
```

---

## 8.1 ResBlock

[
[B, 256, 32, 32]
\to
[B, 512, 32, 32]
]

---

## 8.2 SpatialTransformer

外部 shape 不变：

[
[B, 512, 32, 32]
\to
[B, 512, 32, 32]
]

所以：

[
h3 \in \mathbb{R}^{B \times 512 \times 32 \times 32}
]

---

# 9. `downsample2`

```python
x = self.downsample2(h3)
```

空间再减半：

[
[B, 512, 32, 32]
\to
[B, 512, 16, 16]
]

---

# 10. middle block

```python
x = self.mid(x, emb, context)
```

`mid` 是：

```text
ResBlock(512 -> 512)
SpatialTransformer(512)
ResBlock(512 -> 512)
```

所以整体 shape 不变：

[
[B, 512, 16, 16]
\to
[B, 512, 16, 16]
]

---

# 11. upsample1

```python
x = self.upsample1(x)
```

上采样把空间放大 2 倍，通道不变：

[
[B, 512, 16, 16]
\to
[B, 512, 32, 32]
]

---

# 12. 和 h3 做 skip concat

```python
x = torch.cat([x, h3], dim=1)
```

这里：

* `x`: `[B, 512, 32, 32]`
* `h3`: `[B, 512, 32, 32]`

沿通道拼接：

[
[B, 512+512, 32, 32]
====================

[B, 1024, 32, 32]
]

---

# 13. up1

```python
x = self.up1(x, emb, context)
```

`up1` 是：

```text
ResBlock(1024 -> 256)
SpatialTransformer(256)
```

所以：

## 13.1 ResBlock

[
[B, 1024, 32, 32]
\to
[B, 256, 32, 32]
]

## 13.2 SpatialTransformer

[
[B, 256, 32, 32]
\to
[B, 256, 32, 32]
]

所以：

[
x \in \mathbb{R}^{B \times 256 \times 32 \times 32}
]

---

# 14. upsample2

```python
x = self.upsample2(x)
```

空间放大：

[
[B, 256, 32, 32]
\to
[B, 256, 64, 64]
]

---

# 15. 和 h2 拼接

```python
x = torch.cat([x, h2], dim=1)
```

* `x`: `[B, 256, 64, 64]`
* `h2`: `[B, 256, 64, 64]`

拼完：

[
[B, 512, 64, 64]
]

---

# 16. up2

```python
x = self.up2(x, emb, context)
```

`up2` 是：

```text
ResBlock(512 -> 128)
SpatialTransformer(128)
```

所以：

## 16.1 ResBlock

[
[B, 512, 64, 64]
\to
[B, 128, 64, 64]
]

## 16.2 SpatialTransformer

[
[B, 128, 64, 64]
\to
[B, 128, 64, 64]
]

所以：

[
x \in \mathbb{R}^{B \times 128 \times 64 \times 64}
]

---

# 17. 和 h1 再拼接

```python
x = torch.cat([x, h1], dim=1)
```

* `x`: `[B, 128, 64, 64]`
* `h1`: `[B, 128, 64, 64]`

拼接后：

[
[B, 256, 64, 64]
]

---

# 18. up3

```python
x = self.up3(x, emb, context)
```

`up3` 只有一个 `ResBlock(256 -> 128)`

所以：

[
[B, 256, 64, 64]
\to
[B, 128, 64, 64]
]

---

继续。

你这里停在最后一层输出，我们把这部分完整补上，并且把 **输出层内部 tensor 是怎么流动的** 讲清楚。

---

# 19. 输出层

前面 `up3` 结束后，当前张量是：

[
x \in \mathbb{R}^{[B,128,64,64]}
]

代码是：

```python
return self.out(x)
```

而 `self.out` 定义为：

```python
self.out = nn.Sequential(
    nn.GroupNorm(32 if base_ch >= 32 else 1, base_ch),
    SiLU(),
    nn.Conv2d(base_ch, out_ch, 3, padding=1),
)
```

在我们这个例子里：

* `base_ch = 128`
* `out_ch = 4`

所以就是：

```python
GroupNorm(32, 128)
SiLU()
Conv2d(128 -> 4, kernel_size=3, padding=1)
```

---

## 19.1 先过 GroupNorm

```python
x = GroupNorm(x)
```

输入输出 shape **不变**：

[
[B,128,64,64]
\to
[B,128,64,64]
]

这里发生的不是 shape 变化，而是：

* 对每个样本做归一化
* 以 group 为单位规范 feature statistics
* 让最后输出前的 feature 更稳定

所以你可以把它理解成：

> shape 不变，但 feature 被“整理了一下”

---

## 19.2 再过 SiLU

```python
x = SiLU(x)
```

shape 还是不变：

[
[B,128,64,64]
\to
[B,128,64,64]
]

这里也不改变维度，只做逐元素非线性变换。

---

## 19.3 最后过 3×3 卷积

```python
x = Conv2d(128 -> 4, 3x3, padding=1)(x)
```

这里才真正把通道数从 128 压到 4：

[
[B,128,64,64]
\to
[B,4,64,64]
]

因为：

* kernel = 3
* padding = 1

所以空间尺寸 (64\times64) 保持不变，只改变通道数。

---

# 20. 最终输出是什么

所以整个输出层最终给出：

$$
\epsilon_\theta(x_t,t,\text{text}) \in \mathbb{R}^{[B,4,64,64]}
$$

也就是：

* 与输入 latent `x_t` 同 shape
* 预测的是该 latent 上的噪声（或其他 parameterization，比如 v）

---

# 21. 把最后三步连起来看

你停下来的那一段，完整写出来就是：

---

## 18. up3

```python
x = self.up3(x, emb, context)
```

只有一个 `ResBlock(256 -> 128)`，所以：

[
[B,256,64,64]
\to
[B,128,64,64]
]

---

## 19. 输出层

```python
x = self.out(x)
```

内部依次是：

### GroupNorm

[
[B,128,64,64]
\to
[B,128,64,64]
]

### SiLU

[
[B,128,64,64]
\to
[B,128,64,64]
]

### Conv2d(128 → 4)

[
[B,128,64,64]
\to
[B,4,64,64]
]

---

# 22. 所以整个 forward 最后闭环了

从一开始到最后，你的 tensor 走的是：

[
[B,4,64,64]
\to
[B,128,64,64]
\to
[B,128,64,64]
\to
[B,256,64,64]
\to
[B,256,32,32]
\to
[B,512,32,32]
\to
[B,512,16,16]
\to
[B,512,16,16]
\to
[B,512,32,32]
\to
[B,1024,32,32]
\to
[B,256,32,32]
\to
[B,256,64,64]
\to
[B,512,64,64]
\to
[B,128,64,64]
\to
[B,256,64,64]
\to
[B,128,64,64]
\to
[B,4,64,64]
]

最后回到和输入同样的 shape。

这就是 diffusion U-Net 最关键的一个性质：

> **输入 noisy latent 是什么 shape，输出预测噪声通常就是什么 shape。**

---

# 23. 最后再补一句：为什么必须输出 4 通道

因为你这里输入的是 latent：

$$
x_t \in \mathbb{R}^{[B,4,64,64]}
$$

模型学习的是：

$$
\epsilon_\theta(x_t,t,\text{text})
$$

也就是说，网络要为 latent 中的**每个位置、每个通道**预测对应噪声，所以输出必须与输入 latent 逐元素对齐。

所以：

* 输入 4 通道 latent
* 输出也必须 4 通道

如果是在像素空间做 diffusion，比如 RGB 图像，那通常就是：

[
[B,3,H,W]\to[B,3,H,W]
]

---

# 24. 你这一段可以补成最终版本

你原文最后这段，补完可以写成：

---

# 18. up3

```python
x = self.up3(x, emb, context)
```

`up3` 只有一个 `ResBlock(256 -> 128)`，所以：

[
[B,256,64,64]
\to
[B,128,64,64]
]

---

# 19. 输出层

```python
return self.out(x)
```

`out` 是：

* `GroupNorm`
* `SiLU`
* `Conv2d(128 -> 4, 3x3, padding=1)`

所以内部 tensor 变化为：

[
[B,128,64,64]
\to
[B,128,64,64]
\to
[B,128,64,64]
\to
[B,4,64,64]
]

最终输出为：

[
[B,4,64,64]
]

---

如果你愿意，我下一条可以直接把 **整个 StableStyleUNet 的 forward 全部改写成“代码 + 每一行后面跟 shape 注释”的版本**，这样你可以直接拿去当学习笔记。
