好。我们现在把 **CLIP** 从“原理”到“最小实现”完整讲一遍，并在关键地方把 **tensor shape** 自然带进去。

你可以把这次学习目标定成：

1. 搞懂 **CLIP 到底在学什么**
2. 会写一个 **最小可运行版 CLIP**
3. 看懂每一步 **tensor 怎么变**
4. 明白它为什么能给 text-to-image / Stable Diffusion 提供文本条件

---

# 一、先一句话理解 CLIP

CLIP 的核心不是“生成”，而是：

> **把图片和文本映射到同一个语义空间里，让匹配的图文更接近，不匹配的更远。**

也就是学两个编码器：

* image encoder：图片 → 向量
* text encoder：文本 → 向量

然后训练时希望：

* 正确配对的 image/text 向量相似度高
* 错误配对的 image/text 向量相似度低

---

# 二、CLIP 学的到底是什么

假设一个 batch 里有 (B) 对图文：

$$
(image_1, text_1), (image_2, text_2), \dots, (image_B, text_B)
$$

CLIP 会得到：

* 图像特征：
  $$
  I \in \mathbb{R}^{B \times D}
  $$
* 文本特征：
  $$
  T \in \mathbb{R}^{B \times D}
  $$

然后做相似度矩阵：

$$
S = I T^\top
$$

shape 是：

$$
[B, D] \times [D, B] = [B, B]
$$

其中：

* 第 (i) 行第 (j) 列表示 `image_i` 和 `text_j` 的相似度
* 对角线 (S_{ii}) 应该最高，因为那是真正匹配的图文对

所以 CLIP 本质上是在做：

> **一个 batch 内的双向对比学习**

---

# 三、CLIP 的整体结构图

最经典的 CLIP 结构：

```text
image -----------------> image encoder -----------------> image embedding ----\
                                                                               -> similarity matrix -> contrastive loss
text  -----------------> text encoder  -----------------> text embedding  ----/
```

再具体一点：

```text
image: [B, 3, H, W]
    -> CNN / ViT
    -> pooled feature
    -> linear projection
    -> [B, D]

text: [B, L]
    -> token embedding
    -> Transformer
    -> pooled feature
    -> linear projection
    -> [B, D]
```

---

# 四、先看最核心的数学目标

CLIP 训练时通常会：

1. 把 image/text feature 做 L2 normalize
2. 用一个可学习温度参数 `logit_scale`
3. 得到相似度矩阵
4. 分别做 image→text 和 text→image 的 cross-entropy

公式上可以写成：

$$
\hat I = \frac{I}{|I|}, \qquad \hat T = \frac{T}{|T|}
$$

相似度：

$$
\text{logits} = \tau \hat I \hat T^\top
$$

其中 (\tau) 是温度缩放。

然后标签是：

$$
y = [0,1,2,\dots,B-1]
$$

因为 batch 中第 (i) 张图应该匹配第 (i) 条文本。

损失：

$$
\mathcal L = \frac{1}{2}\Big(
\text{CE}(\text{logits}, y) + \text{CE}(\text{logits}^\top, y)
\Big)
$$

这就是 CLIP 的核心。

---

# 五、我们先写一个最小版 CLIP

为了讲清楚，我先不用完整 OpenAI CLIP 那么复杂，而是写一个**教学版最小实现**：

* 图像编码器：用 ViT 风格或简单 CNN
* 文本编码器：用 Transformer encoder
* 最后做对比学习

我先给你一个**结构清晰版**，然后解释 tensor 变化。

---

# 六、先写文本编码器

文本输入一般是 token ids：

$$
text \in \mathbb{R}^{[B,L]}
$$

其中：

* (B): batch size
* (L): 序列长度

比如：

$$
[B, 77]
$$

这很像 CLIP text encoder 里的固定长度文本。

---

## 6.1 Transformer block

我们先复用你前面学过的标准 Transformer block。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        qkv = self.qkv(x)  # [B, L, 3D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, L, Hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, H, L, Hd]
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
```

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 6.2 文本编码器

```python
class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 77,
        width: int = 512,
        layers: int = 6,
        heads: int = 8,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.context_length = context_length
        self.width = width

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.pos_embedding = nn.Parameter(torch.randn(1, context_length, width))

        self.transformer = nn.ModuleList([
            TransformerBlock(width, heads) for _ in range(layers)
        ])

        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Linear(width, embed_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        text_ids: [B, L]
        """
        x = self.token_embedding(text_ids)   # [B, L, width]
        x = x + self.pos_embedding[:, :x.shape[1], :]  # [B, L, width]

        for block in self.transformer:
            x = block(x)   # [B, L, width]

        x = self.ln_final(x)  # [B, L, width]

        # 最简单教学版：用最后一个 token 或平均池化
        x = x.mean(dim=1)     # [B, width]

        x = self.text_projection(x)  # [B, embed_dim]
        return x
```

---

## 6.3 文本部分 tensor 怎么变

假设：

* `B = 32`
* `L = 77`
* `width = 512`
* `embed_dim = 512`

输入：

$$
text_ids: [32, 77]
$$

### token embedding

$$
[32,77] \to [32,77,512]
$$

### 加 position embedding

$$
[32,77,512] \to [32,77,512]
$$

### 过多层 transformer

$$
[32,77,512] \to [32,77,512]
$$

### 池化（这里教学版用 mean）

$$
[32,77,512] \to [32,512]
$$

### projection

$$
[32,512] \to [32,512]
$$

最终得到：

$$
text_features \in \mathbb{R}^{[32,512]}
$$

---

# 七、再写图像编码器

CLIP 的图像编码器可以是：

* ResNet
* ViT

为了和你前面学的内容接上，我先用 **ViT 风格 image encoder**。这样最统一。

---

## 7.1 Patch embedding

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)      # [B, D, Gh, Gw]
        x = x.flatten(2)      # [B, D, N]
        x = x.transpose(1, 2) # [B, N, D]
        return x
```

---

## 7.2 图像编码器

```python
class VisionEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=width,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, width))

        self.transformer = nn.ModuleList([
            TransformerBlock(width, heads) for _ in range(layers)
        ])

        self.ln_post = nn.LayerNorm(width)
        self.image_projection = nn.Linear(width, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, 224, 224]
        """
        B = images.shape[0]

        x = self.patch_embed(images)   # [B, N, width]

        cls = self.cls_token.expand(B, -1, -1)   # [B,1,width]
        x = torch.cat([cls, x], dim=1)           # [B,N+1,width]

        x = x + self.pos_embedding[:, :x.shape[1], :]  # [B,N+1,width]

        for block in self.transformer:
            x = block(x)   # [B,N+1,width]

        x = self.ln_post(x)     # [B,N+1,width]
        x = x[:, 0]             # [B,width] 取 cls token
        x = self.image_projection(x)  # [B,embed_dim]
        return x
```

---

## 7.3 图像部分 tensor 怎么变

假设：

* 输入图像：
$$
  [32,3,224,224]
$$
* patch size = 16
* width = 768
* embed_dim = 512

### patch embedding

$$
[32,3,224,224]
\to
[32,768,14,14]
\to
[32,196,768]
$$

### 加 cls token

$$
[32,196,768]
\to
[32,197,768]
$$

### 加 position embedding

$$
[32,197,768]
\to
[32,197,768]
$$

### transformer 多层

$$
[32,197,768]
\to
[32,197,768]
$$

### 取 cls token

$$
[32,197,768]
\to
[32,768]
$$

### projection

$$
[32,768]
\to
[32,512]
$$

最终：

$$
image_features \in \mathbb{R}^{[32,512]}
$$

---

# 八、现在把 CLIP 主体拼起来

这一步最关键。

```python
class CLIP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 77,
        embed_dim: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()

        self.image_encoder = VisionEncoder(
            img_size=image_size,
            patch_size=patch_size,
            width=768,
            layers=12,
            heads=12,
            embed_dim=embed_dim,
        )

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            context_length=context_length,
            width=512,
            layers=6,
            heads=8,
            embed_dim=embed_dim,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor):
        """
        images:   [B, 3, 224, 224]
        text_ids: [B, L]
        """
        image_features = self.image_encoder(images)   # [B, D]
        text_features = self.text_encoder(text_ids)   # [B, D]

        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)  # [B, D]
        text_features = F.normalize(text_features, dim=-1)    # [B, D]

        logit_scale = self.logit_scale.exp()

        # similarity matrix
        logits_per_image = logit_scale * image_features @ text_features.t()   # [B, B]
        logits_per_text = logits_per_image.t()                                # [B, B]

        return logits_per_image, logits_per_text
```

---

# 九、CLIP forward 的 tensor 变化自然走一遍

假设 batch size 是 32，embedding dim 是 512。

输入：

* 图像：
  $$
  images: [32,3,224,224]
  $$
* 文本：
  $$
  text_ids: [32,77]
  $$

---

## 9.1 image encoder 输出

```python
image_features = self.image_encoder(images)
```

得到：

$$
[32,512]
$$

---

## 9.2 text encoder 输出

```python
text_features = self.text_encoder(text_ids)
```

得到：

$$
[32,512]
$$

---

## 9.3 normalize

```python
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)
```

shape 不变：

$$
[32,512] \to [32,512]
$$

这里只是把每个向量归一化到单位球面上，方便用余弦相似度。

---

## 9.4 相似度矩阵

```python
logits_per_image = image_features @ text_features.t()
```

这里：

* `image_features`: `[32,512]`
* `text_features.t()`: `[512,32]`

所以：

$$
[32,512] \times [512,32] = [32,32]
$$

得到：

$$
logits_per_image \in \mathbb{R}^{[32,32]}
$$

这个矩阵里：

* 第 (i) 行表示第 (i) 张图和所有文本的相似度
* 第 (j) 列表示所有图片和第 (j) 条文本的相似度

对角线应该最大。

---

# 十、CLIP 的 loss 怎么写

这是你真正要会实现的地方。

```python
def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2
```

---

## 为什么标签是 `0...B-1`

因为 batch 里第 (i) 张图，应该和第 (i) 条文本匹配。

所以：

* `image_i` 的正确类别是 `text_i`
* `text_i` 的正确类别是 `image_i`

这就是 batch 内对比学习。

---

# 十一、最小训练代码长什么样

```python
model = CLIP(vocab_size=50000)

images = torch.randn(32, 3, 224, 224)
text_ids = torch.randint(0, 50000, (32, 77))

logits_per_image, logits_per_text = model(images, text_ids)
loss = clip_loss(logits_per_image, logits_per_text)

loss.backward()
```

---

# 十二、现在说说真正的 OpenAI CLIP 和这个教学版的差别

我们这个版本是“能帮助你理解并自己写出来”的版本。
真正的 CLIP 还会更复杂一些：

## 文本侧

* 用 causal mask 的 Transformer
* 常常不是 mean pooling，而是取 `EOT` token 对应位置
* tokenizer 也更复杂

## 图像侧

* 可以用 ResNet 或 ViT
* projection 和 pooling 会有细节差异

## 训练侧

* 更大 batch
* 更大数据集
* 更稳定的数值实现

但核心思想完全一样：

> **图文双塔编码器 + 对比学习**

---

# 十三、CLIP 为什么对 text-to-image 很重要

这一步特别关键。

CLIP 的文本编码器之所以重要，是因为它学会了：

> **把自然语言 prompt 映射成一个对图像语义很有用的表示**

所以后来很多 text-to-image 模型会直接拿 CLIP text encoder 或类似文本编码器来做条件。

比如：

* 给 U-Net 的 cross-attention 提供 `context`
* 给 diffusion 提供 prompt embeddings

也就是说，你现在学的 CLIP，本质上是在学：

> **“文本如何变成图像模型能理解的条件向量/条件 token”**

---

# 十四、你可以怎么把 CLIP 和前面学的内容对应起来

## CLIP 的 image encoder

和你学过的 ViT 很像：

* patch embedding
* cls token
* transformer blocks
* 最后 pooled feature

## CLIP 的 text encoder

和 NLP Transformer encoder 很像：

* token embedding
* position embedding
* transformer blocks
* pooled feature

## Stable Diffusion 的 text conditioning

和 CLIP text encoder 的输出强相关：

* CLIP 学的是图文对齐空间
* diffusion 用 text tokens 做 cross-attention 条件

---

# 十五、最值得你背下来的 shape 主线

假设 batch size = 32。

### 图像支路

$$
[32,3,224,224]
\to
[32,196,768]
\to
[32,197,768]
\to
[32,512]
$$

### 文本支路

$$
[32,77]
\to
[32,77,512]
\to
[32,77,512]
\to
[32,512]
$$

### 对比矩阵

$$
[32,512] \times [512,32]
\to
[32,32]
$$

---

# 十六、最短总结

**CLIP 的实现本质就是两个编码器：**

* 图像编码器把图片变成向量
* 文本编码器把文本变成向量

然后把这两个向量做归一化、计算相似度矩阵，并用 batch 内对比学习让正确图文对靠近、错误图文对远离。

---

如果你愿意，下一步最适合的是我继续带你做两件事中的一个：

第一种：**把这个教学版 CLIP 升级成更像 OpenAI CLIP 的 text encoder**，也就是加入 causal mask 和 EOT token 取特征。
第二种：**把 CLIP 的 text encoder 输出接到我们前面写的 StableStyleUNet 里，串起来变成 text-to-image 条件输入**。
