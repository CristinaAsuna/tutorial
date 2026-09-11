"""
reference_mae.py
================
MAE (Masked Autoencoders) 完整标准参考实现
Kaiming He et al., CVPR 2022

涵盖:
1. patchify 与 unpatchify (无损 4D 图像与 2D Patch 序列可逆转换)
2. random_masking (经典双重 argsort 洗牌、抽取与原地还原)
3. 2D Sin-Cos 正余弦位置编码
4. 非对称 Encoder (仅计算 25% token) + Decoder (填充 mask_token 还原空间拓扑)
5. Patch 级归一化 Masked MSE Loss
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Patchify 与 Unpatchify
# ==============================================================================
def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    将图像切片为 Patch 向量序列 (4D -> 3D)

    Tensor 变换推导:
      imgs: (B, 3, H, W)
      令 h = H // p, w = W // p
      reshape: (B, 3, h, p, w, p)
      permute: (B, h, w, p, p, 3) 按照空间 Patch 网格聚集像素
      reshape: (B, h*w, p*p*3)
    """
    if imgs.ndim != 4:
        raise ValueError("imgs must have shape (B, C, H, W).")
    B, C, H, W = imgs.shape
    p = patch_size
    if p <= 0 or H != W or H % p != 0:
        raise ValueError(f"image size ({H}, {W}) must be square and divisible by patch_size={p}.")
    h = w = H // p

    # (B, 3, h, p, w, p) -> (B, h, w, p, p, 3)
    x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
    # 展平为 (B, N, p*p*3)
    patches = x.reshape(B, h * w, p * p * C)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int = 16, channels: int = 3) -> torch.Tensor:
    """
    将 Patch 向量序列还原为原始 2D 图像 (3D -> 4D)

    Tensor 变换推导:
      patches: (B, N, p*p*C)
      令 h = w = int(sqrt(N))
      reshape: (B, h, w, p, p, C)
      permute: (B, C, h, p, w, p) 恢复通道在前、像素在后的维度排列
      reshape: (B, C, h*p, w*p)
    """
    if patches.ndim != 3:
        raise ValueError("patches must have shape (B, N, patch_size**2 * channels).")
    B, N, D = patches.shape
    p = patch_size
    C = channels
    if p <= 0 or C <= 0 or D != p * p * C:
        raise ValueError("patch width must equal patch_size**2 * channels.")
    h = w = int(N ** 0.5)
    if h * w != N:
        raise ValueError(f"patch count {N} must be a perfect square.")

    # (B, h, w, p, p, C) -> (B, C, h, p, w, p)
    x = patches.reshape(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4)
    imgs = x.reshape(B, C, h * p, w * p)
    return imgs


# ==============================================================================
# 2. Random Masking (Kaiming He 双重 argsort 核心算法)
# ==============================================================================
def random_masking(x: torch.Tensor, mask_ratio: float = 0.75):
    """
    执行随机掩码:
    1. 随机打乱全部 Token
    2. 只保留前 len_keep 个可见 Token
    3. 记录用于解码时原样还原位置的 ids_restore
    4. 生成二值 Mask (0 为保留, 1 为被遮蔽)

    输入:
      x: (B, N, D)
    输出:
      x_masked:    (B, len_keep, D) 仅 25% 长度！
      mask:        (B, N)           0 为保留，1 为遮蔽
      ids_restore: (B, N)           记录每个原始位置在打乱后序列中的索引
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (B, N, D).")
    if not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must satisfy 0 < mask_ratio <= 1 for masked reconstruction loss.")
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    # 1. 为每个样本的每个 patch 生成独立均匀分布随机噪声
    noise = torch.rand(B, N, device=x.device)  # (B, N)

    # 2. 第一次 argsort: 获得升序排列的索引 (小噪声在前，大噪声在后)
    ids_shuffle = torch.argsort(noise, dim=1)  # (B, N)

    # 3. 第二次 argsort: 对排好序的索引再次 argsort，即可获得逆映射 (恢复原始空间顺序的索引)
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # (B, N)

    # 4. 取出前 len_keep 个索引作为可见 Patch
    ids_keep = ids_shuffle[:, :len_keep]  # (B, len_keep)

    # 5. 利用 gather 按照 ids_keep 抽取可见 Token
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # 6. 生成二值 Mask: 0 为保留 (可见), 1 为遮蔽 (Masked)
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    # 同样用 ids_restore 恢复原始空间顺序
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


# ==============================================================================
# 3. 2D Sin-Cos 正余弦位置编码
# ==============================================================================
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """计算 1D 正余弦位置编码: pos (M,) -> (M, embed_dim)"""
    if embed_dim % 2 != 0:
        raise ValueError("1D sin-cos embed_dim must be even.")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    构造 2D 正余弦位置编码 (网格笛卡尔积)
    grid_size: int (例如 14, 对应 14x14 个 patch)
    """
    if embed_dim % 4 != 0:
        raise ValueError("2D sin-cos embed_dim must be divisible by 4.")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 2 x 14 x 14
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    # 高度维度和宽度维度各占 embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)                  # (H*W, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ==============================================================================
# 4. ViT 基础 Block
# ==============================================================================
class Block(nn.Module):
    """标准的 Pre-LN Transformer 编码器块，内建 SDPA (FlashAttention 原生加速)"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads.")
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Self-Attention with Pre-LN
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(attn_out)

        # FFN with Pre-LN
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# 5. 完整的 MaskedAutoencoderViT 架构
# ==============================================================================
class MaskedAutoencoderViT(nn.Module):
    """
    非对称 Masked Autoencoder
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        if img_size <= 0 or patch_size <= 0 or img_size % patch_size != 0:
            raise ValueError("img_size must be positive and divisible by patch_size.")
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.norm_pix_loss = norm_pix_loss

        # ---------------- 编码器 (Encoder - 重型) ----------------
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # ---------------- 解码器 (Decoder - 轻量化非对称) ----------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        """Use the fixed sin-cos positions and the initialization used by official MAE."""
        # 初始化固定的 2D 正余弦位置编码
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Conv patch projection is treated as a flattened linear layer in official MAE.
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.patch_embed.weight.shape[0], -1))
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        # 初始化 mask_token 和 cls_token
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        """
        编码器前向:
        1. 切片成 Patch 并过 Conv 投影: (B, 3, H, W) -> (B, N, D_enc)
        2. 加上位置编码 (不含 cls_token)
        3. 随机掩码抽取 25% Token: (B, len_keep, D_enc)
        4. 拼入 cls_token: (B, 1 + len_keep, D_enc)
        5. 过重型 Encoder 提取特征
        """
        if x.ndim != 4 or x.shape[1] != self.in_chans:
            raise ValueError(f"x must have shape (B, {self.in_chans}, H, W).")
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"expected image size ({self.img_size}, {self.img_size}), got {tuple(x.shape[-2:])}.")

        # (B, 3, H, W) -> (B, D_enc, h, w) -> (B, N, D_enc)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # 加上位置编码 (此时还未拼 cls_token，所以 pos_embed 取 1 之后的切片)
        x = x + self.pos_embed[:, 1:, :]

        # 随机掩码抽取
        x, mask, ids_restore = random_masking(x, mask_ratio)

        # 拼入 cls_token 并补上 cls_token 对应的位置编码
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + len_keep, D_enc)

        # 编码器计算
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        """
        解码器前向:
        1. 投影到解码器维度 D_dec
        2. 剥离 cls_token
        3. 用 mask_token 补齐被遮蔽的 75% 空缺: 变为 (B, N, D_dec)
        4. 【绝妙关键】用 ids_restore 进行 gather，按空间原始顺序全部还原回位！
        5. 拼回 cls_token，加上解码器位置编码，过轻量 Decoder，最后投影预测像素 (B, N, p*p*3)
        """
        # 1. 维度投影
        x = self.decoder_embed(x)

        # 2. 剥离 cls_token
        cls_tok = x[:, :1, :]
        x_patches = x[:, 1:, :]  # (B, len_keep, D_dec)

        # 3. 补充 mask_tokens
        B, len_keep, D_dec = x_patches.shape
        N = ids_restore.shape[1]
        num_masked = N - len_keep
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # 4. 拼合并利用 ids_restore 恢复空间排列
        x_all = torch.cat([x_patches, mask_tokens], dim=1)  # (B, N, D_dec)
        x_restored = torch.gather(x_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))

        # 5. 拼回 cls_token 并加上解码器位置编码
        x = torch.cat([cls_tok, x_restored], dim=1)  # (B, 1 + N, D_dec)
        x = x + self.decoder_pos_embed

        # 6. 过 Decoder Transformer Blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 7. 预测 Patch 像素 (移除 cls_token)
        pred = self.decoder_pred(x[:, 1:, :])  # (B, N, p*p*3)
        return pred

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """
        计算 Masked MSE Loss:
        1. 将原图 patchify 得到真实 target (B, N, p*p*3)
        2. 若启用 norm_pix_loss，对每个 patch 内的像素做均值方差归一化
        3. 计算均方误差 (pred - target) ** 2
        4. 【只在 mask == 1 (被遮蔽区域) 计算 Loss！】
        """
        target = patchify(imgs, self.patch_size)  # (B, N, p*p*3)
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {tuple(pred.shape)} must match target shape {tuple(target.shape)}.")
        if mask.shape != target.shape[:2]:
            raise ValueError("mask must have shape (B, num_patches).")

        if self.norm_pix_loss:
            # 在每个 patch 的像素通道维度上求均值和方差
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6)**0.5

        # 计算像素误差 (B, N, p*p*3) -> (B, N)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        # 只对掩码部分求平均
        masked_count = mask.sum()
        if masked_count <= 0:
            raise ValueError("masked loss requires at least one masked patch.")
        loss = (loss * mask).sum() / masked_count
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
