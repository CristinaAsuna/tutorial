"""
lesson4_mae_model.py
====================
【MAE 关卡 4】：完整非对称 MAE 模型与单 Patch 归一化损失 (Full MAE Model)

在这最后一关，我们将把前面写好的 patchify, random_masking 和 2D Pos Embed 汇聚起来，
搭建出完整的非对称掩码自编码器！

本文件需要你实现 3 个核心方法:
1. forward_encoder: 25% 可见 Token 的高算力编码
2. forward_decoder: 填充 mask_token、ids_restore 空间还原与轻量解码
3. forward_loss: 单 Patch 像素归一化与只在遮蔽区域计算的 Masked MSE

测试命令:
  /Users/max/codebase/.ml/.venv/bin/python lesson4_mae_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# 引入已验证的前序组件
from lesson1_patchify import patchify
from lesson2_random_masking import random_masking
from lesson3_pos_embed import get_2d_sincos_pos_embed
from reference_mae import Block


class MaskedAutoencoderViT(nn.Module):
    """
    非对称 MAE 完整模型
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,          # 编码器重型维度
        depth: int = 4,                # 本地教学用 4 层
        num_heads: int = 8,
        decoder_embed_dim: int = 384,  # 解码器轻量维度 (非对称设计!)
        decoder_depth: int = 2,        # 解码器层数更少更浅
        decoder_num_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.norm_pix_loss = norm_pix_loss

        # 1. 编码器层
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # 2. 解码器层
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        # 填充 2D 正余弦位置编码
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【输入】: x: (B, 3, H, W)
        【输出】:
          - x: (B, 1 + len_keep, D_enc)
          - mask: (B, N)
          - ids_restore: (B, N)
        """
        # =========================================================================
        # TODO 4.1: 请实现 Encoder 前向过程
        # 步骤提示:
        # 1. 卷积切片投影: x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D_enc)
        # 2. 加上位置编码 (不含 cls 位置): x = x + self.pos_embed[:, 1:, :]
        # 3. 随机掩码: x, mask, ids_restore = random_masking(x, mask_ratio)
        # 4. 准备 cls_token (加上其位置编码):
        #    cls_token = self.cls_token + self.pos_embed[:, :1, :]
        #    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # 5. 拼接 cls_token: x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + len_keep, D_enc)
        # 6. 依次过 self.blocks 和 self.norm
        # 7. 返回 x, mask, ids_restore
        # =========================================================================
        raise NotImplementedError("TODO 4.1 尚未实现！请实现 forward_encoder")

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        【输入】:
          - x: (B, 1 + len_keep, D_enc) 编码器输出的潜变量
          - ids_restore: (B, N) 原始顺序恢复索引
        【输出】:
          - pred: (B, N, p*p*3) 重构的每个 Patch 的像素预测值
        """
        # =========================================================================
        # TODO 4.2: 请实现 Decoder 前向过程
        # 步骤提示:
        # 1. 投影到解码器维度: x = self.decoder_embed(x)
        # 2. 拆离 cls_token 和可见 patch:
        #    cls_tok = x[:, :1, :]
        #    x_patches = x[:, 1:, :]  # (B, len_keep, D_dec)
        # 3. 生成 mask_tokens 补齐长度到 N:
        #    B, len_keep, D_dec = x_patches.shape
        #    N = ids_restore.shape[1]
        #    num_masked = N - len_keep
        #    mask_tokens = self.mask_token.repeat(B, num_masked, 1)
        # 4. 拼合并用 ids_restore 恢复空间拓扑:
        #    x_all = torch.cat([x_patches, mask_tokens], dim=1)  # (B, N, D_dec)
        #    x_restored = torch.gather(x_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))
        # 5. 拼回 cls_token 并加上解码器位置编码:
        #    x = torch.cat([cls_tok, x_restored], dim=1)  # (B, 1 + N, D_dec)
        #    x = x + self.decoder_pos_embed
        # 6. 依次过 self.decoder_blocks 和 self.decoder_norm
        # 7. 预测像素 (切除 cls_token): pred = self.decoder_pred(x[:, 1:, :])  # (B, N, p*p*3)
        # 8. 返回 pred
        # =========================================================================
        raise NotImplementedError("TODO 4.2 尚未实现！请实现 forward_decoder")

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        【输入】:
          - imgs: (B, 3, H, W) 原图
          - pred: (B, N, p*p*3) 预测像素
          - mask: (B, N) 0 代表可见，1 代表被遮蔽
        【输出】:
          - loss: 标量标量均方误差 (仅在被遮蔽的 patch 上计算!)
        """
        # =========================================================================
        # TODO 4.3: 请实现单 Patch 归一化与 Masked MSE Loss
        # 步骤提示:
        # 1. target = patchify(imgs, self.patch_size)  # (B, N, p*p*3)
        # 2. 若 self.norm_pix_loss 为 True:
        #    mean = target.mean(dim=-1, keepdim=True)
        #    var = target.var(dim=-1, keepdim=True)
        #    target = (target - mean) / (var + 1e-6)**0.5
        # 3. 计算每个 patch 的平方差损失: loss = (pred - target) ** 2
        # 4. 在像素通道维度求平均: loss = loss.mean(dim=-1)  # (B, N)
        # 5. 【核心】只在 mask == 1 的位置求平均:
        #    loss = (loss * mask).sum() / mask.sum()
        # 6. 返回 loss
        # =========================================================================
        raise NotImplementedError("TODO 4.3 尚未实现！请实现 forward_loss")

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ==============================================================================
# 单元测试验证
# ==============================================================================
def run_test():
    print("\n========== 开始测试 MAE 关卡 4 ==========")
    B, C, H, W = 2, 3, 224, 224
    P = 16
    N = (H // P) * (W // P)  # 196
    mask_ratio = 0.75
    len_keep = int(N * (1 - mask_ratio))

    model = MaskedAutoencoderViT(
        img_size=H,
        patch_size=P,
        embed_dim=256,
        depth=2,
        decoder_embed_dim=128,
        decoder_depth=2
    )

    imgs = torch.randn(B, C, H, W)

    # 1. 测试 Encoder
    latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio=mask_ratio)
    assert latent.shape == (B, 1 + len_keep, 256), f"Encoder 输出形状错误: {latent.shape}"
    print("✅ TODO 4.1 (forward_encoder) 通过测试！")

    # 2. 测试 Decoder
    pred = model.forward_decoder(latent, ids_restore)
    assert pred.shape == (B, N, P * P * C), f"Decoder 预测形状错误: {pred.shape}"
    print("✅ TODO 4.2 (forward_decoder) 通过测试！")

    # 3. 测试 Loss 与反向传播
    loss = model.forward_loss(imgs, pred, mask)
    assert loss.dim() == 0 and loss.item() > 0, "Loss 必须为正标量"
    loss.backward()
    assert model.patch_embed.weight.grad is not None, "卷积 Patch 映射必须接收到反向传播梯度！"
    assert model.decoder_pred.weight.grad is not None, "解码预测头必须接收到反向传播梯度！"
    print(f"✅ TODO 4.3 (forward_loss & 梯度反向传播) 通过测试！Loss = {loss.item():.4f}")
    print("🏆 恭喜通关！你已经手写出了完整的 Masked Autoencoder！\n")


if __name__ == "__main__":
    run_test()
