"""
run_mae_demo.py
===============
MAE 完整前向与视觉重建全流程演练脚本！

演示内容:
1. 构造具有鲜明几何图案的测试图像
2. 演示 75% 掩码遮蔽 (196 个 patch 只保留 49 个)
3. 演示编码器非对称省显存计算
4. 演示解码器原地复原并重构像素
5. 验证单 Patch 像素归一化后的 Masked MSE Loss

运行命令:
  /Users/max/codebase/.ml/.venv/bin/python run_mae_demo.py
"""

import torch
from edu_core.training import seed_everything
from reference_mae import MaskedAutoencoderViT, patchify, random_masking, unpatchify


def main():
    print("\n" + "=" * 70)
    print("  MAE (Masked Autoencoders) 端到端实战演练")
    print("=" * 70)

    seed_everything(42)
    device = "cpu"
    B, C, H, W = 2, 3, 224, 224
    P = 16
    mask_ratio = 0.75

    # 1. 实例化 MAE 模型
    print("\n[1] 初始化非对称 ViT 架构...")
    model = MaskedAutoencoderViT(
        img_size=H,
        patch_size=P,
        embed_dim=768,          # 重型 Encoder
        depth=6,
        num_heads=12,
        decoder_embed_dim=384,  # 轻型 Decoder
        decoder_depth=4,
        decoder_num_heads=6,
        norm_pix_loss=True
    ).to(device)
    model.eval()

    # 计算参数量对比
    enc_params = sum(p.numel() for p in model.blocks.parameters()) + sum(p.numel() for p in model.patch_embed.parameters())
    dec_params = sum(p.numel() for p in model.decoder_blocks.parameters()) + sum(p.numel() for p in model.decoder_pred.parameters())
    print(f"  Encoder 参数量: {enc_params / 1e6:.2f} M")
    print(f"  Decoder 参数量: {dec_params / 1e6:.2f} M (仅占 Encoder 的 {dec_params / enc_params:.1%})")

    # 2. 构造测试图像 (带红蓝渐变)
    print("\n[2] 准备输入图像...")
    imgs = torch.zeros(B, C, H, W, device=device)
    # 给第一通道加水平渐变，第二通道加垂直渐变
    imgs[:, 0, :, :] = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
    imgs[:, 1, :, :] = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
    print(f"  输入图像形状: {imgs.shape}")

    # 3. 前向计算
    print(f"\n[3] 执行 75% 掩码前向计算 (mask_ratio={mask_ratio})...")
    with torch.no_grad():
        loss, pred, mask = model(imgs, mask_ratio=mask_ratio)

    total_patches = model.num_patches
    visible_patches = int(total_patches * (1 - mask_ratio))
    masked_patches = total_patches - visible_patches

    print(f"  Total Patches:   {total_patches} (14x14)")
    print(f"  Visible Patches: {visible_patches} (仅 {visible_patches/total_patches:.0%} 参与 Encoder 重型计算！)")
    print(f"  Masked Patches:  {masked_patches} (75% 被遮蔽并由 Decoder 还原重构)")
    print(f"  Masked MSE Loss: {loss.item():.4f}")

    # 4. 可视化重构逻辑演示
    print("\n[4] 模拟图像掩码与预测还原...")
    # 真实 target patch
    target_patches = patchify(imgs, patch_size=P)

    # 模拟被遮蔽的图像 (将 mask == 1 的 patch 设为灰色 0.5)
    masked_target = target_patches.clone()
    masked_target[mask.bool()] = 0.5
    masked_img = unpatchify(masked_target, patch_size=P)

    # 模拟重构图像: 未遮蔽部分保留原图，遮蔽部分用 pred 填充
    reconstructed_patches = target_patches.clone()
    # 如果使用了 norm_pix_loss，需要将 pred 逆归一化回原始像素值
    mean = target_patches.mean(dim=-1, keepdim=True)
    var = target_patches.var(dim=-1, keepdim=True)
    unnorm_pred = pred * (var + 1e-6)**0.5 + mean
    reconstructed_patches[mask.bool()] = unnorm_pred[mask.bool()]
    reconstructed_img = unpatchify(reconstructed_patches, patch_size=P)

    print(f"  被遮蔽图像形状: {masked_img.shape} (保留 25% 真实像素，其余填充为灰色)")
    print(f"  重构图像形状:   {reconstructed_img.shape} (75% 遮蔽部分由 MAE 补全)")

    # 5. 语义级断言：验证算法不只是“能跑且 shape 对”。
    print("\n[5] 运行参考实现语义断言...")
    round_trip = unpatchify(patchify(imgs, patch_size=P), patch_size=P)
    assert torch.equal(round_trip, imgs), "patchify/unpatchify 必须无损往返"

    tokens = torch.randn(2, 16, 8)
    _, test_mask, ids_restore = random_masking(tokens, mask_ratio=0.75)
    assert torch.equal(torch.sort(ids_restore, dim=1).values, torch.arange(16).expand(2, -1))
    assert torch.equal(test_mask.sum(dim=1), torch.full((2,), 12.0))

    # 使用极小模型验证反向传播与 masked-only loss 语义，避免 demo 过慢。
    tiny = MaskedAutoencoderViT(
        img_size=32, patch_size=8, embed_dim=64, depth=1, num_heads=4,
        decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4,
    )
    tiny_imgs = torch.randn(2, 3, 32, 32)
    tiny_loss, tiny_pred, tiny_mask = tiny(tiny_imgs, mask_ratio=0.75)
    tiny_loss.backward()
    assert torch.isfinite(tiny_loss) and tiny.patch_embed.weight.grad is not None
    assert not hasattr(tiny, "decoder_decoder"), "forward 不应向模型写入临时属性"

    target = patchify(tiny_imgs, patch_size=8)
    pred_a = target.clone()
    pred_b = target.clone()
    pred_b[tiny_mask == 0] = 123.0  # 可见 patch 的预测不应影响 masked loss
    assert torch.allclose(tiny.forward_loss(tiny_imgs, pred_a, tiny_mask), tiny.forward_loss(tiny_imgs, pred_b, tiny_mask))
    try:
        random_masking(tokens, mask_ratio=0.0)
        raise AssertionError("mask_ratio=0 必须被拒绝")
    except ValueError:
        pass
    print("  ✅ 往返、置换、masked-only loss、梯度与边界检查均通过")
    print("\n" + "=" * 70)
    print("  🎉 MAE 端到端数据流与算法机制演示圆满完成！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
