import unittest

import torch

from attention import CrossAttention, MHSA
from clip import Clip, ClipEmbedding, ClipLayer
from decoder import VAE_AttentionBlock, VAE_Decoder, VAE_ResidualBlock
from diffusion import (
    Diffusion,
    TimeEmbedding,
    UNET,
    UNET_Attn,
    UNET_Outlayer,
    UNET_Residual,
    Upsample,
)
from encoder import VAE_Encoder


class TorchTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)


class TestAttentionModules(TorchTestCase):
    def test_mhsa_output_shape(self):
        layer = MHSA(8, 64)
        x = torch.randn(2, 16, 64)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_mhsa_causal_mask_output_shape(self):
        layer = MHSA(8, 64)
        x = torch.randn(2, 16, 64)
        y = layer(x, causal_mask=True)
        self.assertEqual(y.shape, x.shape)

    def test_cross_attention_output_shape(self):
        layer = CrossAttention(num_heads=8, dim=64, dim_cross=48)
        x = torch.randn(2, 16, 64)
        text = torch.randn(2, 11, 48)
        y = layer(x, text)
        self.assertEqual(y.shape, x.shape)


class TestVAEBlocks(TorchTestCase):
    def test_attention_block_output_shape(self):
        block = VAE_AttentionBlock(64)
        x = torch.randn(2, 64, 8, 8)
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_residual_block_same_channels(self):
        block = VAE_ResidualBlock(64, 64)
        x = torch.randn(2, 64, 8, 8)
        y = block(x)
        self.assertEqual(y.shape, x.shape)

    def test_residual_block_channel_projection(self):
        block = VAE_ResidualBlock(64, 128)
        x = torch.randn(2, 64, 8, 8)
        y = block(x)
        self.assertEqual(y.shape, (2, 128, 8, 8))

    def test_encoder_output_shape(self):
        encoder = VAE_Encoder()
        x = torch.randn(2, 3, 32, 32)
        noise = torch.randn(2, 4, 4, 4)
        y = encoder(x, noise)
        self.assertEqual(y.shape, (2, 4, 4, 4))

    def test_decoder_output_shape(self):
        decoder = VAE_Decoder()
        x = torch.randn(2, 4, 4, 4)
        y = decoder(x)
        self.assertEqual(y.shape, (2, 3, 32, 32))

    def test_encoder_decoder_roundtrip_shape(self):
        encoder = VAE_Encoder()
        decoder = VAE_Decoder()
        x = torch.randn(2, 3, 32, 32)
        noise = torch.randn(2, 4, 4, 4)

        latents = encoder(x, noise)
        recon = decoder(latents)

        self.assertEqual(latents.shape, (2, 4, 4, 4))
        self.assertEqual(recon.shape, x.shape)


class TestClipModules(TorchTestCase):
    def test_clip_embedding_output_shape(self):
        embedding = ClipEmbedding(vocab_size=1000, dim=32, seq_len=8)
        tokens = torch.randint(0, 1000, (2, 8))
        y = embedding(tokens)
        self.assertEqual(y.shape, (2, 8, 32))

    def test_clip_layer_output_shape(self):
        layer = ClipLayer(n_heads=4, dim=32)
        x = torch.randn(2, 8, 32)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_clip_output_shape(self):
        model = Clip()
        tokens = torch.randint(0, 49408, (2, 77))
        y = model(tokens)
        self.assertEqual(y.shape, (2, 77, 768))


class TestDiffusionBuildingBlocks(TorchTestCase):
    def test_time_embedding_output_shape(self):
        layer = TimeEmbedding(320)
        x = torch.randn(1, 320)
        y = layer(x)
        self.assertEqual(y.shape, (1, 1280))

    def test_unet_output_layer_shape(self):
        layer = UNET_Outlayer(320, 4)
        x = torch.randn(2, 320, 8, 8)
        y = layer(x)
        self.assertEqual(y.shape, (2, 4, 8, 8))

    def test_upsample_output_shape(self):
        layer = Upsample(64)
        x = torch.randn(2, 64, 8, 8)
        y = layer(x)
        self.assertEqual(y.shape, (2, 64, 16, 16))

    def test_unet_residual_output_shape(self):
        layer = UNET_Residual(320, 640)
        x = torch.randn(2, 320, 8, 8)
        time = torch.randn(1, 1280)
        y = layer(x, time)
        self.assertEqual(y.shape, (2, 640, 8, 8))

    def test_unet_attention_output_shape(self):
        layer = UNET_Attn(n_heads=8, n_emb=40, d_context=768)
        x = torch.randn(2, 320, 8, 8)
        context = torch.randn(2, 77, 768)
        y = layer(x, context)
        self.assertEqual(y.shape, x.shape)


class TestDiffusionPipeline(TorchTestCase):
    def test_unet_output_shape(self):
        model = UNET()
        latent = torch.randn(2, 4, 8, 8)
        context = torch.randn(2, 77, 768)
        time = torch.randn(1, 1280)
        y = model(latent, context, time)
        self.assertEqual(y.shape, (2, 320, 8, 8))

    def test_diffusion_output_shape(self):
        model = Diffusion()
        latent = torch.randn(2, 4, 8, 8)
        context = torch.randn(2, 77, 768)
        time = torch.randn(1, 320)
        y = model(latent, context, time)
        self.assertEqual(y.shape, latent.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
