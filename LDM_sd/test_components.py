import unittest

import torch
from torch import nn
from PIL import Image

from attention import CrossAttention, MHSA
from clip import Clip, ClipEmbedding, ClipLayer
from decoder import VAE_AttentionBlock, VAE_Decoder, VAE_ResidualBlock
from ddpm import DDPMSampler
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
from pipeline import HEIGHT, WIDTH, generate, get_time_embedding, rescale


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
    class OnesAttention(nn.Module):
        def forward(self, x, causal_mask=False):
            return torch.ones_like(x)

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

    def test_clip_layer_uses_post_attention_residual(self):
        layer = ClipLayer(n_heads=1, dim=4)
        layer.norm1 = nn.Identity()
        layer.norm2 = nn.Identity()
        layer.attn = self.OnesAttention()

        with torch.no_grad():
            layer.ln1.weight.zero_()
            layer.ln1.bias.zero_()
            layer.ln2.weight.zero_()
            layer.ln2.bias.zero_()

        x = torch.randn(2, 3, 4)
        y = layer(x)

        self.assertTrue(torch.allclose(y, x + 1.0))


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


class TestDDPMSampler(TorchTestCase):
    def test_set_strength_truncates_timesteps(self):
        sampler = DDPMSampler(torch.Generator(device="cpu"))
        sampler.set_inference_timesteps(10)
        original = sampler.timesteps.clone()

        sampler.set_strength(0.6)

        self.assertEqual(len(sampler.timesteps), 6)
        self.assertTrue(torch.equal(sampler.timesteps, original[4:]))

    def test_add_noise_matches_closed_form(self):
        seed = 123
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sampler = DDPMSampler(generator)
        samples = torch.randn(2, 4, 8, 8)
        timestep = torch.tensor([10, 20], dtype=torch.long)

        noisy = sampler.add_noise(samples, timestep)

        expected_generator = torch.Generator(device="cpu").manual_seed(seed)
        expected_noise = torch.randn(samples.shape, generator=expected_generator)
        alphas_cumprod = sampler.alphas_cumprod
        sqrt_alpha = alphas_cumprod[timestep].sqrt().view(2, 1, 1, 1)
        sqrt_one_minus = (1 - alphas_cumprod[timestep]).sqrt().view(2, 1, 1, 1)
        expected = sqrt_alpha * samples + sqrt_one_minus * expected_noise

        self.assertTrue(torch.allclose(noisy, expected))

    def test_step_matches_closed_form_at_t0(self):
        sampler = DDPMSampler(torch.Generator(device="cpu"))
        sampler.set_inference_timesteps(1000)

        t = 0
        latents = torch.randn(1, 4, 8, 8)
        model_output = torch.randn(1, 4, 8, 8)

        actual = sampler.step(t, latents, model_output)

        alpha_prod_t = sampler.alphas_cumprod[t]
        pred_x0 = (latents - (1 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt()
        expected = pred_x0

        self.assertTrue(torch.allclose(actual, expected))


class TestPipelineHelpers(TorchTestCase):
    class DummyTokenizer:
        class Batch:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        def batch_encode_plus(self, texts, padding, max_length):
            return self.Batch([[1] * max_length for _ in texts])

    class DummyClip(nn.Module):
        def forward(self, tokens):
            batch, seq_len = tokens.shape
            return torch.zeros(batch, seq_len, 768, dtype=torch.float32, device=tokens.device)

    class DummyDiffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, latents, context, time):
            self.batch_sizes.append(latents.shape[0])
            return torch.zeros_like(latents)

    class DummyDecoder(nn.Module):
        def forward(self, latents):
            batch = latents.shape[0]
            return torch.zeros(batch, 3, HEIGHT, WIDTH, device=latents.device)

    class DummyEncoder(nn.Module):
        def forward(self, image, noise):
            return noise

    def test_rescale_maps_endpoints(self):
        x = torch.tensor([0.0, 127.5, 255.0])
        y = rescale(x, (0, 255), (-1, 1))
        expected = torch.tensor([-1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(y, expected))

    def test_time_embedding_at_zero(self):
        emb = get_time_embedding(0)
        self.assertEqual(emb.shape, (1, 320))
        self.assertTrue(torch.allclose(emb[:, :160], torch.ones(1, 160)))
        self.assertTrue(torch.allclose(emb[:, 160:], torch.zeros(1, 160)))

    def test_generate_uses_cfg_batching(self):
        diffusion = self.DummyDiffusion()
        models = {
            "clip": self.DummyClip(),
            "diffusion": diffusion,
            "decoder": self.DummyDecoder(),
            "encoder": self.DummyEncoder(),
        }

        image = generate(
            prompt="a cat",
            uncond_prompt="",
            do_cfg=True,
            models=models,
            tokenizer=self.DummyTokenizer(),
            device="cpu",
            idle_device="cpu",
            seed=0,
            n_inference_steps=1,
        )

        self.assertEqual(image.shape, (HEIGHT, WIDTH, 3))
        self.assertEqual(diffusion.batch_sizes, [2])

    def test_generate_img2img_output_shape(self):
        diffusion = self.DummyDiffusion()
        models = {
            "clip": self.DummyClip(),
            "diffusion": diffusion,
            "decoder": self.DummyDecoder(),
            "encoder": self.DummyEncoder(),
        }
        input_image = Image.new("RGB", (64, 64), color=(128, 64, 32))

        image = generate(
            prompt="a cat",
            uncond_prompt="",
            input_image=input_image,
            strength=0.5,
            do_cfg=False,
            models=models,
            tokenizer=self.DummyTokenizer(),
            device="cpu",
            idle_device="cpu",
            seed=0,
            n_inference_steps=2,
        )

        self.assertEqual(image.shape, (HEIGHT, WIDTH, 3))
        self.assertEqual(diffusion.batch_sizes, [1])
import torch

from clip import Clip
from decoder import VAE_Decoder
from diffusion import Diffusion
from encoder import VAE_Encoder


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_stats(name, model):
    total, trainable = count_params(model)
    print(f"\n{name}")
    print(f"  total params:     {total:,}")
    print(f"  trainable params: {trainable:,}")


def print_layer_stats(name, model):
    print(f"\n{name} layer-wise params")
    for param_name, param in model.named_parameters():
        print(f"  {param_name:50s} {str(list(param.shape)):20s} {param.numel():,}")



if __name__ == "__main__":
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()
    diffusion = Diffusion()
    clip = Clip()

    models = {
        "VAE_Encoder": encoder,
        "VAE_Decoder": decoder,
        "Diffusion": diffusion,
        "CLIP": clip,
    }

    grand_total = 0

    print("Model Parameter Summary")
    print("=" * 60)

    for name, model in models.items():
        print_model_stats(name, model)
        total, _ = count_params(model)
        grand_total += total

    print("\n" + "=" * 60)
    print(f"All modules total params: {grand_total:,}")
    unittest.main(verbosity=2)
