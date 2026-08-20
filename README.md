# Generative AI — Learning & From-Scratch Implementations

A personal learning repository for generative AI: understanding the math and rebuilding
generative models from scratch in PyTorch, alongside course labs, tutorials, and experiments.

## Generative Models — Progress

Implemented from scratch in [`from_scratch/`](from_scratch/):

- [x] **Autoencoder (AE)** — `from_scratch/vae/`
- [x] **Variational Autoencoder (KL-VAE)** — `from_scratch/vae/`
- [x] **Vector-Quantized VAE (VQ-VAE)** — `from_scratch/vqvae/`
- [x] **VQGAN** — `from_scratch/vqgan/`
- [x] **Flow Matching** — `from_scratch/flow_matching/`
- [x] **Score Matching** — `from_scratch/score_matching/`
- [ ] **DDPM** (Denoising Diffusion Probabilistic Models) — `from_scratch/diffusion/` — next up
- [ ] **VAR** (Visual Autoregressive) — planned

## Repository Layout

| Directory | Description |
| --- | --- |
| [`from_scratch/`](from_scratch/) | Main repo: generative models implemented from scratch (VAE family, flow/score matching, diffusion). |
| [`DL-Demos/`](DL-Demos/) | Cloned [DL-Demos](https://github.com/SingleZombie/DL-Demos) — deep-learning demos (CNN, RNN, Transformer, ddpm, ddim, pixelcnn, …). |
| [`cs336/`](cs336/) | Stanford CS336 *Language Modeling from Scratch* — assignments. |
| [`diffusers_test/`](diffusers_test/) | Hugging Face `diffusers` experiments and tutorials (intro notebooks, Stable Diffusion pipelines). |
| [`iap-diffusion-labs/`](iap-diffusion-labs/) | MIT 6.S184/6.S975 *Generative AI with SDEs* labs. |
| `data/`, `work_dirs/` | Datasets and training outputs (gitignored). |
