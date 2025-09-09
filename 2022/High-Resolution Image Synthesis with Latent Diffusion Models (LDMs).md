---
title: "High-Resolution Image Synthesis with Latent Diffusion Models (LDMs)"
authors:
  - Robin Rombach
  - Andreas Blattmann
  - Dominik Lorenz
  - Patrick Esser
  - Björn Ommer
year: 2022
venue: "CVPR 2022"
dataset:
  - LAION-400M
  - ImageNet
  - COCO
tags:
  - generative-models
  - diffusion-models
  - latent-representations
  - computer-vision
  - text-to-image
  - stable-diffusion
arxiv: "https://arxiv.org/abs/2112.10752"
related:
  - "[[Denoising Diffusion Probabilistic Models (DDPM, 2020)]]"
  - "[[Denoising Diffusion Implicit Models (DDIM, 2020)]]"
  - "[[Stable Diffusion]]"
  - "[[Autoencoders (VAE)]]"
  - "[[Transformers in Vision]]"
---

# Summary
Latent Diffusion Models (LDMs) proposed training diffusion models not in pixel space, but in a **compressed latent space** learned by a **variational autoencoder (VAE)**. This drastically reduced compute costs while enabling **high-resolution image synthesis** (e.g., 512×512, 1024×1024). With conditioning mechanisms (like text embeddings from CLIP/Transformers), LDMs became the foundation of **Stable Diffusion** and modern text-to-image generation.

# Key Idea (one-liner)
> Learn to denoise in a compressed latent space instead of pixel space, enabling efficient and scalable high-resolution image synthesis.

# Method
- **Autoencoder (VAE)**:
  - Encoder compresses image into lower-dimensional latent space.
  - Decoder reconstructs images from latent space.
- **Diffusion process**:
  - Train diffusion model in latent space instead of raw pixels.
  - Forward: gradually add Gaussian noise to latent.
  - Reverse: learn denoising process.
- **Conditioning**:
  - Text encoder (e.g., CLIP or Transformer) conditions the diffusion model.
  - Cross-attention mechanism aligns text embeddings with latent features.
- **Efficiency**:
  - Latent space ~48× smaller than pixels → compute & memory savings.
  - Enables training on high-resolution images with commodity GPUs.

# Results
- High-quality, diverse images at resolutions up to 1024×1024.
- Orders-of-magnitude more efficient than pixel-based diffusion.
- First open, practical text-to-image model: foundation for **Stable Diffusion**.
- Competitive with GANs on FID/IS, but more stable and controllable.

# Why it Mattered
- Made diffusion models **practical and scalable**.
- Enabled **open-source text-to-image generation** (Stable Diffusion).
- Replaced GANs as dominant paradigm for generative vision.
- Demonstrated flexibility: image editing, inpainting, super-resolution.

# Architectural Pattern
- [[Autoencoder (VAE)]] → compress/reconstruct images.
- [[Diffusion Model]] → trained in latent space.
- [[Cross-Attention Conditioning]] → text-to-image alignment.
- [[Transformer/UNet Backbones]] → scalable denoising networks.

# Connections
- **Predecessors**:
  - [[DDPM (2020)]] — pixel-space diffusion.
  - [[DDIM (2020)]] — improved inference sampling.
- **Successors**:
  - [[Stable Diffusion (2022)]] — open-source deployment.
  - Imagen (Google, 2022), Parti — large-scale text-to-image transformers.
- **Influence**:
  - Foundation for generative art, controllable synthesis, multimodal diffusion.
  - Widely used in industry and research.

# Implementation Notes
- Training requires large datasets (e.g., LAION-400M).
- Cross-attention conditioning crucial for text/image alignment.
- VAE compression ratio determines efficiency vs reconstruction fidelity.
- Inference with 20–50 denoising steps (fast vs quality trade-off).
- Pretrained weights widely available (Stable Diffusion family).

# Critiques / Limitations
- Image fidelity depends on VAE reconstruction quality.
- Sensitive to dataset biases (web-scale LAION data).
- Text alignment imperfect (hallucinations, misinterpretations).
- Requires strong GPUs for training and large memory for inference at scale.

# Repro / Resources
- Paper: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Official repo: [CompVis GitHub](https://github.com/CompVis/latent-diffusion)
- Models: Stable Diffusion (SD 1.x, 2.x, XL).
- Datasets: [[LAION-400M]], [[COCO]], [[ImageNet]].

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Matrix multiplications in encoder/decoder.
  - Latent representations as compressed vectors.

- **Probability & Statistics**
  - Gaussian noise in forward diffusion.
  - Sampling from learned distributions.

- **Calculus**
  - Gradients of reconstruction + denoising losses.
  - Backprop through diffusion and VAE.

- **Signals & Systems**
  - Compression (encoder) and reconstruction (decoder).
  - Noise addition/removal as filtering.

- **Data Structures**
  - Latent space tensors.
  - Conditioning embeddings from text.

- **Optimization Basics**
  - SGD/AdamW optimizers.
  - Trade-off between steps (speed) and quality.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Balancing reconstruction loss and perceptual loss in VAE.
  - Training stability in diffusion models.
  - Guidance (classifier-free guidance) for controllability.

- **Numerical Methods**
  - Sampling efficiency (DDIM, ancestral sampling).
  - Trade-off between steps vs fidelity.
  - Memory optimization for high resolution.

- **Machine Learning Theory**
  - Latent-variable modeling.
  - Diffusion models vs GANs (mode coverage, stability).
  - Conditioning as conditional generative modeling.

- **Computer Vision**
  - High-resolution synthesis.
  - Applications: inpainting, super-resolution, text-to-image.
  - Evaluation metrics (FID, IS, CLIPScore).

- **Neural Network Design**
  - UNet backbone for denoising.
  - Cross-attention for multimodal conditioning.
  - VAE encoder/decoder architecture.

- **Transfer Learning**
  - Pretrained LDM → fine-tuning for style, domain, personalization.
  - LoRA, DreamBooth for downstream adaptation.

- **Research Methodology**
  - Ablations: VAE compression ratio, conditioning strategies.
  - Benchmarks against GANs and pixel-based diffusion.
  - Scaling laws for resolution and dataset size.
