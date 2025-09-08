---
title: "Stable Diffusion (2022)"
aliases:
  - Stable Diffusion
  - Latent Diffusion Models
authors:
  - Robin Rombach
  - Andreas Blattmann
  - Dominik Lorenz
  - Patrick Esser
  - Björn Ommer
year: 2022
venue: "arXiv preprint"
doi: "10.48550/arXiv.2112.10752"
arxiv: "https://arxiv.org/abs/2112.10752"
code: "https://github.com/CompVis/stable-diffusion"
citations: 7000+
dataset:
  - LAION-5B (filtered subsets)
  - COCO (evaluation)
tags:
  - paper
  - diffusion
  - text-to-image
  - latent-space
  - open-source
fields:
  - vision
  - language
  - generative-models
  - multimodal
related:
  - "[[Imagen (2022)]]"
  - "[[Parti (2022)]]"
  - "[[Muse (2023)]]"
predecessors:
  - "[[GLIDE (2021)]]"
  - "[[LAION-5B (2022)]]"
successors:
  - "[[Stable Diffusion XL (2023)]]"
  - "[[Open multimodal foundation models (2023+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Stable Diffusion** introduced **Latent Diffusion Models (LDMs)**, moving diffusion from pixel space to a **compressed latent space**, making training and sampling vastly more efficient. Released as fully open-source, it democratized access to **high-quality text-to-image generation** and sparked the global generative AI ecosystem.

# Key Idea
> Perform diffusion in the **latent space of a pretrained autoencoder**, not pixel space — reducing cost while retaining fidelity. Train on **open LAION data**, release models openly.

# Method
- **Autoencoder**: Trains a VQ-style variational autoencoder (VAE) to compress images into latent space.  
- **Diffusion**: Denoising diffusion model trained in latent space (lower dimensionality).  
- **Text conditioning**: Cross-attention with CLIP text encoder.  
- **Sampling**: Classifier-free guidance for controllability.  
- **Resolution**: Efficiently scales to 512×512 and beyond.  

# Results
- Generated **photorealistic, high-resolution images** from text prompts.  
- Achieved quality comparable to Imagen and DALLE-2.  
- Orders of magnitude more efficient than pixel-space diffusion.  

# Why it Mattered
- **Open release** (weights + code + dataset pipeline) — democratized text-to-image research.  
- Enabled explosion of **applications, fine-tuned models, extensions (ControlNet, DreamBooth)**.  
- Introduced **latent diffusion paradigm**, now standard in generative modeling.  

# Architectural Pattern
- VAE (image compression).  
- U-Net diffusion in latent space.  
- Text conditioning with CLIP embeddings.  

# Connections
- Built on GLIDE (text-guided diffusion) and LAION-5B (dataset).  
- Competitor to Imagen (Google) and DALLE-2 (OpenAI) — but open-source.  
- Spawned Stable Diffusion XL and ControlNet ecosystem.  

# Implementation Notes
- Training feasible on high-end consumer GPUs.  
- Latent space diffusion ~10–100× cheaper than pixel diffusion.  
- HuggingFace integration accelerated adoption.  

# Critiques / Limitations
- Training dataset (LAION) noisy, biased, and copyright-concerning.  
- Lower text alignment compared to Imagen.  
- Ethical concerns: misuse for deepfakes, harmful imagery.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What diffusion models are (iterative denoising).  
- Why compressing to **latent space** makes training faster.  
- Basics of text conditioning via CLIP.  
- Applications: art, design, concept visualization.  

## Postgraduate-Level Concepts
- Comparison of pixel-space vs latent-space diffusion.  
- Trade-offs: efficiency vs fidelity.  
- Ecosystem: fine-tuning (LoRA, DreamBooth), controllability (ControlNet).  
- Societal/ethical issues: copyright, safety, democratization vs misuse.  

---

# My Notes
- Stable Diffusion was the **iPhone moment** of generative AI: open, efficient, widely adopted.  
- Latent diffusion = elegant idea: compress first, then diffuse.  
- Open question: How to balance **openness vs responsible use** in dataset/model release?  
- Possible extension: Extend latent diffusion to **video, 3D, multimodal foundation models**.  

---
