---
title: "Consistency Models (2023)"
aliases:
  - Consistency Models
  - CM
authors:
  - Yang Song
  - Prafulla Dhariwal
  - Mark Chen
  - Ilya Sutskever
year: 2023
venue: "arXiv preprint"
doi: "10.48550/arXiv.2303.01469"
arxiv: "https://arxiv.org/abs/2303.01469"
code: "https://github.com/openai/consistency_models"
citations: 600+
dataset:
  - CIFAR-10
  - ImageNet
  - LSUN
  - COCO (evaluation)
tags:
  - paper
  - diffusion
  - generative-models
  - fast-sampling
  - consistency
fields:
  - vision
  - generative-models
related:
  - "[[Stable Diffusion Turbo (2023)]]"
  - "[[DDIM (2020)]]"
  - "[[Score-based Models (2020)]]"
predecessors:
  - "[[DDPM (2020)]]"
  - "[[Score-based Generative Models (2020)]]"
successors:
  - "[[Consistency Distillation (2023+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Consistency Models (CMs)** introduced a new family of generative models that can produce **high-quality samples in just 1–2 steps**, bypassing the need for long iterative diffusion sampling. They combine ideas from **diffusion models** and **distillation**, offering both fast sampling and competitive quality.

# Key Idea
> Train a neural network to directly map from random noise to the data distribution in a **single forward pass**, while preserving consistency across intermediate steps.

# Method
- **Consistency training**:  
  - Ensure that outputs at different noise levels are **consistent mappings** to the same data distribution.  
- **Model types**:  
  - **Consistency Models**: train from scratch.  
  - **Consistency Distillation**: distill pretrained diffusion models into consistency models.  
- **Sampling**: 1–2 steps (orders of magnitude faster than DDPM).  

# Results
- Achieved competitive FID scores on CIFAR-10, ImageNet, LSUN.  
- Generated high-quality samples with just 1–2 evaluations.  
- Distilled models retained teacher diffusion quality at far fewer steps.  

# Why it Mattered
- Removed the main bottleneck of diffusion models: **slow iterative sampling**.  
- Offered a **parallel path to distillation** (different from SD Turbo).  
- Potential for real-time generative applications at scale.  

# Architectural Pattern
- Similar U-Net backbone to diffusion models.  
- Training objective: enforce **consistency across denoising trajectories**.  

# Connections
- Related to **DDIM (2020)** (fast deterministic sampling).  
- Complements **Stable Diffusion Turbo (2023)** (distillation).  
- OpenAI’s follow-up to diffusion sampling speedups.  

# Implementation Notes
- Open-source implementation available.  
- Distillation variant more practical (use pretrained diffusion models).  
- Training requires careful schedule design.  

# Critiques / Limitations
- From-scratch CMs lag behind SOTA diffusion models in quality.  
- Distilled CMs dependent on teacher model capacity.  
- Not yet fully scaled to billion-parameter, internet-scale datasets.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Diffusion models require many steps → slow.  
- Consistency models reduce steps to **1–2** → fast.  
- Idea of distillation: compress teacher model into faster student.  
- Applications: real-time art, image editing, interactive AI.  

## Postgraduate-Level Concepts
- Consistency regularization across noise scales.  
- Comparison of CM vs distillation vs DDIM acceleration.  
- Trade-offs: speed vs fidelity vs generalization.  
- Potential for extending to **video, 3D, multimodal models**.  

---

# My Notes
- Consistency Models were a **major breakthrough**: true 1-step generative modeling.  
- Still early days: need scaling to Imagen/SDXL levels.  
- Open question: Will **consistency training replace iterative diffusion** in production systems?  
- Possible extension: Combine **CM speed** with **latent diffusion efficiency** → real-time multimodal foundation models.  

---
