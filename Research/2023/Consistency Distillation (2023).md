---
title: "Consistency Distillation (2023)"
aliases:
  - Consistency Distillation
  - CD
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
citations: 500+
dataset:
  - CIFAR-10
  - ImageNet
  - LSUN
  - COCO (evaluation)
tags:
  - paper
  - diffusion
  - generative-models
  - distillation
  - fast-sampling
  - consistency
fields:
  - vision
  - generative-models
related:
  - "[[Consistency Models (2023)]]"
  - "[[Stable Diffusion Turbo (2023)]]"
  - "[[DDIM (2020)]]"
predecessors:
  - "[[DDPM (2020)]]"
  - "[[Diffusion Models Beat GANs (2021)]]"
successors:
  - "[[Real-Time Foundation Models (2024+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Consistency Distillation (CD)** is a technique to distill **pretrained diffusion models** into **1–2 step generative models** while maintaining high sample quality. It makes **consistency models practical**, as they can inherit knowledge from large, well-trained diffusion teachers instead of training from scratch.

# Key Idea
> Distill a **multi-step diffusion model** into a **single-step (or few-step) student** using a **consistency training objective**, enabling orders-of-magnitude faster sampling without retraining on raw data.

# Method
- **Teacher**: A pretrained diffusion model (e.g., DDPM, Imagen, Stable Diffusion).  
- **Student**: A consistency model trained with teacher outputs.  
- **Objective**: Enforce consistency between teacher predictions at different noise levels.  
- **Outcome**: Student can generate high-quality samples in 1–2 denoising steps.  

# Results
- On CIFAR-10, ImageNet, LSUN: achieved **FID scores close to teacher** with 1–2 steps.  
- Orders of magnitude faster sampling vs original diffusion.  
- Outperformed prior fast-sampling methods like DDIM.  

# Why it Mattered
- Solved the practicality gap of consistency models.  
- Provided a general recipe to turn any diffusion model into a **real-time generator**.  
- Opened the door to **interactive, low-latency generative AI**.  

# Architectural Pattern
- Teacher diffusion → student consistency model.  
- Training with consistency regularization.  
- Sampling in 1–2 steps.  

# Connections
- Builds directly on **Consistency Models (2023)**.  
- Related to **Stable Diffusion Turbo (2023)** (another distillation path).  
- Complements fast samplers like **DDIM** but more principled.  

# Implementation Notes
- Can be applied to any pretrained diffusion model.  
- Training requires access to teacher outputs, not raw data.  
- Code and pretrained demos available.  

# Critiques / Limitations
- Quality still slightly below teacher at very high resolutions.  
- Distillation cost upfront (one-time).  
- Limited adoption so far compared to latent diffusion distillations.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Distillation: teacher–student training paradigm.  
- Why fewer steps = faster generation.  
- Difference between training from scratch vs distilling an existing model.  
- Applications: fast image generation, interactive tools.  

## Postgraduate-Level Concepts
- Consistency regularization across noise scales.  
- Distillation trade-offs: fidelity vs efficiency.  
- Comparison of CD vs adversarial distillation (SD Turbo).  
- Extending CD to multimodal models (text-to-video, 3D).  

---

# My Notes
- CD = the **practical engine** for consistency models.  
- Biggest promise: **turn any diffusion model into real-time** without retraining on raw data.  
- Open question: Can CD scale to **text-to-video and 3D diffusion** efficiently?  
- Possible extension: Combine **Consistency Distillation + latent diffusion** for optimal efficiency.  

---
