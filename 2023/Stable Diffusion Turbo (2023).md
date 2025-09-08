---
title: "Stable Diffusion Turbo (2023)"
aliases:
  - SD Turbo
  - Stable Diffusion Turbo
authors:
  - Stability AI Research
  - Robin Rombach
  - Andreas Blattmann
  - et al.
year: 2023
venue: "arXiv preprint"
doi: "10.48550/arXiv.2311.16543"
arxiv: "https://arxiv.org/abs/2311.16543"
code: "https://huggingface.co/stabilityai/sdxl-turbo"
citations: 100+
dataset:
  - Same LAION-derived + refined datasets as SDXL
tags:
  - paper
  - diffusion
  - text-to-image
  - distillation
  - real-time
fields:
  - vision
  - generative-models
  - multimodal
related:
  - "[[Stable Diffusion (2022)]]"
  - "[[Stable Diffusion XL (2023)]]"
  - "[[Imagen (2022)]]"
  - "[[Parti (2022)]]"
predecessors:
  - "[[Stable Diffusion XL (2023)]]"
successors:
  - "[[Next-Gen Distilled Diffusion Models (2024+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**Stable Diffusion Turbo (SD Turbo)** distilled **SDXL** into a model that can generate **photorealistic images in a single step**. Using **adversarial distillation**, it achieved real-time inference while retaining much of SDXL’s fidelity, bringing text-to-image synthesis closer to interactive applications.

# Key Idea
> Apply **knowledge distillation + adversarial training** to compress SDXL into a fast student model that generates high-quality images in just **1–4 denoising steps**.

# Method
- **Teacher**: SDXL base + refiner diffusion pipeline.  
- **Student**: Distilled model trained to approximate teacher outputs.  
- **Distillation**: Match teacher’s intermediate steps with far fewer iterations.  
- **Adversarial training**: Use a discriminator to sharpen realism and avoid blurriness.  
- **Sampling**: Real-time capable (1–4 steps instead of 50–100).  

# Results
- Generated images with **quality close to SDXL** at a fraction of compute cost.  
- Achieved **interactive, near real-time text-to-image generation** on consumer GPUs.  
- Demonstrated scalability of adversarial distillation for diffusion.  

# Why it Mattered
- Solved diffusion’s biggest bottleneck: **sampling speed**.  
- Enabled new use cases: interactive design, AR/VR, on-device creative tools.  
- First widely released **real-time open-source diffusion model**.  

# Architectural Pattern
- Teacher–student distillation.  
- Adversarial discriminator to stabilize quality.  
- Minimal diffusion steps at inference.  

# Connections
- Successor to **SDXL (2023)**.  
- Related to **Progressive Distillation (Salimans & Ho, 2022)**.  
- Complements open-source ecosystem (ControlNet, LoRA fine-tuning).  

# Implementation Notes
- HuggingFace weights released publicly.  
- Runs on consumer GPUs at real-time speed.  
- Supports same ecosystem of fine-tuning (LoRA, ControlNet).  

# Critiques / Limitations
- Slightly lower fidelity than full SDXL at high resolutions.  
- Adversarial component adds instability in training.  
- Limited to image generation; extensions to video still experimental.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What knowledge distillation is (teacher–student training).  
- Why reducing diffusion steps makes generation faster.  
- Basics of adversarial training (generator vs discriminator).  
- Applications: interactive art, real-time image editing.  

## Postgraduate-Level Concepts
- Trade-offs in diffusion distillation (fidelity vs efficiency).  
- Distillation vs native fast samplers (e.g., DDIM, consistency models).  
- Integration with ecosystem: how SD Turbo enables LoRA + ControlNet workflows at speed.  
- Implications for **on-device generative AI**.  

---

# My Notes
- SD Turbo was the **efficiency breakthrough** for open diffusion models.  
- Feels like the missing piece: **real-time diffusion** without losing quality.  
- Open question: Can distillation extend to **video diffusion** or **multimodal generation**?  
- Possible extension: Combine Turbo with **consistency models** for even faster, stable inference.  

---
