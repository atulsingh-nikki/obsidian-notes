---
title: "Denoising Diffusion Implicit Models (DDIM, 2020)"
aliases:
  - DDIM
  - Denoising Diffusion Implicit Models
authors:
  - Jiaming Song
  - Chenlin Meng
  - Stefano Ermon
year: 2020
venue: "arXiv"
doi: "10.48550/arXiv.2010.02502"
arxiv: "https://arxiv.org/abs/2010.02502"
code: "https://github.com/ermongroup/ddim"
citations: 6000+
dataset:
  - CIFAR-10
  - LSUN
  - CelebA-HQ
  - ImageNet subsets
tags:
  - paper
  - diffusion-model
  - generative-model
  - fast-sampling
  - unsupervised
fields:
  - vision
  - generative-models
  - deep-learning
related:
  - "[[Denoising Diffusion Probabilistic Models (DDPM, 2020)]]"
  - "[[Score-Based Generative Modeling (NCSN, 2019/2020)]]"
  - "[[Latent Diffusion Models (LDM, 2022)]]"
predecessors:
  - "[[DDPM (Ho et al., 2020)]]"
successors:
  - "[[Latent Diffusion Models (2022)]]"
  - "[[Consistency Models (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**DDIM** proposed an **implicit non-Markovian sampling process** for diffusion models, dramatically reducing the number of required sampling steps compared to DDPM, while maintaining or even improving sample quality. This made diffusion models **practically usable** for large-scale image synthesis.

# Key Idea
> Reformulate the reverse diffusion process as a **deterministic or stochastic implicit model**, allowing flexible sampling with far fewer steps.

# Method
- **DDPM baseline**: Requires hundreds–thousands of steps for high-quality samples.  
- **DDIM reparameterization**:  
  - Derives a family of non-Markovian diffusion processes with the same marginal distributions.  
  - Sampling can be **deterministic (ODE-like)** or **stochastic (SDE-like)**.  
- **Key property**: Generates samples consistent with the same training objective (no retraining required).  
- **Sampling speed**: 10–50 steps instead of 1000+.  

# Results
- Matched or outperformed DDPMs with far fewer sampling steps.  
- CIFAR-10 FID competitive with GANs of the time.  
- Showed smooth interpolation in latent space due to deterministic formulation.  

# Why it Mattered
- Solved the **efficiency bottleneck** of diffusion models.  
- Enabled diffusion to become a mainstream generative model, later powering large-scale models (LDMs, Stable Diffusion).  
- Influenced follow-ups in fast inference (DDIM inversion, consistency models, DPM-Solver).  

# Architectural Pattern
- Same U-Net backbone as DDPM.  
- Modified sampling procedure (implicit steps).  
- Training unchanged from DDPM.  

# Connections
- Builds directly on **DDPM (Ho et al., 2020)**.  
- Predecessor to **Latent Diffusion Models (2022)**.  
- Related to **ODE/SDE view of diffusion** (Song et al., 2021).  

# Implementation Notes
- Works with pretrained DDPMs (no retraining needed).  
- Sampling steps are tunable (quality-speed tradeoff).  
- Deterministic sampling enables latent interpolation and inversion.  

# Critiques / Limitations
- Still slower than GANs for high-res generation.  
- Deterministic DDIM lacks stochastic diversity unless noise added back.  
- Needs large U-Nets for image fidelity.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Diffusion as gradual denoising.  
- Sampling steps and efficiency trade-offs.  
- Difference between deterministic vs stochastic generative models.  

## Postgraduate-Level Concepts
- Non-Markovian diffusion processes.  
- ODE/SDE interpretations of generative models.  
- Inversion and interpolation in diffusion models.  

---

# My Notes
- DDIM was the **practical breakthrough**: made diffusion usable beyond toy datasets.  
- Deterministic sampling → cool property: **invertible diffusion** (later used in editing/inversion tasks).  
- Open question: Can implicit samplers be merged with **consistency training** for one-step generation?  
- Possible extension: Combine DDIM inversion with **video diffusion** for precise frame-level editing.  

---
