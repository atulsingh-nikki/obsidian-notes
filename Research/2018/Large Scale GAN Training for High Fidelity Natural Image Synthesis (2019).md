---
title: "Large Scale GAN Training for High Fidelity Natural Image Synthesis (2019)"
aliases: 
  - BigGAN
  - Large Scale GAN Training
authors:
  - Andrew Brock
  - Jeff Donahue
  - Karen Simonyan
year: 2019
venue: "ICLR"
doi: "10.48550/arXiv.1809.11096"
arxiv: "https://arxiv.org/abs/1809.11096"
code: "https://github.com/ajbrock/BigGAN-PyTorch"
citations: 10,000+
dataset:
  - ImageNet (1k classes, 128x128, 256x256, 512x512 resolutions)
tags:
  - paper
  - gan
  - generative-models
  - image-synthesis
fields:
  - vision
  - generative-models
related:
  - "[[StyleGAN (2019)]]"
  - "[[Progressive GAN (2018)]]"
predecessors:
  - "[[Progressive Growing of GANs (2018)]]"
successors:
  - "[[StyleGAN (2019)]]"
  - "[[Diffusion Models]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
BigGAN demonstrated that scaling up GANs—**larger models, larger batch sizes, larger datasets**—could dramatically improve the **fidelity and diversity** of generated images. Trained on ImageNet, BigGAN produced some of the most photorealistic images of its time.

# Key Idea
> GAN performance scales with **model size, batch size, and dataset scale**, but requires careful stabilization techniques.

# Method
- **Architecture**:  
  - Based on ResNet-style generator and discriminator.  
  - Spectral normalization applied to both networks for stability.  
  - Class-conditional GAN with shared embeddings.  
- **Scaling**:  
  - Very large models (up to 166M parameters).  
  - Batch sizes up to 2048 on TPUs.  
- **Tricks for stability**:  
  - Orthogonal regularization.  
  - Truncation trick (sampling latent vectors from truncated Gaussian for higher fidelity).  
  - Shared embeddings for efficient conditional generation.  

# Results
- Achieved **state-of-the-art Inception Score (IS) and FID** on ImageNet.  
- Generated **128×128, 256×256, and 512×512** resolution images with unprecedented realism.  
- Demonstrated trade-off between sample fidelity and variety (truncation trick).  

# Why it Mattered
- Showed that **scaling laws apply to GANs**, not just supervised models.  
- Produced the first GAN images that rivaled real ImageNet samples.  
- Paved the way for StyleGAN and later diffusion models.  

# Architectural Pattern
- ResNet-based GAN architecture.  
- Spectral normalization for stability.  
- Conditional generation with class embeddings.  

# Connections
- **Contemporaries**: Progressive GAN (2018), SAGAN (2018).  
- **Influence**: StyleGAN (2019), diffusion models (2020+).  

# Implementation Notes
- Extremely resource-intensive (TPUs, huge batch sizes).  
- Truncation trick allows user to trade-off fidelity vs diversity.  
- Orthogonal regularization stabilizes training but needs tuning.  

# Critiques / Limitations
- Computationally expensive → impractical for most labs.  
- Sensitive to hyperparameters and training setup.  
- Still unstable compared to diffusion models.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1809.11096)  
- [Official PyTorch implementation](https://github.com/ajbrock/BigGAN-PyTorch)  
- [Colab demos available]  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Matrix multiplications in ResNet layers.  
- **Probability & Statistics**: Sampling from truncated Gaussian.  
- **Optimization Basics**: GAN loss minimax game.  

## Postgraduate-Level Concepts
- **Neural Network Design**: ResNet-based generators/discriminators.  
- **Generative Models**: Conditional GANs, truncation trick.  
- **Research Methodology**: Scaling analysis for generative models.  
- **Advanced Optimization**: Spectral normalization, stability techniques.  

---

# My Notes
- BigGAN proved scaling works for GANs like it does for Transformers.  
- Relevant to **video synthesis** — scaling temporal GANs might boost realism.  
- Open question: Can **diffusion scaling** fully replace GAN scaling for high fidelity?  
- Possible extension: Explore truncation-like sampling strategies for **diffusion-based video editing**.  

---
