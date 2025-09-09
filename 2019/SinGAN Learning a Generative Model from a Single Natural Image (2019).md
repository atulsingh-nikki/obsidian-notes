---
title: "SinGAN: Learning a Generative Model from a Single Natural Image (2019)"
aliases: 
  - SinGAN
  - Single-Image GAN
authors:
  - Tamar Rott Shaham
  - Tali Dekel
  - Tomer Michaeli
year: 2019
venue: "ICCV"
doi: "10.1109/ICCV.2019.00089"
arxiv: "https://arxiv.org/abs/1905.01164"
code: "https://github.com/tamarott/SinGAN"
citations: 5000+
dataset:
  - Single natural images (no multi-image dataset)
tags:
  - paper
  - gan
  - image-synthesis
  - single-image-learning
fields:
  - vision
  - generative-models
  - image-editing
related:
  - "[[PatchGAN (2017)]]"
  - "[[Single Image GAN Applications]]"
predecessors:
  - "[[GAN (2014)]]"
  - "[[Pix2Pix (2017)]]"
successors:
  - "[[ConSinGAN (2020)]]"
  - "[[Diffusion-based single-image models]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
SinGAN introduced a **multi-scale generative adversarial network** trained on a **single natural image**. Unlike traditional GANs that need large datasets, SinGAN learns internal patch statistics of one image and can generate diverse, realistic variations of it.

# Key Idea
> Leverage the **internal patch distribution** of a single natural image, across scales, to train a generative model capable of producing new plausible samples.

# Method
- **Architecture**:  
  - Multi-scale pyramid of GANs, each trained to capture image statistics at a different resolution.  
  - Coarse scales learn global structure, fine scales learn local texture details.  
- **Training**:  
  - Trains sequentially from coarse to fine scales on patches from the single image.  
  - Each scale’s generator takes input noise + upsampled image from previous scale.  
- **Output**: Generates new images resembling the input’s “visual world.”  

# Results
- Generated realistic samples from a **single input image**.  
- Applications:  
  - Image super-resolution.  
  - Harmonization (inserting objects into an image).  
  - Editing/paint-to-image.  
  - Animation synthesis.  
  - Image blending.  
- Outperformed prior patch-based synthesis methods.  

# Why it Mattered
- Showed that **GANs don’t always need large datasets**.  
- Highlighted the power of **internal statistics of natural images**.  
- Opened a research line on **single-image generative modeling**.  

# Architectural Pattern
- Multi-scale pyramid of GANs.  
- Patch-based adversarial loss.  
- Noise injection at each scale.  

# Connections
- **Contemporaries**: StyleGAN (2019), BigGAN.  
- **Influence**: ConSinGAN, Single Image Diffusion models.  

# Implementation Notes
- Training is image-specific (slow, per-image model).  
- Generated diversity limited to patch-level variations.  
- Not suitable for large-scale datasets.  

# Critiques / Limitations
- Cannot generalize across images; one model per input image.  
- Global structure often inconsistent across samples.  
- Diffusion models now outperform SinGAN in single-image generation.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1905.01164)  
- [Official PyTorch code](https://github.com/tamarott/SinGAN)  
- [Colab demo notebooks available]  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Upsampling and pyramid operations.  
- **Probability & Statistics**: Patch distribution modeling.  
- **Optimization Basics**: GAN adversarial losses.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Multi-scale GANs.  
- **Computer Vision**: Image statistics and patch recurrence.  
- **Research Methodology**: Evaluating generative models without datasets.  
- **Advanced Optimization**: Sequential training across scales.  

---

# My Notes
- Relevant for **video editing from a single example frame** (texture, backgrounds).  
- Open question: Can diffusion replace the **multi-scale GAN pyramid** for one-image learning?  
- Possible extension: Use **SinGAN-like internal patch statistics** to regularize diffusion in low-data video editing.  

---
