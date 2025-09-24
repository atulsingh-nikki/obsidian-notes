---
title: "RetinexNet: Deep Retinex Decomposition for Low-Light Image Enhancement (2018)"
aliases:
  - RetinexNet
  - Deep Retinex
authors:
  - Chen Wei
  - Wenjing Wang
  - Wenhan Yang
  - Wei Liu
  - Jiaying Liu
year: 2018
venue: CVPR
doi: 10.1109/CVPR.2018.00298
citations: 3,000+
tags:
  - paper
  - deep-learning
  - low-light-enhancement
  - retinex
  - computer-vision
fields:
  - vision
  - computational-photography
  - image-enhancement
related:
  - "[[Multi-Scale Retinex (MSR, 1997)]]"
  - "[[Retinex Theory of Color Vision (1971)|Retinex]]"
  - "[[MSRCR Multi-Scale Retinex with Color Restoration]]"
predecessors:
  - MSRCR
successors:
  - KinD (2019)
  - EnlightenGAN (2019)
  - Zero-DCE (2020)
impact: ⭐⭐⭐⭐
status: read
---

# Summary
**RetinexNet** reformulates **Retinex theory** in a deep learning framework to tackle **low-light image enhancement**.  
It learns to decompose an input image into **reflectance** (intrinsic scene colors) and **illumination** components, and then adjusts illumination to brighten the image while preserving natural colors.

# Key Idea
> Train a **deep decomposition network** to split images into reflectance + illumination according to Retinex theory, then enhance illumination with a lightweight adjustment network.

# Method
- **Decomposition Net (Decom-Net):**  
  Learns to separate input low-light image into reflectance (shared across exposures) and illumination maps.  
- **Illumination Adjustment Net:**  
  Enhances illumination map with spatial smoothness constraints.  
- **Reflectance Refinement:**  
  Denoising applied to reflectance to reduce noise amplification.  
- **Losses:**  
  Reconstruction loss, reflectance consistency loss, illumination smoothness loss.  

# Results
- Produces visually pleasing low-light enhancement with reduced noise.  
- Outperforms traditional Retinex and histogram-based methods.  
- Dataset: **LOL (Low-Light dataset)** introduced with paired low/normal-light images.  

# Why it Mattered
- Brought **Retinex theory into the deep learning era**.  
- Showed decomposition + learning is effective for low-light tasks.  
- Influenced later models like KinD, Zero-DCE, and Retinex-inspired GANs.  

# Architectural Pattern
- Encoder-decoder networks for decomposition.  
- Illumination refinement branch with smoothness regularization.  

# Connections
- Continuation of Retinex → MSR → MSRCR.  
- Competes with GAN-based enhancement (EnlightenGAN, 2019).  
- Inspired lightweight Retinex models for mobile/embedded use.  

# Implementation Notes
- Requires paired low-light/normal-light training data (LOL dataset).  
- Architecture is lightweight but not real-time for high-res.  
- Widely reimplemented in PyTorch/TensorFlow.  

# Critiques / Limitations
- Struggles with extreme low-light + noise.  
- Requires ground truth well-lit images → supervision bottleneck.  
- Sometimes produces over-saturated or unnatural colors.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Retinex decomposition: illumination × reflectance = observed image.  
- Why separating lighting from surface helps enhancement.  
- Basics of encoder-decoder neural networks.  

## Postgraduate-Level Concepts
- Deep Retinex decomposition learning with explicit constraints.  
- Loss design: reconstruction, smoothness, consistency.  
- Role of paired datasets in supervised low-light enhancement.  
- Extensions: unpaired training, GAN-based enhancement, zero-shot approaches.  

---

# My Notes
- RetinexNet = **modern bridge between classical Retinex theory and deep learning**.  
- Landmark: introduced the **LOL dataset**, standard in low-light research.  
- Sparked a wave of Retinex-inspired deep models (KinD, KinD++, Zero-DCE).  
- Still a reference point in low-light literature.  

---
