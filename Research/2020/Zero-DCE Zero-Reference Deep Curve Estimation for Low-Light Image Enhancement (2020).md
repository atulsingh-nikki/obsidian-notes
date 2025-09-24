---
title: "Zero-DCE: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (2020)"
aliases:
  - Zero-DCE
authors:
  - Chengming Guo
  - Chongyi Li
  - Jichang Guo
  - Chen Change Loy
year: 2020
venue: CVPR
doi: 10.1109/CVPR42600.2020.01260
citations: 3,500+
tags:
  - paper
  - deep-learning
  - low-light-enhancement
  - zero-shot-learning
fields:
  - vision
  - computational-photography
related:
  - "[[RetinexNet Deep Retinex Decomposition for Low-Light Image Enhancement (2018)|RetinexNet]]"
  - "[[KinD: Kindling the Darkness — A Practical Low-Light Image Enhancer (2019)]]"
predecessors:
  - Retinex-inspired methods
successors:
  - Zero-DCE++
  - EnlightenGAN (comparable unpaired methods)
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**Zero-DCE** introduced a **zero-reference learning framework** for low-light enhancement.  
It avoided the need for paired low/normal-light datasets by directly learning a **light-curve mapping function** optimized with non-reference losses.

# Key Idea
> Learn per-pixel **curve estimation functions** that map low-light inputs to enhanced outputs, guided by perceptual and statistical constraints — no paired training data required.

# Method
- **Deep Curve Estimation Network (DCE-Net):** Predicts parametric curves for pixel intensity adjustment.  
- **Zero-Reference Losses:**  
  - Exposure control loss  
  - Color constancy loss  
  - Illumination smoothness loss  
  - Spatial consistency loss  
- **End-to-end training without ground truth.**  

# Results
- Outperformed supervised Retinex-based methods on real-world low-light images.  
- Robust, lightweight, and efficient → deployable on mobile devices.  
- Extended to **Zero-DCE++ (2021)** with better robustness.  

# Critiques
- Sometimes introduces color casts.  
- Limited to intensity remapping (not structural edits).  

# Educational Connections
- Undergrad: Why paired data is a bottleneck.  
- Postgrad: Designing **non-reference losses** for unsupervised training.  

---
