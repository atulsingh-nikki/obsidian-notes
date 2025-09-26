---
title: "KinD: Kindling the Darkness — A Practical Low-Light Image Enhancer (2019)"
aliases:
  - KinD
  - KinD++
authors:
  - Yuanjie Zhang
  - Yongjie Zhang
  - Yulun Zhang
  - Huchuan Lu
  - Ling Shao
year: 2019
venue: ACM Multimedia (MM)
doi: 10.1145/3343031.3351084
citations: 1,800+
tags:
  - paper
  - deep-learning
  - low-light-enhancement
  - retinex
fields:
  - vision
  - computational-photography
related:
  - "[[Zero-DCE Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (2020)|Zero-DCE]]"
  - "[[Retinex Theory of Color Vision (1971)|Retinex]]"
predecessors:
  - RetinexNet
successors:
  - KinD++
impact: ⭐⭐⭐⭐
status: read
---

# Summary
**KinD** extended RetinexNet by jointly addressing **illumination enhancement** and **reflectance restoration**.  
It tackled RetinexNet’s weaknesses in noise handling and over-saturation, aiming for both **brightening** and **denoising** in a unified framework.

# Key Idea
> A two-branch network for **illumination correction** and **reflectance restoration**, coupled with task-specific losses.

# Method
- **Decomposition:** Similar to RetinexNet, split into reflectance + illumination.  
- **Illumination Enhancement Branch:** Smooth and correct illumination maps.  
- **Reflectance Restoration Branch:** Suppress noise/artifacts, restore detail.  
- **Losses:**  
  - Reflectance consistency  
  - Illumination smoothness  
  - Structural similarity  
- **KinD++ (2020):** Improved robustness, faster, better perceptual quality.  

# Results
- Enhanced low-light images with reduced noise.  
- Better structure preservation compared to RetinexNet.  
- Released code/dataset → widely adopted baseline.  

# Critiques
- Still requires supervised paired training.  
- Sometimes oversmooths fine details.  

# Educational Connections
- Undergrad: Enhancing brightness vs denoising.  
- Postgrad: Multi-branch design for decomposition tasks, specialized losses.  

---
