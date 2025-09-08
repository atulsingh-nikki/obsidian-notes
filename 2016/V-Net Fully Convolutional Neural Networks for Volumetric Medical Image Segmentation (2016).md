---
title: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation (2016)"
aliases:
  - V-Net
authors:
  - Fausto Milletari
  - Nassir Navab
  - Seyed-Ahmad Ahmadi
year: 2016
venue: "3DV (International Conference on 3D Vision)"
doi: "10.1109/3DV.2016.79"
arxiv: "https://arxiv.org/abs/1606.04797"
citations: 7000+
dataset:
  - Prostate MRI
tags:
  - paper
  - medical-imaging
  - segmentation
  - cnn
  - volumetric
fields:
  - vision
  - medical-imaging
  - segmentation
related:
  - "[[U-Net (2015)]]"
  - "[[3D U-Net (2016)]]"
  - "[[Attention U-Net (2018)]]"
predecessors:
  - "[[U-Net (2015)]]"
successors:
  - "[[3D U-Net (2016)]]"
  - "[[nnU-Net (2018)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**V-Net** extended the U-Net architecture to **3D volumetric medical image segmentation** using fully convolutional 3D CNNs. It introduced a **Dice loss function** for handling class imbalance, making it highly effective for medical datasets where foreground (e.g., tumors, organs) is small relative to background.

# Key Idea
> Apply fully convolutional 3D CNNs to volumetric data, with encoder–decoder design and residual connections, optimized using **Dice loss** instead of cross-entropy.

# Method
- **Architecture**:  
  - Encoder–decoder structure (similar to U-Net).  
  - Residual blocks for stability.  
  - 3D convolutions throughout.  
- **Loss**:  
  - **Dice coefficient loss**: directly optimizes overlap between predicted and ground-truth segmentations, handling severe class imbalance.  
- **Training**: Patch-based sampling of volumes for efficiency.  

# Results
- Achieved strong performance on **prostate MRI segmentation**.  
- Outperformed 2D approaches by leveraging full 3D context.  
- Demonstrated robustness to class imbalance.  

# Why it Mattered
- One of the first successful **3D CNN architectures** for volumetric segmentation.  
- Introduced **Dice loss**, now a standard in medical imaging.  
- Inspired later volumetric and hybrid models like **3D U-Net** and **nnU-Net**.  

# Architectural Pattern
- 3D encoder–decoder CNN.  
- Residual connections.  
- Dice-based optimization.  

# Connections
- Successor to **U-Net (2015)**.  
- Predecessor to **3D U-Net (2016)**, **Attention U-Net (2018)**, **nnU-Net (2018)**.  
- Related to volumetric applications in CT, MRI, PET.  

# Implementation Notes
- Computationally heavy (3D convolutions).  
- Patch-wise training mitigates GPU memory limits.  
- Dice loss particularly helpful for skewed class distributions.  

# Critiques / Limitations
- Focused on single-organ segmentation (prostate MRI).  
- Heavy memory requirements compared to 2D CNNs.  
- Performance depends strongly on preprocessing/patch sampling.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What volumetric segmentation means (3D medical scans like MRI/CT).  
- Difference between 2D CNNs (images) and 3D CNNs (volumes).  
- Why class imbalance is a problem in medical data.  
- Example: detecting a small tumor within a large MRI scan.  

## Postgraduate-Level Concepts
- Dice loss derivation and gradient properties.  
- Encoder–decoder design in 3D CNNs.  
- Trade-offs between patch-wise vs full-volume training.  
- Extensions: multi-modal MRI segmentation, hybrid CNN-transformers for 3D.  

---

# My Notes
- V-Net = **U-Net’s 3D cousin** with Dice loss innovation.  
- Set the stage for volumetric segmentation as the medical imaging standard.  
- Open question: Will **transformers replace 3D CNNs** in volumetric segmentation, or will hybrid designs dominate?  
- Possible extension: Integrating V-Net with **self-supervised pretraining on 3D medical scans**.  

---
