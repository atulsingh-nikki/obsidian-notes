---
title: "Holistically-Nested Edge Detection (2015)"
aliases: 
  - HED
  - Holistically-Nested Edge Detector
authors:
  - Saining Xie
  - Zhuowen Tu
year: 2015
venue: "ICCV"
doi: "10.1109/ICCV.2015.164"
arxiv: "https://arxiv.org/abs/1504.06375"
code: "https://github.com/s9xie/hed"
citations: 9000+
dataset:
  - BSDS500
  - NYUDv2
tags:
  - paper
  - edge-detection
  - cnn
  - vision
fields:
  - vision
  - deep-learning
related:
  - "[[Canny Edge Detector (1986)]]"
  - "[[Structured Edge Detection (Dollar & Zitnick, 2013)]]"
predecessors:
  - "[[Canny Edge Detector (1986)]]"
  - "[[DeepNet for Edge Detection (2014)]]"
successors:
  - "[[RCF: Richer Convolutional Features for Edge Detection]]"
  - "[[DexiNed]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---


# Summary
Holistically-Nested Edge Detection (HED) is a **deep learning–based edge detector** that formulates edge detection as a **pixel-wise classification problem**. It introduces deep supervision at multiple network stages, enforcing meaningful intermediate feature learning and fusing them to produce crisp, high-quality edge maps.

# Key Idea
> Train a fully convolutional network with deep supervision at multiple scales to predict edges holistically.

# Method
- Based on a **fully convolutional network (FCN)** architecture.  
- Uses **deep supervision**: each intermediate convolutional layer produces a side-output, supervised with ground-truth edges.  
- Final prediction is a **weighted fusion of multi-scale outputs**, capturing both fine and coarse details.  
- Trained with cross-entropy loss adapted for highly imbalanced edge vs. non-edge pixels.  

# Results
- Achieved state-of-the-art edge detection on **BSDS500**, significantly surpassing Canny and Structured Edges.  
- Produced sharper edges with better object boundary alignment.  
- Also applied successfully to **depth edges (NYUDv2)**.  

# Why it Mattered
- First strong **deep CNN edge detector**, replacing decades of hand-crafted methods.  
- Introduced **deep supervision**, later used in segmentation and other vision tasks.  
- Became a standard baseline for edge/boundary detection research.  

# Architectural Pattern
- FCN backbone (based on VGG).  
- Multi-scale side outputs + fused prediction.  
- Deep supervision — influencing later models (e.g., deeply-supervised nets, RCF).  

# Connections
- **Contemporaries**: FCN for segmentation (Long et al., 2015).  
- **Influence**: RCF, DexiNed, modern boundary-aware segmentation methods.  

# Implementation Notes
- Requires class-balanced loss to address edge/non-edge imbalance.  
- Pre-training on ImageNet improves performance.  
- Fusion weights can be learned or fixed.  

# Critiques / Limitations
- Struggles with extremely thin structures or fine textures.  
- Computationally heavier than classical detectors.  
- Later works improved on fusion and feature richness (e.g., RCF).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1504.06375)  
- [Code (Caffe)](https://github.com/s9xie/hed)  
- [PyTorch reimplementation](https://github.com/xwjabc/hed-pytorch)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Convolutions and feature maps.  
- **Probability & Statistics**: Logistic loss for edge vs. non-edge.  
- **Optimization Basics**: Backpropagation with multi-loss supervision.  
- **Signals & Systems**: Multi-scale edge representations.  

## Postgraduate-Level Concepts
- **Numerical Methods**: Balancing class imbalance in pixel-wise losses.  
- **Neural Network Design**: Deep supervision, multi-scale fusion.  
- **Computer Vision**: Edge detection benchmarks (BSDS500).  
- **Research Methodology**: Comparative evaluation against hand-crafted methods.  

---

# My Notes
- Strong relevance to **mask boundary refinement** in my current projects.  
- Open question: Can **diffusion models** learn edge priors implicitly, removing need for explicit edge nets?  
- Possible extension: Fuse HED-like supervision with **video temporal edges** for object tracking.  
