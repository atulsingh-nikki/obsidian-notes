---
title: Deep Image Matting (2017)
aliases:
  - DIM
  - Deep Image Matting
authors:
  - Ning Xu
  - Brian Price
  - Scott Cohen
  - Thomas Huang
year: 2017
venue: CVPR
doi: 10.1109/CVPR.2017.699
arxiv: https://arxiv.org/abs/1703.03872
code: https://github.com/foamliu/Deep-Image-Matting
citations: 3000+
dataset:
  - Adobe Image Matting Dataset (released with paper)
tags:
  - paper
  - matting
  - alpha-matte
  - segmentation
  - deep-learning
fields:
  - vision
  - graphics
  - image-editing
related:
  - "[[Real-Time High-Resolution Background Matting (2021)]]"
  - "[[MODNet Mobile Portrait Matting (2020)|MODNet (2020)]]"
  - "[[Robust Video Matting (RVM, 2021)|Robust Video Matting]]"
predecessors:
  - "[[Classical Trimap-based Matting]]"
successors:
  - "[[MODNet (2020)]]"
  - "[[Background Matting (2021)]]"
  - "[[RVM (2021)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**Deep Image Matting (DIM)** was the first deep learning approach to tackle the **image matting problem** — extracting precise foreground alpha mattes from natural images. It significantly outperformed classical sampling- and propagation-based matting methods.

# Key Idea
> Formulate matting as an **end-to-end learning problem**: given an input image + a user-provided **trimap**, a deep encoder–decoder network directly predicts the alpha matte.

# Method
- **Inputs**: RGB image + trimap (foreground, background, unknown regions).  
- **Network**: Encoder–decoder CNN predicts dense alpha matte.  
- **Losses**:  
  - Alpha prediction loss.  
  - Compositional loss: enforce consistency when recombining foreground + background.  
- **Dataset**: Introduced **Adobe Image Matting Dataset** with 431 unique foregrounds composited onto 50,000 backgrounds.  

# Results
- Outperformed classical matting methods on benchmarks.  
- Produced accurate alpha mattes, especially for hair and thin structures.  
- Generalized well to novel images.  

# Why it Mattered
- **First deep matting model** → sparked a wave of deep learning matting research.  
- Introduced **synthetic compositing dataset** still used for training matting models.  
- Influenced later real-time and background-free methods (MODNet, RVM).  

# Architectural Pattern
- Encoder–decoder CNN (U-Net style).  
- Supervision via compositional reconstruction.  

# Connections
- Predecessor to MODNet (2020), Real-Time Background Matting (2021), and RVM (2021).  
- Related to semantic segmentation but focused on fine-grained alpha transparency.  

# Implementation Notes
- Relies on user-provided trimaps (limiting automation).  
- Large synthetic dataset necessary for training.  
- Public dataset + baseline code spurred adoption.  

# Critiques / Limitations
- Requires trimaps → impractical for real-time use.  
- Training dataset synthetic → potential domain gap in natural images.  
- Not optimized for speed; heavy for real-time deployment.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Difference between segmentation (hard mask) and matting (soft alpha values).  
- Trimap: user annotation dividing known foreground, background, and uncertain regions.  
- Basics of encoder–decoder CNNs.  
- Compositing: combining foreground and background using alpha matte.  

## Postgraduate-Level Concepts
- Loss design for matting (alpha vs compositional consistency).  
- Synthetic dataset creation for matting supervision.  
- Comparison of deep vs classical matting pipelines.  
- Evolution toward trimap-free matting methods (MODNet, RVM).  

---

# My Notes
- DIM was the **ImageNet moment** for matting — deep learning took over.  
- Trimap reliance = strength (guidance) but also limitation (not user-friendly).  
- Open question: Can models achieve DIM-level quality **without trimaps**? (answered later by MODNet, RVM).  
- Possible extension: Train DIM-style networks with **diffusion priors** for even sharper fine structures.  

---
