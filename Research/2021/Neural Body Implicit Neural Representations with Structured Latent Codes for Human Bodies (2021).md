---
title: "Neural Body: Implicit Neural Representations with Structured Latent Codes for Human Bodies (2021)"
aliases:
  - Neural Body
  - Human Performance NeRF
authors:
  - Sida Peng
  - Yuanqing Zhang
  - Yinghao Xu
  - Qianqian Wang
  - Qing Shuai
  - Hujun Bao
  - Xiaowei Zhou
year: 2021
venue: "CVPR (Oral)"
doi: "10.1109/CVPR46437.2021.00484"
arxiv: "https://arxiv.org/abs/2012.15838"
code: "https://github.com/zju3dv/neuralbody"
citations: 1500+
dataset:
  - ZJU-MoCap dataset
  - Multi-view human performance sequences
tags:
  - paper
  - nerf
  - human-performance
  - implicit-representations
  - animatable
fields:
  - vision
  - graphics
  - 3d-human
related:
  - "[[NeRF (2020)]]"
  - "[[Animatable NeRF (2021)]]"
  - "[[HumanNeRF (2022)]]"
predecessors:
  - "[[Animatable NeRF (2021)]]"
successors:
  - "[[HumanNeRF (2022)]]"
  - "[[Avatar Methods (2022+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Neural Body** improved on Animatable NeRF by embedding **structured latent codes on SMPL body vertices**, enabling better generalization across human motions and identities. It achieved high-quality **novel view synthesis** of clothed humans in motion using multi-view video.

# Key Idea
> Represent humans as a **set of latent codes anchored on a parametric skeleton (SMPL)**, which conditions NeRF’s radiance field, enabling structured and generalizable animatable reconstructions.

# Method
- **SMPL skeleton**: Provides parametric pose and shape prior.  
- **Latent codes**: Learned feature vectors attached to SMPL vertices.  
- **NeRF field**: Conditioned on interpolated latent codes + canonical coordinates.  
- **Rendering**: Standard NeRF volume rendering.  
- **Training**: Multi-view sequences of humans with known poses.  

# Results
- Outperformed Animatable NeRF in detail and consistency.  
- Captured dynamic clothing and surface detail more robustly.  
- Handled long performance sequences better.  

# Why it Mattered
- Introduced **structured latent codes** for human NeRFs, improving robustness and generalization.  
- Established a strong baseline for animatable human avatars.  
- Widely used in subsequent works (HumanNeRF, Neural Actor, avatar systems).  

# Architectural Pattern
- NeRF backbone.  
- Structured latent codes anchored on SMPL mesh.  
- Skeleton-driven conditioning.  

# Connections
- Successor to **Animatable NeRF (2021)**.  
- Predecessor to **HumanNeRF (2022)** and avatar-specific NeRFs.  
- Related to implicit field + skeleton hybrid approaches.  

# Implementation Notes
- Requires multi-view video for training.  
- Relies on accurate SMPL pose/shape estimation.  
- Code + ZJU-MoCap dataset released for benchmarks.  

# Critiques / Limitations
- Still struggles with loose garments and long hair.  
- Training/inference still computationally heavy.  
- Generalization to monocular input not solved.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What SMPL is: a **parametric human body model** for shape + pose.  
- Why conditioning NeRF on skeleton priors helps with human animation.  
- Difference between **unstructured latent codes** vs **structured latent codes**.  
- Basics of multi-view capture for training human models.  

## Postgraduate-Level Concepts
- Latent code anchoring on SMPL vertices as structured conditioning.  
- Hybrid models combining parametric skeletons with implicit NeRF fields.  
- Comparison of Animatable NeRF vs Neural Body in generalization.  
- Extensions toward free-viewpoint video and animatable avatars.  

---

# My Notes
- Neural Body is a **cleaner and stronger version of Animatable NeRF**.  
- Clever idea: structured latent codes anchored on SMPL → generalizable conditioning.  
- Open question: Can this structured approach scale to **monocular internet video**?  
- Possible extension: Fuse Neural Body priors with **Gaussian splatting avatars** for real-time animatable humans.  

---
