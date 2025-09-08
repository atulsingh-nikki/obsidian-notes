---
title: "Animatable NeRF: Deformable Neural Radiance Fields (2021)"
aliases:
  - Animatable NeRF
  - Deformable NeRF for Humans
authors:
  - Sida Peng
  - Yuanqing Zhang
  - Yinghao Xu
  - Qianqian Wang
  - Qing Shuai
  - Hujun Bao
  - Xiaowei Zhou
year: 2021
venue: "ICCV"
doi: "10.1109/ICCV48922.2021.01220"
arxiv: "https://arxiv.org/abs/2105.02872"
code: "https://github.com/zju3dv/animatable_nerf"
citations: 2000+
dataset:
  - ZJU-MoCap dataset
  - Captured multi-view human performance data
tags:
  - paper
  - nerf
  - 3d-reconstruction
  - human-performance
  - animatable
fields:
  - vision
  - graphics
  - neural-representations
related:
  - "[[NeRF (2020)]]"
  - "[[PIFuHD (2020)]]"
  - "[[Neural Body (2021)]]"
predecessors:
  - "[[PIFuHD (2020)]]"
  - "[[NeRF (2020)]]"
successors:
  - "[[Neural Body (2021)]]"
  - "[[HumanNeRF (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Animatable NeRF** extended NeRF to **clothed human modeling**, enabling novel view synthesis and animation of dynamic human bodies. It introduced **pose-dependent deformation fields** that map points between canonical (rest) space and posed space, allowing NeRF to generalize across motions.

# Key Idea
> Factorize human appearance into a **canonical space NeRF** + **pose-conditioned deformation field**, making radiance fields **animatable** for arbitrary poses.

# Method
- **Canonical NeRF**: Represents a human in rest pose.  
- **Deformation field**: Neural function maps query points from posed space to canonical space using pose info.  
- **Pose conditioning**: Uses SMPL body model skeleton for guidance.  
- **Rendering**: Volume rendering as in NeRF, but warped through canonical–posed mapping.  
- **Training**: Multi-view posed humans with known skeletal poses.  

# Results
- Enabled **novel view synthesis** of clothed humans in different poses.  
- Produced realistic geometry and appearance across a range of motions.  
- Outperformed baselines on ZJU-MoCap dataset.  

# Why it Mattered
- First NeRF variant tailored for **animatable human avatars**.  
- Bridged implicit radiance fields with parametric body models.  
- Foundational for follow-ups like Neural Body, HumanNeRF, and avatar-focused NeRFs.  

# Architectural Pattern
- Canonical NeRF backbone.  
- Pose-conditioned deformation MLP.  
- Skeleton-driven warping functions.  

# Connections
- Built on NeRF and inspired by PIFu/PIFuHD for clothed humans.  
- Predecessor to **Neural Body (2021)** (embedding SMPL features directly).  
- Related to **D-NeRF (2021)** but specialized for human motion.  

# Implementation Notes
- Needs skeleton pose input at training/inference.  
- Warping regularized to avoid artifacts.  
- Public PyTorch implementation with ZJU dataset.  

# Critiques / Limitations
- Struggles with loose clothing and fast motions.  
- Requires multi-view captures; monocular generalization weak.  
- Training computationally expensive.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Difference between **static NeRF** and **dynamic/animatable NeRFs**.  
- Skeleton-based pose conditioning (SMPL model).  
- Canonical vs posed space in human modeling.  
- Why volumetric rendering works for view synthesis.  

## Postgraduate-Level Concepts
- Neural deformation fields for canonical–posed mapping.  
- Regularization of deformation to maintain geometry consistency.  
- Trade-offs between NeRF implicit fields and explicit mesh-based avatars.  
- Extensions toward **real-time animatable human avatars**.  

---

# My Notes
- Animatable NeRF was the **first strong step toward digital avatars**.  
- Clever use of skeleton-guided deformation fields.  
- Open question: How to handle **loose garments and hair dynamics**?  
- Possible extension: Combine Animatable NeRF with **Gaussian Splatting** for real-time animatable humans.  

---
