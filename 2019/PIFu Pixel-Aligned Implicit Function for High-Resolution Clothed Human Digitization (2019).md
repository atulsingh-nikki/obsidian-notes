---
title: "PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization (2019)"
aliases:
  - PIFu
  - Pixel-Aligned Implicit Function
authors:
  - Shunsuke Saito
  - Zeng Huang
  - Ryota Natsume
  - Shigeo Morishima
  - Angjoo Kanazawa
  - Hao Li
year: 2019
venue: "ICCV"
doi: "10.1109/ICCV.2019.00443"
arxiv: "https://arxiv.org/abs/1905.05172"
code: "https://github.com/shunsukesaito/PIFu"
citations: 2000+
dataset:
  - RenderPeople
  - Custom in-the-wild images
tags:
  - paper
  - 3d-reconstruction
  - human-performance
  - implicit-representations
  - monocular
fields:
  - vision
  - graphics
  - 3d-human
related:
  - "[[DeepCap (2020)]]"
  - "[[Learning High Fidelity Depths of Dressed Humans (2021)]]"
  - "[[PIFuHD (2020)]]"
predecessors:
  - "[[Voxel/mesh-based 3D human models (pre-2019)]]"
successors:
  - "[[PIFuHD (2020)]]"
  - "[[Animatable NeRF (2021)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**PIFu** introduced a **pixel-aligned implicit function** for high-resolution 3D human reconstruction from a single image. It achieved detailed clothed human digitization by learning an implicit occupancy function conditioned on per-pixel image features, bridging 2D image space with 3D geometry.

# Key Idea
> Condition implicit 3D occupancy functions on **per-pixel image features**, enabling detailed 3D reconstruction of clothed humans from monocular RGB input.

# Method
- **Input**: A single RGB image of a clothed human.  
- **Pixel-aligned features**: Extract CNN features at each pixel.  
- **Implicit function**: For a 3D query point, project it into image space, sample pixel features, and predict occupancy (inside/outside surface).  
- **Surface extraction**: Use Marching Cubes to obtain a watertight mesh.  
- **Training**: Supervised with 3D scans of clothed humans.  

# Results
- Produced **high-resolution 3D human reconstructions** with clothing detail.  
- Outperformed voxel- and mesh-based baselines in fidelity.  
- Enabled digitization of humans from casual photos.  

# Why it Mattered
- One of the first **implicit function methods for clothed humans**.  
- Showed the power of **pixel-aligned implicit conditioning** for monocular 3D tasks.  
- Widely influential, leading to PIFuHD and animatable implicit models.  

# Architectural Pattern
- CNN encoder for image features.  
- Pixel-aligned implicit MLP for occupancy prediction.  
- Mesh extraction via Marching Cubes.  

# Connections
- Successor to voxel/mesh-based reconstructions.  
- Predecessor to **PIFuHD (higher resolution)** and **NeRF-based human models**.  
- Related to implicit neural representations like DeepSDF.  

# Implementation Notes
- Requires calibrated images (single-view).  
- Surface quality depends on Marching Cubes resolution.  
- Code and pretrained models released.  

# Critiques / Limitations
- Struggles with occlusion and unseen back views.  
- Relies on supervised training with 3D scans.  
- Lacks temporal consistency (single image only).  

---

# Educational Connections

## Undergraduate-Level Concepts
- Basics of **3D representation formats**: voxels, meshes, implicit functions.  
- How projecting a 3D point into 2D image space can condition predictions.  
- What occupancy means in geometry (inside vs outside).  
- Marching Cubes: converting implicit fields to meshes.  

## Postgraduate-Level Concepts
- Pixel-aligned conditioning as a bridge between CNN image features and implicit fields.  
- Occupancy networks vs signed distance functions (SDFs).  
- Advantages of implicit neural fields over mesh parameterizations.  
- Extension of PIFu to temporal/video data (PIFuHD, animatable models).  

---

# My Notes
- PIFu was the **breakthrough for clothed human digitization**: high-res detail from just one image.  
- Inspired a wave of **implicit human modeling** (PIFuHD, Neural Body, Animatable NeRF).  
- Open question: How to solve occlusion and back-view hallucination robustly?  
- Possible extension: Combine PIFu-style pixel alignment with **NeRFs or Gaussian splatting** for temporally consistent 3D humans.  

---
