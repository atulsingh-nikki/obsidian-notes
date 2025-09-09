---
title: "3D Gaussian Splatting for Real-Time Radiance Field Rendering (2023)"
aliases:
  - 3D Gaussian Splatting
  - Gaussian Splatting
authors:
  - Thomas Kerbl
  - Georgios Kopanas
  - Thomas Leimkühler
  - Markus Steinberger
year: 2023
venue: "SIGGRAPH"
doi: "10.1145/3588432.3591490"
arxiv: "https://arxiv.org/abs/2308.04079"
code: "https://github.com/graphdeco-inria/gaussian-splatting"
citations: 2000+
dataset:
  - Real-world captures
  - Synthetic NeRF datasets
tags:
  - paper
  - gaussian-splatting
  - radiance-fields
  - real-time
  - 3d-reconstruction
fields:
  - vision
  - graphics
  - neural-representations
related:
  - "[[NeRF (2020)]]"
  - "[[Instant-NGP (2022)]]"
  - "[[Zip-NeRF (2023)]]"
predecessors:
  - "[[Zip-NeRF (2023)]]"
successors:
  - "[[Gaussian Splatting Variants (2024+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**3D Gaussian Splatting** replaced implicit NeRF MLPs with an **explicit set of 3D Gaussians** to represent scenes, enabling **real-time rendering** of radiance fields. Each Gaussian stores position, covariance (size/shape), color, and opacity. Rendering is done via efficient rasterization of these splats, achieving photorealism at interactive framerates.

# Key Idea
> Represent a scene as a cloud of **learned 3D Gaussians** that directly splat onto the image plane, bypassing costly ray-marching through an MLP.

# Method
- **Initialization**: Start from structure-from-motion point cloud.  
- **Gaussians**: Each point represented as a Gaussian with learnable:  
  - 3D position  
  - Covariance (ellipsoidal extent)  
  - Color/radiance features  
  - Opacity  
- **Differentiable rasterization**: Project Gaussians into screen space and blend them efficiently.  
- **Optimization**: Gradient descent refines Gaussian parameters to match training views.  

# Results
- Training: Minutes to reconstruct a scene.  
- Rendering: Real-time (30–100+ FPS) on GPUs.  
- Quality: Comparable or superior to NeRF/Zip-NeRF, especially for large unbounded scenes.  
- Memory: Larger than NeRF (stores explicit Gaussians), but far faster.  

# Why it Mattered
- First **real-time NeRF alternative** with state-of-the-art quality.  
- Shifted field from **implicit neural fields → explicit point-based representations**.  
- Sparked rapid adoption in AR/VR, 3D mapping, and industry pipelines.  

# Architectural Pattern
- Explicit set of 3D Gaussians.  
- Differentiable rasterizer (splat-based).  
- Direct rendering, no MLP inference per sample.  

# Connections
- Successor to **Zip-NeRF, Instant-NGP**.  
- Predecessor to **Gaussian splatting variants (e.g., 4D dynamic Gaussians, Gaussian surfels, Gaussian avatars)**.  
- Related to point-based graphics and surfel rendering.  

# Implementation Notes
- Requires SFM or COLMAP initialization.  
- Training involves optimizing millions of Gaussians.  
- Rasterizer optimized for GPUs.  
- Open-source implementation widely adopted.  

# Critiques / Limitations
- High memory usage (millions of Gaussians).  
- Not naturally compact like implicit MLP NeRFs.  
- Editing Gaussians harder than editing parametric fields.  
- Struggles with view extrapolation beyond training poses.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Point-based graphics vs mesh/voxel rendering.  
- Why explicit vs implicit representations differ in memory and speed.  
- Basics of rasterization and splatting.  

## Postgraduate-Level Concepts
- Differentiable rendering of Gaussian ellipsoids.  
- Trade-offs between explicit and implicit neural fields.  
- Extensions to dynamic and generative 3D models.  

---

# My Notes
- Gaussian splatting is the **“practical breakthrough”**: NeRF quality + real-time rendering.  
- It feels like the **transformer moment for neural fields** — a new default paradigm.  
- Open question: How to compress Gaussian splats for **mobile/edge deployment**?  
- Possible extension: Combine Gaussian splats with **diffusion models** for interactive scene editing.  

---
