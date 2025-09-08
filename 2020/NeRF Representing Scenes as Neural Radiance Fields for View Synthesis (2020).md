---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (2020)"
aliases:
  - NeRF
  - Neural Radiance Fields
authors:
  - Ben Mildenhall
  - Pratul Srinivasan
  - Matthew Tancik
  - Jonathan T. Barron
  - Ravi Ramamoorthi
  - Ren Ng
year: 2020
venue: "ECCV (Honorable Mention)"
doi: "10.1007/978-3-030-58452-8_24"
arxiv: "https://arxiv.org/abs/2003.08934"
code: "https://github.com/bmild/nerf"
citations: 20,000+
dataset:
  - Synthetic NeRF dataset (Blender scenes)
  - Real captured multi-view datasets
tags:
  - paper
  - nerf
  - 3d-reconstruction
  - view-synthesis
  - implicit-representations
fields:
  - vision
  - graphics
  - neural-representations
related:
  - "[[Multiview Stereo (classical)]]"
  - "[[Mip-NeRF (2021)]]"
  - "[[Instant-NGP (2022)]]"
predecessors:
  - "[[Volumetric Rendering Techniques]]"
successors:
  - "[[Mip-NeRF (2021)]]"
  - "[[Plenoxels (2021)]]"
  - "[[Instant-NGP (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**NeRF** introduced a new paradigm for **view synthesis and 3D scene representation**: represent a scene as a **continuous neural radiance field**, parameterized by a multilayer perceptron (MLP). Given camera poses and images, NeRF learns to map 3D coordinates + viewing directions → emitted color and volume density, enabling photorealistic novel view rendering.

# Key Idea
> Represent a scene as a continuous 5D function (x, y, z, θ, φ) → (color, density), learned by a neural network, and render views using differentiable volumetric rendering.

# Method
- **Neural representation**: MLP takes 3D location + 2D viewing direction as input, outputs RGB color + volume density.  
- **Volumetric rendering**: Render pixels by integrating color/density along camera rays through the field.  
- **Positional encoding**: High-frequency Fourier features applied to inputs to enable the MLP to model fine detail.  
- **Training**: Optimize MLP parameters to minimize photometric error across training images.  

# Results
- Produced **photorealistic novel views** of synthetic and real scenes.  
- Far exceeded traditional view synthesis and 3D reconstruction baselines.  
- Represented a **new class of neural implicit 3D representations**.  

# Why it Mattered
- Sparked the **NeRF revolution**: implicit neural representations for 3D scenes.  
- Unified graphics and vision with differentiable rendering.  
- Laid foundations for 1000+ follow-up works: fast NeRF, dynamic NeRF, generative NeRFs.  

# Architectural Pattern
- MLP with positional encoding.  
- Volumetric rendering integral for supervision.  

# Connections
- Predecessors: volumetric rendering, light field rendering.  
- Successors: Mip-NeRF, NeRF++, Instant-NGP, dynamic NeRFs.  
- Connected to implicit representations in graphics (SDFs, occupancy networks).  

# Implementation Notes
- Training is slow (hours to days per scene).  
- Inference also slow (ray marching per pixel).  
- Later methods focused on efficiency (hash grids, acceleration structures).  

# Critiques / Limitations
- Limited to static scenes.  
- High compute/memory cost.  
- Requires accurate camera poses.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Basics of 3D rendering (rays, colors, integration).  
- How neural networks can approximate functions.  
- Difference between explicit (meshes, voxels) and implicit (fields) representations.  

## Postgraduate-Level Concepts
- Differentiable volumetric rendering.  
- Positional encoding as Fourier feature mapping.  
- Implicit neural representations vs explicit 3D models.  

---

# My Notes
- NeRF is a **paradigm shift**: turned MLPs into continuous scene representations.  
- Inspired entire subfield of **neural rendering**.  
- Open question: How to extend NeRF to **dynamic, unposed, or sparse-view data**?  
- Possible extension: Use NeRF-like fields as **priors for video diffusion** or **scene editing models**.  

---
