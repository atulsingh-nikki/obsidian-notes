---
title: "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields (2021)"
aliases:
  - Mip-NeRF
  - Multiscale NeRF
authors:
  - Jonathan T. Barron
  - Ben Mildenhall
  - Matthew Tancik
  - Peter Hedman
  - Ricardo Martin-Brualla
  - Pratul Srinivasan
year: 2021
venue: "ICCV"
doi: "10.1109/ICCV48922.2021.01471"
arxiv: "https://arxiv.org/abs/2103.13415"
code: "https://github.com/google/mipnerf"
citations: 2500+
dataset:
  - Synthetic NeRF dataset (Blender scenes)
  - Real captured multi-view datasets
tags:
  - paper
  - nerf
  - 3d-reconstruction
  - view-synthesis
  - implicit-representations
  - anti-aliasing
fields:
  - vision
  - graphics
  - neural-representations
related:
  - "[[NeRF (2020)]]"
  - "[[Mip-NeRF 360 (2022)]]"
  - "[[Instant-NGP (2022)]]"
predecessors:
  - "[[NeRF (2020)]]"
successors:
  - "[[Mip-NeRF 360 (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Mip-NeRF** addressed one of NeRF’s biggest shortcomings: **aliasing artifacts** when rendering at different scales. It introduced a **multiscale conical frustum representation** instead of simple rays, allowing NeRF to produce anti-aliased renderings when zooming in/out or at varying distances.

# Key Idea
> Replace rays with **cone-shaped frustums** and encode them with **integrated positional encodings (IPEs)**, so the MLP learns a scale-aware radiance field that naturally avoids aliasing.

# Method
- **Conical frustums**: Each ray sample represented as a frustum with finite width (not an infinitesimal ray).  
- **Integrated positional encoding (IPE)**: Compute expectations of Fourier features over frustums, yielding smooth, scale-aware embeddings.  
- **Network**: Same NeRF-style MLP, but inputs are IPEs instead of raw Fourier features.  
- **Rendering**: Volume rendering across frustums instead of points.  

# Results
- Eliminated aliasing in zoomed-out or fine-detail renderings.  
- Higher fidelity results on both synthetic and real datasets.  
- More efficient training due to better sample representations.  

# Why it Mattered
- Solved NeRF’s aliasing problem, a key step toward practical use.  
- Introduced a general principle: **scale-aware encoding** for neural fields.  
- Opened door to Mip-NeRF 360, fast NeRFs, and hybrid scene representations.  

# Architectural Pattern
- Input: 3D position + direction + scale → IPE.  
- MLP → outputs density + color.  
- Volumetric rendering with cones/frustums.  

# Connections
- Successor to **NeRF (2020)**.  
- Predecessor to **Mip-NeRF 360 (2022)** and **FastNeRF variants**.  
- Related to Fourier positional encodings and scale-space theory.  

# Implementation Notes
- Requires computing expectations of Fourier features analytically.  
- Training/inference overhead is modest compared to vanilla NeRF.  
- Public code released by Google Research.  

# Critiques / Limitations
- Still relatively slow to train (hours per scene).  
- Focused only on anti-aliasing, not on generalizing across large scenes.  
- Struggles with very sparse views.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Aliasing in rendering and signal processing.  
- Why scale-awareness is important in graphics.  
- Basics of positional encoding.  

## Postgraduate-Level Concepts
- Integrated positional encoding mathematics.  
- Cone vs ray integration in volume rendering.  
- Scale-space representations in neural fields.  

---

# My Notes
- Mip-NeRF made NeRF **practically usable for multi-scale rendering**.  
- Shows how subtle mathematical tweaks (IPEs) solve major perceptual issues.  
- Open question: Can IPE-style embeddings help in **video diffusion models**, where aliasing across scales is a problem?  
- Possible extension: Scale-aware embeddings for **video editing tasks** that require zoom/pan consistency.  

---
