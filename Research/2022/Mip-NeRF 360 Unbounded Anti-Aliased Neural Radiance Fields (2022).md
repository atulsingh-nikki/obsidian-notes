---
title: "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields (2022)"
aliases:
  - Mip-NeRF 360
  - Unbounded Mip-NeRF
authors:
  - Jonathan T. Barron
  - Ben Mildenhall
  - Dor Verbin
  - Pratul P. Srinivasan
  - Peter Hedman
year: 2022
venue: "CVPR"
doi: "10.1109/CVPR52688.2022.00256"
arxiv: "https://arxiv.org/abs/2111.12077"
code: "https://github.com/google-research/mipnerf360"
citations: 1500+
dataset:
  - Real-world unbounded outdoor/indoor captures
tags:
  - paper
  - nerf
  - mip-nerf
  - 3d-reconstruction
  - view-synthesis
  - implicit-representations
fields:
  - vision
  - graphics
  - neural-representations
related:
  - "[[NeRF (2020)]]"
  - "[[Mip-NeRF (2021)]]"
  - "[[Instant-NGP (2022)]]"
predecessors:
  - "[[Mip-NeRF (2021)]]"
successors:
  - "[[Zip-NeRF (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Mip-NeRF 360** extended Mip-NeRF to handle **unbounded, real-world 360° scenes**. It introduced new parameterizations and sampling strategies to avoid artifacts in large-scale, outward-looking captures, while preserving anti-aliasing benefits from Mip-NeRF.

# Key Idea
> Enable NeRF to model **unbounded, real-world 360° environments** with anti-aliasing, using novel scene parameterizations and cone-based sampling.

# Method
- **Scene parameterization**:  
  - Compactified space representation to handle unbounded scenes (similar to inverse sphere parameterization).  
- **Conical frustums + IPE**: Retained Mip-NeRF’s anti-aliasing via integrated positional encodings.  
- **Sampling strategy**: Importance sampling adapted to unbounded geometries.  
- **Regularization**: Losses to encourage smoothness and reduce floating artifacts in large-scale reconstructions.  

# Results
- Produced **high-fidelity 360° renderings** of complex indoor and outdoor environments.  
- Outperformed NeRF and Mip-NeRF on real-world captures with wide baselines.  
- Eliminated aliasing and artifacts at scene boundaries.  

# Why it Mattered
- Took NeRF from **bounded toy scenes → real-world, unbounded 360° captures**.  
- Pushed neural radiance fields closer to practical use in AR/VR and immersive media.  
- Served as a key stepping stone to more efficient/generalizable NeRF variants.  

# Architectural Pattern
- Same MLP core as NeRF.  
- Cone-based frustums for scale-awareness.  
- Unbounded scene parameterization + regularization.  

# Connections
- Successor to **Mip-NeRF (2021)**.  
- Predecessor to **Zip-NeRF (2023)** and **Instant-NGP (2022)**.  
- Related to compactified coordinate mappings (sphere inversion).  

# Implementation Notes
- Training slower than Instant-NGP but higher quality.  
- Requires careful tuning of compactification radius.  
- Released Google Research code + pretrained examples.  

# Critiques / Limitations
- Still computationally heavy (hours/days training).  
- Not real-time; inference expensive for interactive use.  
- Limited generalization across scenes (scene-specific training).  

---

# Educational Connections

## Undergraduate-Level Concepts
- 360° capture basics.  
- Why unbounded space is harder than bounded scenes.  
- Aliasing in rendering.  

## Postgraduate-Level Concepts
- Coordinate compactification for unbounded domains.  
- Sampling strategies for neural volumetric rendering.  
- Regularization for stability in large-scale neural fields.  

---

# My Notes
- Mip-NeRF 360 **bridged the lab-to-world gap** for NeRFs.  
- Key insight: unbounded space needs different parameterization, not just more data.  
- Open question: How to make unbounded NeRFs **real-time** without quality loss?  
- Possible extension: Integrate Mip-NeRF 360 principles into **video NeRFs** for outdoor dynamic scenes.  

---
