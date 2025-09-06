---
title: "D-NeRF: Neural Radiance Fields for Dynamic Scenes (2021)"
aliases:
  - D-NeRF
  - Dynamic NeRF
authors:
  - Albert Pumarola
  - Enric Corona
  - Gerard Pons-Moll
  - Francesc Moreno-Noguer
year: 2021
venue: "CVPR"
doi: "10.1109/CVPR46437.2021.01335"
arxiv: "https://arxiv.org/abs/2011.13961"
code: "https://github.com/albertpumarola/D-NeRF"
citations: 3000+
dataset:
  - Synthetic dynamic scenes
  - Captured deformable object sequences
tags:
  - paper
  - nerf
  - dynamic-nerf
  - 3d-reconstruction
  - view-synthesis
fields:
  - vision
  - graphics
  - neural-representations
related:
  - "[[NeRF (2020)]]"
  - "[[NSFF: Neural Scene Flow Fields (2021)]]"
  - "[[HyperNeRF (2021)]]"
predecessors:
  - "[[NeRF (2020)]]"
successors:
  - "[[HyperNeRF (2021)]]"
  - "[[Dynamic Gaussian Splatting (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**D-NeRF** extended NeRF from static to **dynamic scenes**, enabling photorealistic novel view synthesis of deformable and moving objects. It modeled dynamics by learning a deformation field that maps points from a canonical static space to their time-dependent positions.

# Key Idea
> Factorize a dynamic scene into a **canonical NeRF** + a **deformation field** that warps points over time, allowing consistent 3D radiance representation for moving objects.

# Method
- **Canonical NeRF**: Represents the scene in a static "rest" space.  
- **Deformation field**: Neural function that maps canonical coordinates + time → deformed space coordinates.  
- **Volume rendering**: Same as NeRF, but with deformed coordinates per time step.  
- **Training**: Optimize both radiance field and deformation jointly using only multi-view video input.  

# Results
- Synthesized realistic novel views of deforming objects.  
- Worked on both synthetic and real captured sequences.  
- Outperformed baselines in dynamic 3D reconstruction and view synthesis.  

# Why it Mattered
- First major step beyond **static NeRF** to **dynamic/deformable NeRFs**.  
- Enabled applications in video-based 3D reconstruction and animation.  
- Paved way for later dynamic NeRFs (HyperNeRF, NSFF, Nerfies).  

# Architectural Pattern
- Canonical NeRF (static MLP).  
- Deformation MLP conditioned on time.  
- Combined with volumetric rendering.  

# Connections
- Complementary to **NSFF (Neural Scene Flow Fields, 2021)**, which learned scene flow directly.  
- Extended by **HyperNeRF (2021)** for topologically complex deformations.  
- Related to **Dynamic Gaussian Splatting (2023)** for efficiency.  

# Implementation Notes
- Training slower than static NeRF due to deformation field.  
- Deformation regularization needed for stability.  
- Public PyTorch implementation available.  

# Critiques / Limitations
- Struggles with large, topological changes (e.g., open/close motions).  
- High compute cost for video-length sequences.  
- Requires dense multi-view input for best results.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Static vs dynamic 3D scene representation.  
- Idea of canonical space + deformation.  
- Time as an input to neural networks.  

## Postgraduate-Level Concepts
- Canonical-to-deformed mapping via neural fields.  
- Regularization in dynamic neural representations.  
- Comparison with optical flow and scene flow.  

---

# My Notes
- D-NeRF is the **first bridge** from static NeRF to dynamic, deformable scenes.  
- Key insight: canonical + deformation field factorization.  
- Open question: How to handle **topological changes** without failure (solved partially by HyperNeRF)?  
- Possible extension: Use deformation fields as **temporal priors in video diffusion models**.  

---
