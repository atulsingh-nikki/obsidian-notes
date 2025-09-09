---
title: "Spherical CNNs (2018)"
aliases: 
  - Spherical Convolutional Neural Networks
authors:
  - Taco Cohen
  - Mario Geiger
  - Jonas Köhler
  - Max Welling
year: 2018
venue: "ICLR"
doi: "10.48550/arXiv.1801.10130"
arxiv: "https://arxiv.org/abs/1801.10130"
code: "https://github.com/deepmind/spherecnn" # unofficial reimplementations
citations: 2000+
dataset:
  - ShapeNet
  - ModelNet40
  - Omnidirectional image datasets
tags:
  - paper
  - cnn
  - spherical-data
  - 3d
fields:
  - vision
  - geometry
  - robotics
related:
  - "[[Group Equivariant CNNs (2016)]]"
  - "[[PointNet (2017)]]"
predecessors:
  - "[[Group Equivariant CNNs (2016)]]"
successors:
  - "[[Gauge Equivariant CNNs (2019)]]"
  - "[[SO(3) Equivariant Networks]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
Spherical CNNs extended convolutional neural networks to functions defined on the **sphere (S²)**. They achieve **rotation equivariance under SO(3)**, making them suitable for 3D shape analysis, spherical signals, and omnidirectional vision.

# Key Idea
> Generalize convolutions from planar images to spherical signals by defining convolution operations that are **equivariant to 3D rotations**.

# Method
- Defines spherical convolution: correlation of spherical signals with spherical filters.  
- Ensures **rotation equivariance**: rotating the input rotates the output feature maps predictably.  
- Uses **spherical harmonics** to compute convolutions efficiently in spectral domain.  
- Applies to 3D objects (surface functions) and panoramic images.  

# Results
- Outperformed standard CNNs on 3D shape classification (ModelNet40).  
- Demonstrated robustness to arbitrary 3D rotations.  
- Showed competitive results on omnidirectional vision tasks.  

# Why it Mattered
- Enabled learning directly on **spherical domains** without distortions from projections.  
- Advanced the field of **geometric deep learning** (learning on non-Euclidean domains).  
- Important for robotics, astronomy, 3D vision, and omnidirectional cameras.  

# Architectural Pattern
- Spectral convolution using spherical harmonics.  
- SO(3) equivariant feature maps.  
- Layer stacking similar to CNNs but rotation-equivariant.  

# Connections
- **Contemporaries**: PointNet++, graph neural networks for meshes.  
- **Influence**: Gauge Equivariant CNNs, SE(3)-Transformers.  

# Implementation Notes
- Spherical harmonic transforms can be computationally heavy.  
- Requires band-limiting filters for efficiency.  
- Works best with clean spherical signals; irregular sampling introduces errors.  

# Critiques / Limitations
- Computationally expensive compared to planar CNNs.  
- Limited scalability for very high-resolution spherical data.  
- Replaced in practice by more flexible equivariant networks (Gauge CNNs, SE(3)-Transformers).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1801.10130)  
- [Unofficial PyTorch implementation](https://github.com/deepmind/spherecnn)  
- [Follow-up: Gauge Equivariant CNNs (2019)](https://arxiv.org/abs/1902.04615)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Basis expansions (spherical harmonics).  
- **Probability & Statistics**: Invariance/equivariance principles.  
- **Geometry**: Functions on manifolds (sphere).  

## Postgraduate-Level Concepts
- **Neural Network Design**: Equivariant architectures.  
- **Geometric Deep Learning**: Learning on non-Euclidean domains.  
- **Research Methodology**: Evaluating rotational robustness.  
- **Advanced Optimization**: Spectral convolution with harmonic transforms.  

---

# My Notes
- Relevant for **video stitching and panoramic video editing**.  
- Open question: Can spherical CNNs combine with **transformers** for global context on spherical domains?  
- Possible extension: Apply spherical equivariant models for **VR/AR 360° content editing**.  

---
