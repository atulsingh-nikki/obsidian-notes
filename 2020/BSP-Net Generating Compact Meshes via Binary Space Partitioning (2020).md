---
title: "BSP-Net: Generating Compact Meshes via Binary Space Partitioning (2020)"
aliases:
  - BSP-Net
  - Binary Space Partitioning for Mesh Generation
authors:
  - Zhiqin Chen
  - Andrea Tagliasacchi
  - Hao Zhang
year: 2020
venue: "CVPR (Best Student Paper)"
doi: "10.1109/CVPR42600.2020.00982"
arxiv: "https://arxiv.org/abs/2005.03083"
code: "https://github.com/czq142857/BSP-NET"
citations: ~700+
dataset:
  - ShapeNet
  - ModelNet40
tags:
  - paper
  - 3d-reconstruction
  - mesh-generation
  - implicit-representations
  - neural-representations
fields:
  - vision
  - graphics
  - neural-3d
related:
  - "[[Occupancy Networks (2019)]]"
  - "[[DeepSDF (2019)]]"
  - "[[ConvONet (2020)]]"
predecessors:
  - "[[DeepSDF (2019)]]"
successors:
  - "[[MeshSDF / PolyGen Approaches]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**BSP-Net** introduced a novel framework for generating **compact, watertight polygonal meshes** directly from neural networks using **Binary Space Partitioning (BSP)**. Unlike prior implicit-field approaches that required expensive iso-surfacing (e.g., Marching Cubes), BSP-Net outputs explicit polygonal mesh structures natively.

# Key Idea
> Represent 3D geometry as a **set of half-spaces** whose intersections form watertight polygonal meshes, enabling direct, efficient mesh generation.

# Method
- **Binary Space Partitioning (BSP)**:  
  - 3D space recursively divided using learned hyperplanes.  
  - Intersections of half-spaces define convex polytopes.  
  - Polytopes combined to form final mesh.  
- **Neural prediction**: Network predicts BSP tree (planes + occupancy).  
- **Mesh generation**: Extract mesh surfaces directly, avoiding marching cubes.  
- **Training**: Supervised with ground-truth meshes from ShapeNet/ModelNet.  

# Results
- Produced **compact meshes** with fewer polygons but high fidelity.  
- Outperformed implicit-field methods (Occupancy Networks, DeepSDF) in mesh compactness and watertightness.  
- Showed efficiency gains: explicit meshes generated in seconds.  

# Why it Mattered
- First neural approach to directly generate **explicit polygonal meshes** via BSP.  
- Addressed key limitations of implicit-field methods (slow extraction, redundancy).  
- Paved the way for hybrid implicit-explicit 3D representations.  

# Architectural Pattern
- Neural net → hyperplanes (BSP tree).  
- Plane intersections → convex cells.  
- Merge → final polygonal mesh.  

# Connections
- Related to **DeepSDF** (implicit signed distance fields).  
- Successor approaches explored mesh generative models (e.g., PolyGen).  
- Connected to **computational geometry (BSP trees)**.  

# Implementation Notes
- Training requires mesh datasets (not just occupancy grids).  
- Output meshes are **watertight by construction**.  
- Open-source implementation available in PyTorch.  

# Critiques / Limitations
- Limited geometric expressivity for highly curved shapes.  
- Scalability issues for very complex topology (many hyperplanes needed).  
- Less flexible than NeRF-like continuous fields.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What is a mesh? Vertices, edges, faces.  
- Difference between implicit vs explicit 3D representations.  
- Basics of Binary Space Partitioning (BSP).  

## Postgraduate-Level Concepts
- Neural BSP tree construction.  
- Trade-offs: compactness vs expressivity.  
- Hybrid implicit-explicit methods in 3D deep learning.  

---

# My Notes
- BSP-Net is a clever **return to explicit geometry** in the NeRF/implicit era.  
- Great for applications where **compact watertight meshes** are required (CAD, graphics).  
- Open question: Can BSP-based representations integrate with **dynamic/deformable NeRFs**?  
- Possible extension: Use BSP as a **regularizer for NeRFs** to enforce mesh-like structure.  
