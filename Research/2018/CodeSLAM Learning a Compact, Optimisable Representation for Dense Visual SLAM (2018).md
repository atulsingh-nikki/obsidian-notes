---
title: "CodeSLAM: Learning a Compact, Optimisable Representation for Dense Visual SLAM (2018)"
aliases: 
  - CodeSLAM
  - Compact Representations for SLAM
authors:
  - Michael Bloesch
  - Jan Czarnowski
  - Ronald Clark
  - Stefan Leutenegger
  - Andrew J. Davison
year: 2018
venue: "CVPR"
doi: "10.1109/CVPR.2018.00424"
arxiv: "https://arxiv.org/abs/1804.00874"
code: "https://github.com/m-bloesch/CodeSLAM"  # unofficial community versions exist
citations: 800+
dataset:
  - ICL-NUIM
  - TUM RGB-D
  - Synthetic training data
tags:
  - paper
  - slam
  - 3d
  - representation-learning
fields:
  - vision
  - robotics
  - mapping
related:
  - "[[DeepVO (2017)]]"
  - "[[ORB-SLAM (2015)]]"
predecessors:
  - "[[Dense SLAM systems]]"
successors:
  - "[[DeepFactors (2019)]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
CodeSLAM introduced a **learned compact latent code** to represent dense scene geometry for **visual SLAM**. Instead of storing per-pixel depth, it learns a low-dimensional code optimisable during SLAM, enabling **joint depth prediction and geometric consistency** in mapping.

# Key Idea
> Use a neural network to predict a compact **latent code** for scene geometry, which can be optimised jointly with camera poses during SLAM.

# Method
- **Encoder–decoder depth network**:  
  - Input: monocular image.  
  - Output: depth map represented via a latent code.  
- **Latent code**: Low-dimensional, compact representation capturing geometry.  
- **Optimisation**: During SLAM, both latent codes and camera poses are jointly optimised for consistency.  
- **Training**: Autoencoder trained with supervised depth and photometric losses.  

# Results
- Showed dense depth maps can be represented with small latent codes.  
- Achieved competitive mapping quality compared to dense SLAM systems.  
- Reduced storage and enabled efficient optimisation.  

# Why it Mattered
- First demonstration that **deep learned representations can be integrated into optimisation-based SLAM pipelines**.  
- Bridged the gap between **deep learning** and **classical geometric SLAM**.  
- Influenced later hybrid SLAM systems (DeepFactors, NICE-SLAM).  

# Architectural Pattern
- Autoencoder for depth with latent code.  
- Joint bundle adjustment over poses + codes.  
- Dense but compact mapping.  

# Connections
- **Contemporaries**: DeepVO, CNN-SLAM.  
- **Influence**: DeepFactors (2019), NICE-SLAM (2021).  

# Implementation Notes
- Code dimension needs tuning: too small = underfitting, too large = inefficient.  
- Requires initial supervised depth training.  
- Integration with SLAM backends can be complex.  

# Critiques / Limitations
- Relies on depth supervision or synthetic data.  
- Latent representation sometimes struggles with unseen geometries.  
- Not as accurate as classical SLAM in high-texture regions.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1804.00874)  
- [Unofficial PyTorch implementation](https://github.com/mbloesch/CodeSLAM)  
- [Related work: DeepFactors (2019)](https://arxiv.org/abs/1904.06789)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Latent vector encoding.  
- **Probability & Statistics**: Photometric consistency losses.  
- **Geometry**: Camera poses, depth maps.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Autoencoder for compact geometry representation.  
- **Computer Vision**: Integration of deep learning into SLAM.  
- **Research Methodology**: Hybrid evaluation vs classical baselines.  
- **Advanced Optimization**: Joint optimisation of codes + poses.  

---

# My Notes
- Strong step toward **learned representations for 3D video editing**.  
- Open question: Can **diffusion priors on codes** improve generalisation in CodeSLAM-like systems?  
- Possible extension: Adapt CodeSLAM codes for **video object-level mapping** in editing workflows.  

---
