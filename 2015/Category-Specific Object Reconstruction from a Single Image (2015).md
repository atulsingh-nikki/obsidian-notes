---
title: "Category-Specific Object Reconstruction from a Single Image (2015)"
aliases: 
  - Category-Specific Reconstruction
  - Kar15 Reconstruction
authors:
  - Abhishek Kar
  - Shubham Tulsiani
  - Joao Carreira
  - Jitendra Malik
year: 2015
venue: "CVPR"
doi: "10.1109/CVPR.2015.7298697"
arxiv: "https://arxiv.org/abs/1506.06457"
code: "https://github.com/shubhtuls/drc"  # (related repo for later work; no official release for this paper)
citations: 1500+
dataset:
  - PASCAL VOC
  - ImageNet
tags:
  - paper
  - 3D-reconstruction
  - single-image
  - shape-prior
fields:
  - vision
  - 3D
related:
  - "[[DynamicFusion]]"
  - "[[Perspective Transformer Nets]]"
predecessors:
  - "[[Shape-from-Silhouette]]"
  - "[[Shape-from-Template]]"
successors:
  - "[[Perspective Transformer Nets]]"
  - "[[Neural Mesh Renderer]]"
  - "[[Category-Specific Mesh Reconstruction (Tulsiani et al.)]]"
impact: ⭐⭐⭐⭐☆
status: "to-read"
---


# Summary
This paper tackles the challenging problem of reconstructing **3D shape from a single image** by learning **category-specific shape priors**. Unlike classical shape-from-X methods, it leverages recognition signals from large annotated datasets to infer likely 3D structure given only one view.

# Key Idea
> Use category-level learned 3D priors to reconstruct plausible object shapes from a single 2D image.

# Method
- Assumes known object category and 2D keypoint annotations.  
- Learns **deformable 3D shape models** from annotated images across a category.  
- Uses **structure-from-motion techniques** on keypoints to induce category-level 3D shape bases.  
- At test time, fits the shape basis to new images by aligning detected keypoints.  
- Produces mesh-based 3D reconstructions, not just depth maps.  

# Results
- Demonstrated reconstructions on **PASCAL VOC** categories (cars, chairs, aeroplanes, etc.).  
- Qualitative 3D mesh recovery from single 2D inputs.  
- Showed ability to generalize across instances in the same category.  

# Why it Mattered
- Among the **first works** to combine recognition signals with geometry for **single-image 3D reconstruction**.  
- Shifted focus from instance-specific to **category-level modeling**, paving the way for deep learning–based methods.  
- Inspired later neural approaches (e.g., DRC, PTN, NMR) that dropped reliance on manual keypoints.  

# Architectural Pattern
- **Linear shape basis + deformation model**.  
- Optimization-based fitting (no end-to-end training).  
- Precursor to differentiable rendering approaches.  

# Connections
- **Contemporaries**: Volumetric methods (e.g., 3D ShapeNets, 2015).  
- **Influence**: Laid foundation for differentiable rendering (2017+) and deep category-specific models.  

# Implementation Notes
- Needs reliable 2D keypoint detectors for inference.  
- Strong supervision (keypoints, category labels) required.  
- Optimization is iterative, not real-time.  

# Critiques / Limitations
- Dependent on accurate keypoint annotations.  
- Limited scalability to arbitrary categories.  
- Reconstructions plausible but not always faithful to input image.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1506.06457)  
- [CVPR 2015 version](https://ieeexplore.ieee.org/document/7298697)  
- [Successor code: DRC (Tulsiani et al.)](https://github.com/shubhtuls/drc)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Basis decomposition for 3D shapes.  
- **Probability & Statistics**: Priors over shape categories.  
- **Optimization Basics**: Fitting shape bases to 2D constraints.  

## Postgraduate-Level Concepts
- **Numerical Methods**: Bundle adjustment & non-linear optimization.  
- **Computer Vision**: Keypoint detection, 3D alignment.  
- **Machine Learning Theory**: Category-level priors and generalization.  
- **Neural Network Design**: Later evolved into differentiable renderers.  

---

# My Notes
- Connects to current interests in **category-specific priors for generative models**.  
- Open question: How to bypass reliance on 2D keypoints while retaining structural accuracy?  
- Possible extension: Combine with **diffusion-based implicit priors** for richer reconstructions.  
