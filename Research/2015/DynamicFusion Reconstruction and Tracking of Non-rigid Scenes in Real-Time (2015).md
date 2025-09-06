---
title: "DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time (2015)"
aliases:
  - DynamicFusion
authors:
  - Richard A. Newcombe
  - Dieter Fox
  - Steven M. Seitz
year: 2015
venue: CVPR
doi: 10.1109/CVPR.2015.7298866
arxiv: https://arxiv.org/abs/1504.06023
code: https://github.com/facebookresearch/DynamicFusion
citations: 6000+
dataset:
  - Custom real-time RGB-D sequences
tags:
  - paper
  - 3D-reconstruction
  - tracking
  - non-rigid
fields:
  - vision
  - graphics
related:
  - "[[KinectFusion]]"
  - "[[VolumeDeform]]"
predecessors:
  - "[[KinectFusion]]"
successors:
  - "[[VolumeDeform]]"
  - "[[DeepDeform]]"
impact: ⭐⭐⭐⭐⭐
status: to-read
---

# Summary
DynamicFusion introduces the first real-time system for reconstructing and tracking **non-rigidly deforming scenes** using a single depth camera. It extends KinectFusion (which assumed rigid scenes) to handle deformable objects like humans, clothing, or soft objects.  

# Key Idea
> Real-time dense 3D reconstruction of non-rigidly deforming surfaces using depth streams.

# Method
- Builds on a **volumetric TSDF representation**.  
- Introduces a **warp field** to align new depth observations with the canonical volume.  
- Optimizes the warp field using a **non-rigid Iterative Closest Point (ICP)** approach.  
- Incrementally fuses depth data into a canonical model.  
- Handles topology changes poorly but enables dynamic scene capture.  

# Results
- Demonstrated real-time performance (~30Hz) on commodity GPUs.  
- Captured human motion, facial deformations, and objects in motion.  
- Outperformed rigid reconstruction baselines by enabling non-rigid modeling.  

# Why it Mattered
- First system to achieve **real-time, dense non-rigid reconstruction**.  
- Sparked a wave of follow-ups in non-rigid tracking and reconstruction (VolumeDeform, DynamicFusion++, DeepDeform).  
- Major milestone for AR/VR, performance capture, and motion analysis.  

# Architectural Pattern
- **TSDF volumetric fusion** extended with a **warp field** for non-rigid alignment.  
- Non-rigid ICP optimization.  
- Influenced hybrid methods that combine deformation graphs and deep learning.  

# Connections
- **Contemporaries**: ElasticFusion (real-time dense SLAM), VolumeDeform (non-rigid deformation with color).  
- **Influence**: Inspired later learning-based systems (DeepDeform, Neural Volumes, NeuralBody).  

# Implementation Notes
- Requires a GPU for real-time optimization.  
- Sensitive to fast motions and occlusions.  
- Warp field regularization is key to stability.  

# Critiques / Limitations
- Cannot handle large topology changes (e.g., splitting surfaces).  
- Limited robustness to fast non-rigid motion.  
- High GPU demand (not feasible on mobile at the time).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1504.06023)  
- [Official CVPR 2015 version](https://ieeexplore.ieee.org/document/7298866)  
- [Code (unofficial implementations)](https://github.com/facebookresearch/DynamicFusion)  
- [Video Demo](https://www.youtube.com/watch?v=k2vGfo9HC3s)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Rigid and non-rigid transformations, ICP alignment.  
- **Probability & Statistics**: Residual minimization in ICP.  
- **Signals & Systems**: Depth data filtering, TSDF.  
- **Optimization Basics**: Gradient-based optimization of warp fields.  

## Postgraduate-Level Concepts
- **Numerical Methods**: Non-linear least squares for warp optimization.  
- **Computer Vision**: Depth-based tracking, SLAM.  
- **Neural Network Design**: Later successors replaced optimization with learned priors.  
- **Research Methodology**: Benchmarks with real RGB-D captures, ablation on warp regularization.  

---

# My Notes
- Connects to **my work on object tracking** and **dynamic scene understanding**.  
- Open question: How to integrate learned priors (optical flow, deformation graphs) for robustness?  
- Possible extension: Use diffusion models for **scene completion in dynamic contexts**.  
