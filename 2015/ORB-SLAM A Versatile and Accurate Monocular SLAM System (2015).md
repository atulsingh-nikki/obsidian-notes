---
title: "ORB-SLAM: A Versatile and Accurate Monocular SLAM System (2015)"
aliases:
  - ORB-SLAM
  - ORB-SLAM2
  - ORB-SLAM3
authors:
  - Raul Mur-Artal
  - J. M. M. Montiel
  - Juan D. Tardós
year: 2015 (ORB-SLAM), 2017 (ORB-SLAM2), 2020 (ORB-SLAM3)
venue: IEEE Transactions on Robotics (T-RO)
doi: 10.1109/TRO.2015.2463671
arxiv: https://arxiv.org/abs/1502.00956
citations: 15000+
tags:
  - paper
  - slam
  - visual-slam
  - robotics
  - computer-vision
fields:
  - robotics
  - computer-vision
  - ar-vr
related:
  - "[[Kimera Metric-Semantic Visual-Inertial SLAM (2019)|Kimera SLAM]]"
  - "[[GTSAM Georgia Tech Smoothing And Mapping Library (2010s)]]"
  - "[[iSAM Incremental Smoothing and Mapping (2007)|iSAM]]"
  - "[[GraphSLAM Simultaneous Localization and Mapping via Graph Optimization (2004)|GraphSLAM]]"
predecessors:
  - "[[PTAM (2007)]]"
successors:
  - "[[ORB-SLAM2 (2017)]]"
  - "[[ORB-SLAM3 (2020)]]"
  - "[[Kimera Metric-Semantic Visual-Inertial SLAM (2019)|Kimera SLAM]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**ORB-SLAM (Mur-Artal et al., 2015)** is a **feature-based visual SLAM system** that became the state of the art for monocular SLAM. It used **ORB features** for robust tracking, mapping, and loop closure, providing accurate and real-time localization and mapping.

# Key Idea
> Use a unified system based on **ORB features** for all SLAM tasks: tracking, mapping, relocalization, and loop closure, within a **graph optimization framework**.

# Method
- **Features**: ORB (Oriented FAST + Rotated BRIEF) descriptors.  
- **Tracking**: Pose estimation via feature matching and motion models.  
- **Mapping**: Keyframe-based sparse map representation.  
- **Loop Closure**: Bag-of-words place recognition + pose graph optimization.  
- **Back-end**: Graph optimization (pose graph + bundle adjustment).  

# Results
- Robust real-time monocular SLAM across diverse datasets.  
- Outperformed PTAM and other systems of the time.  
- Extended to **stereo/RGB-D (ORB-SLAM2, 2017)** and **multi-map, visual-inertial (ORB-SLAM3, 2020)**.  

# Why it Mattered
- Became the **default open-source SLAM system** in research and applications.  
- Widely used in robotics, AR/VR, and autonomous navigation.  
- Inspired later systems (Kimera, VINS-Mono, DSO).  

# Architectural Pattern
- Feature-based SLAM with **keyframes + graph optimization**.  
- Modular: tracking, local mapping, loop closing.  

# Connections
- Successor to **PTAM (2007)** (first real-time monocular SLAM).  
- Predecessor to **Kimera (2019)** (adds semantics) and **VINS-Mono (2018)** (visual-inertial).  
- Built on **graph-based back-ends** (g2o, GTSAM-like).  

# Implementation Notes
- Open-source C++ implementation available.  
- Requires good feature tracking in low-texture scenes.  
- Bundle adjustment = critical for accuracy.  

# Critiques / Limitations
- Sparse maps (not dense).  
- Purely geometric, no semantics.  
- Struggles in low-texture or dynamic environments.  

---

# Educational Connections

## Undergraduate-Level Concepts
- SLAM = robot/AR device builds map + localizes camera.  
- ORB-SLAM uses **features** (corners, descriptors) to track motion.  
- Example: AR headset building a sparse 3D map of a room.  

## Postgraduate-Level Concepts
- Pose graph optimization and bundle adjustment.  
- Place recognition with bag-of-words models.  
- Trade-offs: feature-based vs direct (photometric) SLAM.  
- Extensions: stereo/RGB-D (ORB-SLAM2), visual-inertial (ORB-SLAM3).  

---

# My Notes
- ORB-SLAM = **the workhorse of visual SLAM in the 2010s**.  
- Simplicity + robustness made it dominate academia and industry.  
- Open question: Will feature-based SLAM persist, or will learning-based methods take over fully?  
- Possible extension: Hybrid feature-based + deep descriptors, or semantic ORB-SLAM variants.  

---
