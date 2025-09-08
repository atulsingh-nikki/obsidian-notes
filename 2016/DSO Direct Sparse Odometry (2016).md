---
title: "DSO: Direct Sparse Odometry (2016)"
aliases:
  - DSO
  - Direct Sparse Odometry
authors:
  - Jakob Engel
  - Vladlen Koltun
  - Daniel Cremers
year: 2016
venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)"
doi: "10.1109/TPAMI.2017.2658577"
arxiv: "https://arxiv.org/abs/1607.02565"
citations: 5000+
tags:
  - paper
  - slam
  - visual-odometry
  - direct-method
  - robotics
fields:
  - robotics
  - computer-vision
  - ar-vr
related:
  - "[[LSD-SLAM (2014)]]"
  - "[[ORB-SLAM (2015)]]"
  - "[[VINS-Mono (2018)]]"
predecessors:
  - "[[LSD-SLAM (2014)]]"
successors:
  - "[[VINS-Mono (2018)]]"
  - "[[Deep Direct VO (2020s)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**DSO (Engel, Koltun, Cremers, 2016)** refined direct SLAM into a highly accurate **visual odometry system**. It used **photometric bundle adjustment** over a sliding window of keyframes with a carefully chosen sparse set of pixels, achieving robust and precise odometry without reliance on feature descriptors.

# Key Idea
> Optimize a **sparse photometric error** directly over poses and depths, using only high-gradient pixels, within a sliding window framework. This combines efficiency with accuracy.

# Method
- **Sparse direct alignment**: Only high-gradient pixels used.  
- **Photometric bundle adjustment**: Jointly optimizes camera poses and pixel depths in sliding window.  
- **Full intrinsic + photometric calibration**: Accounts for exposure time, vignetting, camera response.  
- **Windowed optimization**: Limits computation while keeping accuracy.  

# Results
- Outperformed feature-based methods in odometry accuracy.  
- Robust to low-texture environments compared to ORB-SLAM.  
- Provided precise VO but not full loop-closure SLAM.  

# Why it Mattered
- Set a **new standard for visual odometry** accuracy.  
- Showed the power of photometric bundle adjustment.  
- Still widely used as a baseline in VO/SLAM research.  

# Architectural Pattern
- Direct sparse VO.  
- Sliding window bundle adjustment.  

# Connections
- Successor to **LSD-SLAM (2014)**.  
- Complementary to **ORB-SLAM (2015)** (features + loop closure).  
- Predecessor to **VINS-Mono (2018)** and deep direct VO.  

# Implementation Notes
- Requires precise photometric calibration.  
- Sensitive to lighting changes if calibration poor.  
- Open-source C++ implementation available.  

# Critiques / Limitations
- Pure odometry (no loop closures).  
- Limited global consistency.  
- Calibration-heavy compared to feature-based methods.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Odometry = tracking motion step by step.  
- DSO tracks motion by directly comparing pixel intensities (not corners).  
- Example: drone navigation where features are sparse.  

## Postgraduate-Level Concepts
- Photometric error optimization with sliding windows.  
- Joint optimization of poses + depths (bundle adjustment).  
- Calibration importance (camera intrinsics + photometric response).  
- Trade-offs: direct vs feature-based vs hybrid VO/SLAM.  

---

# My Notes
- DSO = **direct VO done right**: accuracy rivaling features, but simpler pipeline.  
- Showed direct methods can match or beat ORB-SLAM in odometry.  
- Open question: Can deep networks replace hand-crafted photometric models while keeping DSO’s efficiency?  
- Possible extension: Neural-DSO with learned pixel selection + photometric models.  

---
