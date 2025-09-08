---
title: "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator (2018)"
aliases:
  - VINS-Mono
  - Visual-Inertial Odometry (VIO)
authors:
  - Tong Qin
  - Peiliang Li
  - Shaojie Shen
year: 2018
venue: "IEEE Transactions on Robotics (T-RO)"
doi: "10.1109/TRO.2018.2853729"
arxiv: "https://arxiv.org/abs/1708.03852"
citations: 8000+
tags:
  - paper
  - slam
  - visual-inertial
  - robotics
  - computer-vision
fields:
  - robotics
  - computer-vision
  - ar-vr
related:
  - "[[DSO (2016)]]"
  - "[[ORB-SLAM (2015)]]"
  - "[[Kimera (2019)]]"
predecessors:
  - "[[ORB-SLAM (2015)]]"
  - "[[DSO (2016)]]"
successors:
  - "[[Kimera (2019)]]"
  - "[[VINS-Fusion (2019)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**VINS-Mono (Qin, Li, Shen, 2018)** introduced a robust and versatile **monocular visual-inertial odometry (VIO) system**. It tightly couples a monocular camera with an IMU to provide drift-reduced pose estimates, enabling reliable operation in challenging environments.

# Key Idea
> Fuse **camera (visual)** and **IMU (inertial)** data in a tightly coupled **nonlinear optimization framework**, reducing drift and improving robustness compared to vision-only SLAM or VO.

# Method
- **Front-end**: Feature tracking using KLT and ORB features.  
- **Back-end**: Nonlinear optimization over sliding window of poses, landmarks, and IMU states.  
- **Loop closure**: Pose graph optimization to reduce drift in long trajectories.  
- **Key outputs**: Pose, velocity, biases, and 3D map points.  

# Results
- High-accuracy localization across indoor/outdoor datasets.  
- Real-time performance on CPU (suitable for drones/embedded systems).  
- Became the go-to open-source VIO system.  

# Why it Mattered
- First widely adopted **open-source VIO framework**.  
- Demonstrated practicality for robotics, AR/VR, and UAV navigation.  
- Foundation for later systems (VINS-Fusion, Kimera-VIO).  

# Architectural Pattern
- Tightly coupled visual-inertial optimization.  
- Sliding-window nonlinear least squares.  
- Loop closure for long-term drift correction.  

# Connections
- Builds on lessons from **ORB-SLAM (feature-based)** and **DSO (direct optimization)**.  
- Predecessor to **VINS-Fusion (2019)** (multi-sensor fusion).  
- Basis of modern VIO in robotics/AR.  

# Implementation Notes
- Open-source C++ implementation widely used.  
- Requires IMU calibration.  
- Sensitive to feature-tracking failures in texture-poor scenes.  

# Critiques / Limitations
- Sparse map (not dense).  
- Loop closure less robust than ORB-SLAM.  
- Struggles under very fast motion if visual tracking fails.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Camera = sees environment; IMU = senses motion.  
- Combining them reduces drift compared to vision-only odometry.  
- Example: drone navigating outdoors where GPS is unavailable.  

## Postgraduate-Level Concepts
- Tightly coupled nonlinear optimization.  
- State vector includes poses, velocities, biases.  
- Loop closure integration in VIO.  
- Extensions to multi-sensor fusion (stereo, LiDAR, GPS).  

---

# My Notes
- VINS-Mono = **the ORB-SLAM of VIO**: robust, versatile, widely adopted.  
- Cemented tightly coupled optimization as the standard for VIO.  
- Open question: Can VINS-style optimization be integrated with deep learned features for robustness in dynamic environments?  
- Possible extension: VINS + transformers for cross-modal feature fusion.  

---
