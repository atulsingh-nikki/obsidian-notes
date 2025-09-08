---
title: "PTAM: Parallel Tracking and Mapping for Small AR Workspaces (2007)"
aliases:
  - PTAM
  - Parallel Tracking and Mapping
authors:
  - Georg Klein
  - David Murray
year: 2007
venue: "ISMAR (IEEE International Symposium on Mixed and Augmented Reality)"
doi: "10.1109/ISMAR.2007.4538852"
citations: 7000+
tags:
  - paper
  - slam
  - visual-slam
  - ar
  - robotics
fields:
  - computer-vision
  - robotics
  - ar-vr
related:
  - "[[GraphSLAM (2004)]]"
  - "[[ORB-SLAM (2015)]]"
  - "[[Visual-Inertial SLAM (2010s)]]"
predecessors:
  - "[[Visual Odometry (1980s–2000s)]]"
successors:
  - "[[ORB-SLAM (2015)]]"
  - "[[DSO (2016)]]"
  - "[[VINS-Mono (2018)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**PTAM (Klein & Murray, 2007)** introduced **Parallel Tracking and Mapping**, the first system to achieve **real-time monocular SLAM** on a standard CPU. It split SLAM into **two parallel threads**: tracking (camera pose estimation) and mapping (map optimization), enabling real-time performance for augmented reality.

# Key Idea
> Decouple SLAM into two threads — one for **fast tracking** of the camera, another for **map building and optimization** in the background — making monocular SLAM feasible in real time.

# Method
- **Tracking thread**:  
  - Estimates camera pose frame-to-frame using keypoints.  
  - Fast enough for interactive AR.  
- **Mapping thread**:  
  - Builds and optimizes a sparse 3D map with bundle adjustment.  
  - Runs asynchronously from tracking.  
- **Keyframe-based approach**: Sparse set of frames chosen for mapping, not all frames.  

# Results
- First **real-time monocular SLAM system** on a CPU.  
- Demonstrated live AR overlays (milestone for AR).  
- Proved feasibility of monocular SLAM in practice.  

# Why it Mattered
- **Paradigm shift**: showed monocular SLAM was possible in real time.  
- Inspired almost all later SLAM systems (ORB-SLAM, LSD-SLAM, DSO).  
- Hugely influential in **AR, robotics, and computer vision**.  

# Architectural Pattern
- Two-threaded SLAM (tracking + mapping).  
- Keyframe-based optimization.  

# Connections
- Successor to **visual odometry** research.  
- Predecessor to **ORB-SLAM (2015)** and modern VSLAM systems.  
- Related to **graph-based optimization (bundle adjustment)**.  

# Implementation Notes
- Sparse feature-based maps (suitable for small AR workspaces).  
- CPU-only implementation in 2007 → major achievement.  
- Limited scalability compared to later SLAM systems.  

# Critiques / Limitations
- Restricted to **small-scale AR workspaces**.  
- Sparse mapping only (not dense).  
- No loop closure detection → drift accumulates.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Visual SLAM lets a camera localize itself without GPS.  
- PTAM tracked features for pose, built a sparse map in parallel.  
- Example: AR demo where virtual objects stay fixed in a small room.  

## Postgraduate-Level Concepts
- Bundle adjustment in background mapping.  
- Importance of decoupling tracking vs mapping for real-time SLAM.  
- Keyframe-based SLAM pipeline (now standard).  
- Contrast with modern systems (ORB-SLAM adds loop closure, scalability).  

---

# My Notes
- PTAM = **the prototype of modern SLAM**.  
- Its AR demo sparked interest in practical SLAM for consumer devices.  
- Open question: Could we revisit PTAM’s simplicity with today’s hardware for ultra-lightweight AR systems?  
- Possible extension: PTAM + deep features for small-scale AR/VR apps.  

---
