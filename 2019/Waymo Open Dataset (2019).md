---
title: "Waymo Open Dataset (2019)"
aliases:
  - Waymo Dataset
  - Waymo Open
authors:
  - Pei Sun
  - Henrik Kretzschmar
  - Xerxes Dotiwalla
  - Aurelien Chouard
  - Vijaysai Patnaik
  - Paul Tsui
  - James Guo
  - Yin Zhou
  - Jonas Krishnan
  - et al.
year: 2019
venue: "CVPR"
doi: "10.48550/arXiv.1912.04838"
arxiv: "https://arxiv.org/abs/1912.04838"
code: "https://waymo.com/open/"
citations: 4500+
dataset:
  - Waymo Open Dataset
tags:
  - dataset
  - tracking
  - multi-object-tracking
  - mot
  - 3d-tracking
  - autonomous-driving
  - multimodal
fields:
  - vision
  - tracking
  - autonomous-driving
  - datasets
related:
  - "[[KITTI Tracking Benchmark (2012)]]"
  - "[[nuScenes Dataset (2019)]]"
  - "[[Argoverse (2019)]]"
predecessors:
  - "[[KITTI Tracking Benchmark (2012)]]"
successors:
  - "[[Waymo Open Dataset v2 (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
The **Waymo Open Dataset** is one of the **largest autonomous driving datasets**, released by Waymo (Google’s self-driving division). It surpassed KITTI and nuScenes in **scale, sensor coverage, and diversity**, quickly becoming the **primary benchmark for 3D detection and multi-object tracking (MOT)**.

# Key Idea
> Provide a **massive-scale, multimodal dataset** collected by a real self-driving fleet, enabling research in 3D perception, tracking, and prediction at production scale.

# Dataset Details
- **Scale**: 1000+ driving segments, each 20s long → ~12M LiDAR frames, ~1M camera images.  
- **Sensors**:  
  - 5 LiDARs (1 top-mounted 360° + 4 short-range).  
  - 5 cameras (wide + narrow FOV).  
- **Annotations**:  
  - 3D bounding boxes with identities for tracking.  
  - Object categories: vehicles, pedestrians, cyclists, signs, etc.  
- **Diversity**: Collected in multiple US cities (sun, rain, dusk, night).  
- **Extras**: HD maps, lane centerlines, motion forecasting labels.  

# Results
- Became the **largest 3D MOT benchmark** at release.  
- Outperformed KITTI and nuScenes in diversity and richness.  
- Used for state-of-the-art benchmarks in detection, tracking, and forecasting.  

# Why it Mattered
- First dataset to reach **self-driving fleet scale**.  
- Rich multimodal annotations → enabled **multi-task research** (detection, tracking, forecasting).  
- Essential for scaling transformer and foundation models in driving.  

# Connections
- Successor to **KITTI (2012)** and **nuScenes (2019)**.  
- Predecessor to **Waymo Open v2 (2022)**, which expanded annotations further.  
- Complementary to Argoverse (2019).  

# Critiques / Limitations
- Data is US-centric → geographic bias.  
- More LiDAR-heavy than camera-heavy.  
- Storage/computation costs are high for small labs.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Difference between 2D MOT (bounding boxes) and 3D MOT (volumes).  
- Why self-driving cars use multiple LiDARs and cameras.  
- Example: tracking cars + cyclists at an intersection.  

## Postgraduate-Level Concepts
- Dataset scale impact on deep learning model generalization.  
- Multimodal sensor fusion challenges (LiDAR + camera).  
- Multi-task learning: detection, tracking, motion forecasting.  
- Foundation model training on large-scale driving datasets.  

---

# My Notes
- Waymo Open = **the ImageNet of 3D MOT**.  
- Its scale enabled training transformer-based 3D perception models.  
- Open question: Can Waymo-scale data be **openly released at global scale** (beyond the US)?  
- Possible extension: **foundation driving model pretraining** across Waymo + nuScenes + Argoverse + KITTI.  

---
