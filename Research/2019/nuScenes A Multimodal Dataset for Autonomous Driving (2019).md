---
title: "nuScenes: A Multimodal Dataset for Autonomous Driving (2019)"
aliases:
  - nuScenes
  - nuScenes Dataset
authors:
  - Holger Caesar
  - Varun Bankiti
  - Alex H. Lang
  - Sourabh Vora
  - Venice Erin Liong
  - Qiang Xu
  - Anush Krishnan
  - Yu Pan
  - Giancarlo Baldan
  - Oscar Beijbom
year: 2019
venue: "CVPR"
doi: "10.48550/arXiv.1903.11027"
arxiv: "https://arxiv.org/abs/1903.11027"
code: "https://www.nuscenes.org/"
citations: 6000+
dataset:
  - nuScenes (multimodal dataset)
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
  - "[[Waymo Open Dataset (2019)]]"
  - "[[Argoverse (2019)]]"
predecessors:
  - "[[KITTI Tracking Benchmark (2012)]]"
successors:
  - "[[nuScenes-lidarseg (2020)]]"
  - "[[Waymo Open Dataset (2019)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**nuScenes** is a **large-scale multimodal dataset** for autonomous driving, released by nuTonomy (later part of Motional). It extended KITTI by providing **10x more data, richer sensor coverage, and HD maps**, becoming a new standard for **3D object detection and tracking**.

# Key Idea
> Provide a **comprehensive multimodal dataset** (cameras, LiDAR, radar, IMU, GPS) with full 360° coverage and annotations for **3D detection, tracking, and motion forecasting**.

# Dataset Details
- **Scale**: 1000 driving scenes (~20 seconds each).  
- **Sensors**:  
  - 6 RGB cameras (360°).  
  - 1 LiDAR (360°).  
  - 5 radars.  
  - GPS/IMU for localization.  
- **Annotations**:  
  - 3D bounding boxes with class + track IDs.  
  - ~1.4M annotated object instances.  
- **Classes**: 23 categories (cars, pedestrians, trucks, bicycles, barriers, etc.).  
- **Extra**: HD maps, weather/time diversity.  

# Results
- Set a new standard for 3D MOT benchmarks.  
- Widely adopted for 3D detection and tracking competitions.  
- Exposed limitations of KITTI (scale, diversity, coverage).  

# Why it Mattered
- First **truly large-scale multimodal 3D MOT dataset**.  
- Essential for advancing research in **autonomous driving**.  
- Drove development of transformer and fusion-based 3D MOT algorithms.  

# Connections
- Successor to **KITTI Tracking Benchmark (2012)**.  
- Contemporary with **Waymo Open Dataset (2019)** and **Argoverse (2019)**.  
- Complementary to MOTChallenge datasets (2D pedestrian MOT).  

# Critiques / Limitations
- Scenes limited to **Boston and Singapore** → geographic bias.  
- 20-second clips only; lacks long continuous driving.  
- Annotation noise in radar data.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Why multimodal sensors (camera + LiDAR + radar) are used in self-driving.  
- Difference between 2D MOT (bounding boxes) and 3D MOT (volumes).  
- What HD maps add to tracking.  
- Example: tracking cars across intersections with occlusion.  

## Postgraduate-Level Concepts
- Multimodal sensor fusion for MOT.  
- Trade-offs between dataset scale and annotation cost.  
- nuScenes as a baseline for transformer-based 3D MOT models.  
- Extensions: trajectory forecasting, intention prediction.  

---

# My Notes
- nuScenes was the **true leap from KITTI**: full 360°, multimodal, large-scale.  
- Pushed research beyond cameras → into full **sensor fusion MOT**.  
- Open question: How to scale beyond **20s scenes** to hour-long trajectories for long-term planning?  
- Possible extension: A **nuScenes-XL** with longer sequences and simulation integration.  

---
