---
title: KITTI Tracking Benchmark (2012)
aliases:
  - KITTI Tracking
  - KITTI MOT
authors:
  - Andreas Geiger
  - Philip Lenz
  - Christoph Stiller
  - Raquel Urtasun
year: 2012
venue: CVPR / IJRR (KITTI dataset release)
doi: 10.1109/CVPR.2012.6248074
arxiv: https://www.cv-foundation.org/openaccess/content_cvpr_2012/papers/Geiger_Automotive_Vision_Benchmark_2012_CVPR_paper.pdf
code: http://www.cvlibs.net/datasets/kitti/eval_tracking.php
citations: 15000+
dataset:
  - KITTI Tracking Benchmark
tags:
  - dataset
  - tracking
  - multi-object-tracking
  - mot
  - autonomous-driving
fields:
  - vision
  - tracking
  - autonomous-driving
  - datasets
related:
  - "[[MOT16 Multi-Object Tracking Benchmark (2016)|MOT16]]"
  - "[[MOT17 Multi-Object Tracking Benchmark (2017)|MOT17]]"
  - "[[MOT20 A Benchmark for Multi-Object Tracking in Crowded Scenes (2020)|MOT20]]"
  - "[[DanceTrack A Benchmark for Multi-Object Tracking in Crowded Scenes (2022)|DanceTrack]]"
predecessors: []
successors:
  - "[[nuScenes Dataset (2019)]]"
  - "[[Waymo Open Dataset (2019)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
The **KITTI Tracking Benchmark** was part of the broader KITTI dataset suite (Geiger et al., 2012), designed for **autonomous driving research**. It provided one of the first large-scale, real-world datasets for **multi-object tracking (MOT)** in driving scenarios, including both **vehicles and pedestrians**.

# Key Idea
> Capture **real-world street scenes** from a car-mounted sensor rig, and provide high-quality annotations for object detection, tracking, and other tasks critical to autonomous driving.

# Dataset Details
- **Content**: Urban, rural, and highway driving scenes.  
- **Sensors**: Stereo cameras, Velodyne LiDAR, GPS/IMU.  
- **Annotations**:  
  - 2D bounding boxes.  
  - 3D bounding boxes.  
  - Object identities across frames.  
- **Classes**: Cars, pedestrians, cyclists, vans, trucks, trams.  
- **Scale**: ~40 sequences, with thousands of annotated frames.  

# Results
- First major benchmark for **3D tracking** in autonomous driving.  
- Became a reference dataset for both **2D MOT** and **3D MOT**.  
- Inspired many early MOT algorithms later benchmarked on MOT16/17.  

# Why it Mattered
- Provided **real driving data**, unlike MOTChallenge’s pedestrian-only focus.  
- Multimodal (camera + LiDAR + GPS) → a precursor to modern autonomous driving datasets.  
- Influenced successors like **nuScenes (2019)** and **Waymo Open Dataset (2019)**.  

# Connections
- Contemporary of ImageNet (classification) and MOT16 (pedestrian MOT).  
- Predecessor to **nuScenes**, **Waymo Open**, **Argoverse**, which expanded to larger scales.  
- Still used for benchmarking tracking in driving environments.  

# Critiques / Limitations
- Limited dataset size compared to modern driving datasets.  
- Geographic bias (collected in Karlsruhe, Germany).  
- Fewer crowded scenes than MOT20/DanceTrack.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Basics of autonomous driving datasets.  
- Why MOT in driving = tracking cars, pedestrians, cyclists.  
- How multimodal data (camera + LiDAR) helps tracking.  
- Example: car following a cyclist in traffic.  

## Postgraduate-Level Concepts
- Transition from **2D MOT benchmarks (MOT16)** to **3D MOT benchmarks (KITTI)**.  
- Impact of multimodal sensor fusion on tracking algorithms.  
- KITTI’s role in **cross-task benchmarks** (detection, segmentation, tracking, odometry).  
- Dataset design trade-offs: scale vs annotation quality.  

---

# My Notes
- KITTI = the **starting line for autonomous driving MOT research**.  
- Its 3D annotations set it apart from MOT16/17 → bridging classic MOT with driving.  
- Open question: Will future datasets follow KITTI’s **multimodal philosophy** or focus on massive scale like LAION-style internet data?  
- Possible extension: A **KITTI-Next** dataset blending LiDAR, radar, and HD maps for long-range MOT.  

---
