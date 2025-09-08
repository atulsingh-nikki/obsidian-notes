---
title: "MOT17: Multi-Object Tracking Benchmark (2017)"
aliases:
  - MOT17
  - MOT17 Dataset
authors:
  - Anton Milan
  - Laura Leal-Taixé
  - Ian Reid
  - Stefan Roth
  - Konrad Schindler
year: 2017
venue: "CVPR Workshops (MOTChallenge)"
doi: "10.1109/CVPRW.2017.546"
arxiv: "https://arxiv.org/abs/1603.00831"
code: "https://motchallenge.net/data/MOT17/"
citations: 5000+
dataset:
  - MOT17 (benchmark dataset)
tags:
  - dataset
  - tracking
  - multi-object-tracking
  - mot
  - pedestrian
fields:
  - vision
  - tracking
  - datasets
related:
  - "[[MOT16 Dataset (2016)]]"
  - "[[MOT20 Dataset (2020)]]"
  - "[[DanceTrack (2022)]]"
  - "[[SORT (2016)]]"
  - "[[DeepSORT (2017)]]"
  - "[[ByteTrack (2022)]]"
predecessors:
  - "[[MOT16 Dataset (2016)]]"
successors:
  - "[[MOT20 Dataset (2020)]]"
  - "[[DanceTrack (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**MOT17** is one of the most widely used benchmarks for **multi-object tracking (MOT)**, extending **MOT16** with improved annotations and more challenging sequences. It became the **de facto standard dataset** for evaluating tracking algorithms like SORT, DeepSORT, and their successors.

# Key Idea
> Provide a standardized, large-scale dataset with **diverse pedestrian tracking scenarios**, allowing consistent benchmarking of MOT algorithms.

# Dataset Details
- **Content**: 14 sequences from urban and campus environments.  
- **Annotations**: 2D bounding boxes with identity labels.  
- **Diversity**: Includes static and moving cameras, varying crowd densities, lighting, and occlusion.  
- **Detections**: Three detection sets (DPM, Faster R-CNN, SDP) for fair comparison.  
- **Size**: ~11,000 frames with 1.1 million bounding boxes.  

# Results
- Defined the benchmark where **SORT (2016)**, **DeepSORT (2017)**, and later trackers were evaluated.  
- Still widely used for benchmarking alongside MOT20 and DanceTrack.  

# Why it Mattered
- Set the **baseline MOT benchmark** for years.  
- Its variety exposed strengths/weaknesses of different tracker families.  
- Enabled fair competition and leaderboard-driven progress in MOT.  

# Connections
- Successor to **MOT16** (fixed annotation errors, added sequences).  
- Predecessor to **MOT20** (denser) and **DanceTrack** (dynamic).  
- Benchmarked many iconic trackers: SORT, DeepSORT, Tracktor, FairMOT, ByteTrack.  

# Critiques / Limitations
- Still limited to **pedestrians**.  
- Mostly 2D bounding boxes; no 3D/pose data.  
- Smaller in scale compared to modern large MOT datasets.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Why benchmarks matter for computer vision.  
- Difference between sparse pedestrian tracking vs dense crowds.  
- Example: campus vs street vs shopping mall tracking.  

## Postgraduate-Level Concepts
- Dataset bias: MOT17 → urban pedestrians only.  
- Why multiple detector sets matter for evaluation fairness.  
- Benchmark-driven research: progress from SORT → DeepSORT → ByteTrack.  
- How MOT17 shaped MOT evaluation standards.  

---

# My Notes
- MOT17 = **the ImageNet of MOT**: small by today’s standards, but hugely influential.  
- Every classic tracker has its MOT17 score quoted.  
- Open question: Should benchmarks evolve to **3D multimodal MOT** (LiDAR, radar, video fusion)?  
- Possible extension: A **MOT17-3D** style benchmark for autonomous driving pedestrians.  

---
