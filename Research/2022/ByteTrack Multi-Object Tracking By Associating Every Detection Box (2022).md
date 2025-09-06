---
title: "ByteTrack: Multi-Object Tracking By Associating Every Detection Box (2022)"
aliases:
  - ByteTrack
  - ByteTrack MOT
authors:
  - Yifu Zhang
  - Peize Sun
  - Yi Jiang
  - Dongdong Yu
  - Zehuan Yuan
  - Ping Luo
  - Wenyu Liu
  - Xinggang Wang
year: 2022
venue: "ECCV"
doi: "10.48550/arXiv.2110.06864"
arxiv: "https://arxiv.org/abs/2110.06864"
code: "https://github.com/ifzhang/ByteTrack"
citations: 3000+
dataset:
  - MOT17
  - MOT20
  - CrowdHuman
  - BDD100K
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - detection
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[DeepSORT (2017)]]"
  - "[[FairMOT (2020)]]"
predecessors:
  - "[[SORT (2016)]]"
  - "[[DeepSORT (2017)]]"
  - "[[FairMOT (2020)]]"
successors:
  - "[[OC-SORT (2023)]]"
  - "[[BoT-SORT (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**ByteTrack** introduced a simple yet powerful approach for **multi-object tracking (MOT)** by associating **all detection boxes**, including low-confidence ones. This allowed it to robustly handle **occlusions and missed detections**, achieving state-of-the-art performance with minimal complexity.

# Key Idea
> Unlike prior trackers that discard low-confidence detections, ByteTrack **keeps and associates every detection box**, using high-confidence detections to anchor tracks and low-confidence ones to recover identities during occlusions.

# Method
- **Detection input**: Uses outputs from modern object detectors (YOLOX in the paper).  
- **High-confidence detections**: Used for standard data association with tracks.  
- **Low-confidence detections**: Associated in a second stage to recover missed objects during occlusions.  
- **Tracking pipeline**:  
  1. Predict motion with Kalman filter.  
  2. First match high-confidence boxes.  
  3. Then match low-confidence ones for recovery.  

# Results
- Outperformed prior SOTA methods on **MOT17 and MOT20** benchmarks.  
- Robust against occlusion and crowded scenarios.  
- Faster and simpler than heavy appearance-based trackers.  

# Why it Mattered
- Showed that **simplicity + good use of detections** can beat complex methods.  
- Became a **new baseline** for MOT research.  
- Widely adopted in **autonomous driving, surveillance, sports analytics**.  

# Architectural Pattern
- Detector (e.g., YOLOX) → ByteTrack association → MOT results.  
- Two-stage matching (high → low confidence).  

# Connections
- Builds on **SORT/DeepSORT** for Kalman filter-based tracking.  
- Related to **FairMOT (2020)** (unified detection + tracking).  
- Inspired **OC-SORT and BoT-SORT (2023)** as successors.  

# Implementation Notes
- Runs in real-time on GPU.  
- Very easy to implement compared to appearance-heavy methods.  
- Released as open-source (PyTorch + YOLOX).  

# Critiques / Limitations
- Depends heavily on quality of detector.  
- Simple motion model (Kalman filter) can fail with complex dynamics.  
- Doesn’t explicitly use appearance features (unlike DeepSORT).  

---

# Educational Connections

## Undergraduate-Level Concepts
- What multi-object tracking (MOT) is.  
- Difference between **detection** and **tracking**.  
- Role of confidence scores in object detection.  
- How occlusion causes tracking failures.  

## Postgraduate-Level Concepts
- Data association algorithms in MOT.  
- Trade-offs between appearance-based and detection-based tracking.  
- ByteTrack’s two-stage matching strategy vs traditional one-stage.  
- Extensions to long-term tracking, autonomous driving, crowded scenes.  

---

# My Notes
- ByteTrack is a **classic case of a simple idea with huge impact**.  
- Reminds me that many ML advances are not about fancier models, but **smarter use of existing outputs**.  
- Open question: How to extend ByteTrack for **3D MOT (LiDAR + camera fusion)**?  
- Possible extension: Integrate ByteTrack with **end-to-end transformer-based trackers**.  

---
