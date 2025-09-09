---
title: "DeepSORT: Deep Learning for Multiple Object Tracking (2017)"
aliases:
  - DeepSORT
  - Deep SORT
authors:
  - Nicolai Wojke
  - Alex Bewley
  - Dietrich Paulus
year: 2017
venue: "ICIP"
doi: "10.1109/ICIP.2017.8296962"
arxiv: "https://arxiv.org/abs/1703.07402"
code: "https://github.com/nwojke/deep_sort"
citations: 7000+
dataset:
  - MOT16
  - MOT17
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - re-identification
  - appearance
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[SORT (2016)]]"
  - "[[ByteTrack (2022)]]"
  - "[[BoT-SORT (2023)]]"
  - "[[OC-SORT (2023)]]"
predecessors:
  - "[[SORT (2016)]]"
successors:
  - "[[ByteTrack (2022)]]"
  - "[[BoT-SORT (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**DeepSORT** extended **SORT** by integrating **appearance features** from a deep neural network (ReID embeddings) with motion information for **multi-object tracking (MOT)**. This reduced identity switches and improved robustness in crowded and occluded environments.

# Key Idea
> Add **deep appearance embeddings** to SORT’s Kalman filter + IoU pipeline, allowing robust association even when motion cues alone are insufficient.

# Method
- **Base tracker**: SORT (Kalman filter + Hungarian assignment).  
- **Appearance features**:  
  - Extract ReID embeddings from a deep CNN trained on person re-identification.  
  - Use cosine similarity to compare detections with existing tracks.  
- **Fusion**: Motion + appearance similarity jointly drive association decisions.  
- **Output**: More stable track identities across occlusions.  

# Results
- Outperformed SORT on MOT16 and MOT17 benchmarks.  
- Significantly reduced **ID switches (IDS)** in crowded scenes.  
- Became a standard baseline in MOT research for years.  

# Why it Mattered
- First **appearance-augmented MOT tracker** widely adopted.  
- Influential in surveillance, autonomous driving, sports analytics.  
- Paved the way for later trackers (ByteTrack, BoT-SORT, OC-SORT).  

# Architectural Pattern
- Detector → ReID feature extractor → motion + appearance fusion → association.  

# Connections
- Predecessor: **SORT (2016)** (motion-only).  
- Successors: **ByteTrack (2022)** (every box association), **BoT-SORT (2023)** (modern ReID integration).  
- Complementary to modern transformer-based end-to-end trackers.  

# Implementation Notes
- Requires a pretrained ReID model.  
- More computationally expensive than SORT but still efficient.  
- Public code widely used as MOT baseline.  

# Critiques / Limitations
- Still reliant on detection quality.  
- ReID features trained on person datasets → less general for non-human MOT.  
- Struggles in long-term occlusions despite improvements.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Basics of MOT and why identity preservation matters.  
- What **ReID (re-identification)** is: matching identities across frames.  
- Fusion of motion and appearance for tracking.  
- Example: tracking people in a crowded video.  

## Postgraduate-Level Concepts
- Metric learning for robust appearance embeddings.  
- Trade-offs between motion-only vs appearance-augmented MOT.  
- Scalability: ReID generalization to non-person objects.  
- Transition from hand-crafted + deep features → end-to-end learned trackers.  

---

# My Notes
- DeepSORT was the **first big leap** in MOT after SORT — appearance integration made MOT practical in real-world crowded scenarios.  
- It shows the evolution path: **SORT → DeepSORT → ByteTrack → OC-SORT/BoT-SORT → Transformers**.  
- Open question: Can lightweight ReID models make appearance tracking as efficient as ByteTrack-style methods?  
- Possible extension: Fuse DeepSORT-style ReID with modern vision transformers for more general MOT across object categories.  

---
