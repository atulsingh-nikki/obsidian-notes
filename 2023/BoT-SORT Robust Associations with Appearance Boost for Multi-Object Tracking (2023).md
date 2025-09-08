---
title: "BoT-SORT: Robust Associations with Appearance Boost for Multi-Object Tracking (2023)"
aliases:
  - BoT-SORT
  - Bag-of-Tricks SORT
authors:
  - Elad Aharon
  - Roy Friedman
  - Tal Hassner
year: 2023
venue: CVPR
doi: 10.48550/arXiv.2206.14651
arxiv: https://arxiv.org/abs/2206.14651
code: https://github.com/NirAharon/BoT-SORT
citations: 300+
dataset:
  - MOT17
  - MOT20
  - DanceTrack
  - BDD100K
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
  - "[[DeepSORT (2017)]]"
  - "[[OC-SORT Observation-Centric SORT (2023)|OC-SORT]]"
  - "[[ByteTrack Multi-Object Tracking By Associating Every Detection Box (2022)|ByteTrack]]"
predecessors:
  - "[[SORT (2016)]]"
  - "[[DeepSORT (2017)]]"
  - "[[ByteTrack (2022)]]"
successors:
  - "[[Next-Gen MOT Transformers (2024+)]]"
impact: ⭐⭐⭐⭐☆
status: read
---

# Summary
**BoT-SORT** enhanced the SORT/ByteTrack family of trackers by **integrating appearance features (ReID embeddings)** with motion cues, improving robustness to **identity switches** in crowded or occluded multi-object tracking (MOT) scenarios.

# Key Idea
> Augment the simple Kalman filter + IoU matching pipeline with **appearance embeddings** from a ReID network, balancing motion and appearance similarity for more reliable associations.

# Method
- **Base tracker**: SORT/ByteTrack association framework.  
- **Appearance boost**:  
  - Extract **ReID features** from detected objects.  
  - Combine them with motion-based similarity for association.  
- **Additional tricks** ("bag of tricks"):  
  - Camera motion compensation.  
  - Refined Kalman filter updates.  
  - Robust cosine distance metrics.  
- **Result**: Stronger identity preservation across frames.  

# Results
- Outperformed ByteTrack and OC-SORT on **MOT17, MOT20, DanceTrack**.  
- Reduced ID switches, especially in crowded scenes.  
- High accuracy while maintaining near real-time performance.  

# Why it Mattered
- Closed the gap between **appearance-heavy trackers (DeepSORT)** and **detection-based trackers (ByteTrack)**.  
- Became one of the strongest open-source MOT baselines.  
- Balanced simplicity with robustness.  

# Architectural Pattern
- Detector → ReID embedding extractor → BoT-SORT association.  
- Motion + appearance fusion for final matching.  

# Connections
- Complementary to **OC-SORT (2023)** (motion refinement vs appearance boost).  
- Successor to **ByteTrack (2022)**.  
- Related to **DeepSORT (2017)** but more modern and effective.  

# Implementation Notes
- Requires training or loading a strong ReID model.  
- More compute-intensive than ByteTrack/OC-SORT but still efficient.  
- Public PyTorch code widely adopted.  

# Critiques / Limitations
- Extra cost of ReID feature extraction.  
- Still uses Kalman filter motion model (hand-crafted).  
- Identity switches not fully eliminated in extreme occlusion cases.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What multi-object tracking is.  
- Role of **appearance features vs motion** in tracking.  
- Why occlusions cause ID switches.  
- How combining multiple cues improves robustness.  

## Postgraduate-Level Concepts
- Re-identification (ReID) embeddings and metric learning.  
- Fusion of heterogeneous features (motion + appearance).  
- Trade-offs between **accuracy, robustness, and efficiency** in MOT.  
- Future: fully end-to-end learned trackers (transformer-based).  

---

# My Notes
- BoT-SORT = **ByteTrack + appearance features + a bag of engineering tricks**.  
- Very practical: achieves state-of-the-art while remaining deployable.  
- Open question: Can **transformer-based trackers subsume ReID + motion fusion**?  
- Possible extension: Apply BoT-SORT principles to **3D MOT with camera–LiDAR fusion**.  

---
