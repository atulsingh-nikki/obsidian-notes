---
title: "TrackFormer: Multi-Object Tracking with Transformers (2021)"
aliases:
  - TrackFormer
authors:
  - Tim Meinhardt
  - Alexander Kirillov
  - Laura Leal-Taixé
  - Christoph Feichtenhofer
year: 2021
venue: "arXiv preprint (later CVPR 2022)"
doi: "10.48550/arXiv.2101.02702"
arxiv: "https://arxiv.org/abs/2101.02702"
code: "https://github.com/aharley/trackformer"
citations: 1200+
dataset:
  - MOT17
  - MOT20
  - CrowdHuman
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - transformers
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[TransTrack (2021)]]"
  - "[[MOTR (2022)]]"
  - "[[FairMOT (2020)]]"
predecessors:
  - "[[DETR (2020)]]"
successors:
  - "[[MOTR (2022)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**TrackFormer** applied **DETR-style transformers** to multi-object tracking (MOT), treating tracking as a **set prediction problem** across frames. It introduced **track queries** that carry over identities frame to frame, linking detections without explicit motion models.

# Key Idea
> Extend **DETR** to MOT by reusing **track queries** across frames, so that queries serve as both detectors and trackers simultaneously.

# Method
- **Base model**: DETR (transformer encoder–decoder).  
- **Track queries**:  
  - Each object identity represented by a query.  
  - Queries propagated across frames to maintain tracks.  
- **Association**: Queries updated by transformer attention handle linking implicitly.  
- **Loss**: End-to-end Hungarian matching for detection + identity preservation.  

# Results
- Achieved competitive performance on MOT17, MOT20.  
- Improved identity preservation compared to CNN-based trackers.  
- Showed transformers’ flexibility for MOT.  

# Why it Mattered
- Among the **first transformer-based MOT approaches**.  
- Pioneered the idea of **end-to-end set-based tracking**.  
- Influenced later models like **MOTR (2022)**.  

# Architectural Pattern
- Transformer with encoder–decoder.  
- Persistent queries for both detection and tracking.  

# Connections
- Contemporary of **TransTrack (2021)** (similar query-based approach).  
- Predecessor to **MOTR (2022)**, which refined query persistence.  
- Complementary to CNN-based trackers like ByteTrack.  

# Implementation Notes
- Computationally heavy compared to SORT/ByteTrack.  
- End-to-end but slower for real-time deployment.  
- PyTorch implementation available.  

# Critiques / Limitations
- High compute cost.  
- Struggles in crowded scenes vs detection-based trackers.  
- Limited scalability at time of publication.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Transformers in vision (DETR).  
- How queries can represent objects.  
- What set prediction means in detection/tracking.  
- Applications: pedestrian/crowd tracking, autonomous driving.  

## Postgraduate-Level Concepts
- End-to-end training with Hungarian loss.  
- Comparison of query persistence vs motion-based association.  
- Trade-offs: elegance vs efficiency in transformer trackers.  
- Scaling TrackFormer to large, real-world MOT datasets.  

---

# My Notes
- TrackFormer = **DETR extended to MOT**.  
- Conceptually clean: one framework for detection + tracking.  
- Open question: Can transformer trackers outperform **lightweight, detector-based MOT (ByteTrack)** in both accuracy and speed?  
- Possible extension: **hybrid models** combining transformer reasoning with efficient detection pipelines.  

---
