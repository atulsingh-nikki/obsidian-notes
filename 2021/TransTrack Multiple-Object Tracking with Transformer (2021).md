---
title: "TransTrack: Multiple-Object Tracking with Transformer (2021)"
aliases:
  - TransTrack
authors:
  - Peize Sun
  - Yi Jiang
  - Rufeng Zhang
  - Jian Wang
  - Changhu Wang
  - Philipp Krähenbühl
  - Zehuan Yuan
year: 2021
venue: "arXiv preprint (later CVPR workshops)"
doi: "10.48550/arXiv.2012.15460"
arxiv: "https://arxiv.org/abs/2012.15460"
code: "https://github.com/PeizeSun/TransTrack"
citations: 700+
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
  - transformers
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[FairMOT (2020)]]"
  - "[[CenterTrack (2020)]]"
  - "[[MOTR (2022)]]"
predecessors:
  - "[[CenterTrack (2020)]]"
  - "[[FairMOT (2020)]]"
successors:
  - "[[MOTR (2022)]]"
  - "[[Next-Gen Transformer MOT (2023+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**TransTrack** was one of the first works to apply **transformers** to multi-object tracking (MOT). By extending **DETR (Detection Transformer)** to predict **object queries across frames**, it jointly handled detection and tracking in a unified framework.

# Key Idea
> Extend **DETR-style object queries** across frames, where queries persist over time to represent object tracks, effectively unifying detection and tracking with transformers.

# Method
- **Base architecture**: DETR (transformer encoder–decoder).  
- **Tracking extension**:  
  - Use **object queries** not only for detection in a single frame but also for **temporal association** across frames.  
  - Each query = one tracked object, updated over time.  
- **Outputs**: Bounding boxes + object IDs across frames.  
- **Training**: End-to-end with bipartite matching (Hungarian loss).  

# Results
- Competitive with strong baselines on MOT17, MOT20.  
- Better long-term association than motion-based trackers.  
- Demonstrated transformers’ potential for MOT.  

# Why it Mattered
- First to bring **transformers into MOT**, aligning with the DETR revolution in detection.  
- Paved the way for **MOTR (2022)** and other transformer-based MOT models.  
- Showed that transformers can unify detection, tracking, and association naturally.  

# Architectural Pattern
- DETR-style encoder–decoder.  
- Persistent object queries across frames.  

# Connections
- Inspired by **DETR (2020)**.  
- Built on ideas from **CenterTrack** (motion offsets) and **FairMOT** (ReID).  
- Direct predecessor to **MOTR (2022)**, which refined query persistence.  

# Implementation Notes
- Slower than real-time due to transformer complexity.  
- Performance improved with stronger backbones.  
- Public PyTorch implementation available.  

# Critiques / Limitations
- Limited scalability to large numbers of objects.  
- Transformers heavy compared to SORT/ByteTrack family.  
- ID switches still occur in heavy occlusion.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What a transformer is in vision tasks.  
- How object queries in DETR represent detections.  
- Idea of reusing queries to track objects over time.  
- Applications: tracking pedestrians, vehicles, sports players.  

## Postgraduate-Level Concepts
- End-to-end bipartite matching loss (Hungarian loss).  
- Query persistence across temporal sequences.  
- Comparison: transformer-based MOT vs Kalman/ReID pipelines.  
- Implications for scaling to 3D MOT and multimodal tracking.  

---

# My Notes
- TransTrack was the **bridge from CNN trackers to transformer-based MOT**.  
- It showed the conceptual elegance: tracking = persistent queries.  
- Open question: How to make transformer MOT efficient enough for **real-time autonomous driving**?  
- Possible extension: Fuse TransTrack with **video transformers (ViT/TimeSformer)** for richer spatio-temporal reasoning.  

---
