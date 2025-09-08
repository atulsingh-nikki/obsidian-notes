---
title: "MOTR: End-to-End Multiple-Object Tracking with Transformers (2022)"
aliases:
  - MOTR
  - Multiple Object Tracking with Transformers
authors:
  - Fangao Zeng
  - Bin Dong
  - Tiancai Wang
  - Xiangyu Zhang
  - Yichen Wei
year: 2022
venue: "ECCV"
doi: "10.48550/arXiv.2105.03247"
arxiv: "https://arxiv.org/abs/2105.03247"
code: "https://github.com/megvii-research/MOTR"
citations: 1000+
dataset:
  - MOT17
  - MOT20
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
  - "[[TransTrack (2021)]]"
  - "[[ByteTrack (2022)]]"
  - "[[FairMOT (2020)]]"
predecessors:
  - "[[TransTrack (2021)]]"
successors:
  - "[[Next-Gen Transformer MOT (2023+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**MOTR** extended **DETR-style transformers** into a fully end-to-end multiple object tracker with **persistent queries** that directly represent tracks. It eliminated hand-crafted motion models and explicit data association, offering a unified solution for detection and tracking.

# Key Idea
> Introduce **persistent track queries** in a transformer decoder that survive across frames, so each query continuously represents an object being tracked.

# Method
- **Base architecture**: DETR (encoder–decoder transformer).  
- **Track queries**:  
  - Initialized from detections.  
  - Persist across frames, updated by transformer attention.  
- **Output**: Bounding boxes + identities in a single framework.  
- **Training**: End-to-end with Hungarian matching (detection + tracking loss).  
- **Association**: Handled implicitly by query persistence.  

# Results
- Outperformed TransTrack and strong CNN baselines on MOT17 and MOT20.  
- Better long-term identity consistency.  
- Demonstrated transformers’ potential for **fully end-to-end MOT**.  

# Why it Mattered
- Removed need for explicit association or motion filtering.  
- Strong step toward **unifying detection and tracking** in one transformer pipeline.  
- Influential in autonomous driving and surveillance research.  

# Architectural Pattern
- Transformer encoder–decoder.  
- Persistent track queries representing object identities.  

# Connections
- Direct successor to **TransTrack (2021)**.  
- Competes with ByteTrack-style detection-association pipelines.  
- Inspired later transformer-based MOT research.  

# Implementation Notes
- More computationally expensive than ByteTrack/SORT family.  
- Training requires large datasets (e.g., BDD100K).  
- Public code and pretrained models available.  

# Critiques / Limitations
- Heavy compute and memory usage compared to classic trackers.  
- Still struggled in crowded scenes compared to detection-based methods.  
- Not real-time for large-scale deployment.  

---

# Educational Connections

## Undergraduate-Level Concepts
- How transformers can unify detection + tracking.  
- What a **query** is in a transformer.  
- Difference between MOT with hand-crafted association vs end-to-end learning.  
- Applications: tracking cars, pedestrians, and players.  

## Postgraduate-Level Concepts
- Persistent queries as dynamic memory for object identities.  
- Hungarian matching for detection + tracking.  
- Trade-offs: end-to-end elegance vs efficiency.  
- Future: extending MOTR to 3D MOT and multimodal setups.  

---

# My Notes
- MOTR was the **flagship transformer MOT** — elegant, fully end-to-end.  
- Represents the “other pole” vs ByteTrack: **theory-driven vs simplicity-driven**.  
- Open question: Can MOTR-scale transformers reach **real-time deployment** in self-driving?  
- Possible extension: Hybrid **ByteTrack motion simplicity + MOTR transformer memory**.  

---
