---
title: "MOTRv2: Improved End-to-End Multi-Object Tracking with Transformers (2023)"
aliases:
  - MOTRv2
authors:
  - Fangao Zeng
  - Tiancai Wang
  - Xiangyu Zhang
  - Yichen Wei
year: 2023
venue: "arXiv preprint"
doi: "10.48550/arXiv.2303.15135"
arxiv: "https://arxiv.org/abs/2303.15135"
code: "https://github.com/megvii-research/MOTRv2"
citations: 150+
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
  - transformers
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[MOTR (2022)]]"
  - "[[TransTrack (2021)]]"
  - "[[TrackFormer (2021)]]"
  - "[[ByteTrack (2022)]]"
predecessors:
  - "[[MOTR (2022)]]"
successors:
  - "[[Next-Gen Transformer MOT (2024+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**MOTRv2** improved **MOTR (2022)** by redesigning how **queries are maintained and updated across frames**, enabling stronger long-term association and higher accuracy in multi-object tracking benchmarks. It represents one of the most refined transformer-based MOT models to date.

# Key Idea
> Enhance **persistent track queries** from MOTR with better initialization, updating, and memory mechanisms, making transformers more robust to long occlusions and crowded scenes.

# Method
- **Architecture**: Still DETR-style encoder–decoder backbone.  
- **Key improvements**:  
  - **Better query initialization**: track queries seeded more reliably from detections.  
  - **Query updating**: improved temporal update rules.  
  - **Long-term association**: refined handling of disappearing/reappearing objects.  
- **Training**: Hungarian matching loss for detection + identity preservation, same as MOTR but with improved query dynamics.  

# Results
- Outperformed MOTR on MOT17, MOT20, DanceTrack, BDD100K.  
- Reduced ID switches significantly in long occlusion scenarios.  
- State-of-the-art among transformer MOT models at publication.  

# Why it Mattered
- Showed transformers can match or surpass detection-association methods like ByteTrack when carefully designed.  
- Improved long-term tracking stability, a major MOT challenge.  
- Extended viability of end-to-end transformer MOT pipelines.  

# Architectural Pattern
- DETR-style encoder–decoder.  
- Persistent queries with enhanced initialization + updating.  

# Connections
- Successor to **MOTR (2022)**.  
- Contemporary with CNN-based successors like **OC-SORT (2023)** and **BoT-SORT (2023)**.  
- Part of trend toward **transformer-native MOT**.  

# Implementation Notes
- Computationally expensive (not yet real-time).  
- Requires large-scale training data (e.g., BDD100K).  
- PyTorch implementation released.  

# Critiques / Limitations
- Still heavy in compute vs SORT/ByteTrack family.  
- Scalability to 3D or multimodal MOT not fully explored.  
- Needs larger datasets for robust generalization.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Review: what MOTR was (persistent queries in transformers).  
- Why improving query initialization helps tracking stability.  
- Real-world impact: fewer ID switches in crowded pedestrian tracking.  

## Postgraduate-Level Concepts
- Deep dive into query dynamics: initialization, update, persistence.  
- Long-term occlusion handling in transformer MOT.  
- Comparison with ByteTrack/OC-SORT (detection association) vs MOTRv2 (end-to-end).  
- Implications for scaling transformers in MOT to 3D and multimodal setups.  

---

# My Notes
- MOTRv2 is the **most polished transformer MOT to date**.  
- Addresses key MOT weaknesses: long occlusion + ID switches.  
- Open question: Can MOTRv2 be distilled or approximated for **real-time tracking** like ByteTrack?  
- Possible extension: Adapt MOTRv2’s query persistence to **3D tracking (camera + LiDAR)** for autonomous driving.  

---
