---
title: "MeMOT: Memory-Enhanced Multiple Object Tracking with Transformers (2022)"
aliases:
  - MeMOT
authors:
  - Zhipeng Cai
  - Limin Wang
  - et al.
year: 2022
venue: "CVPR"
doi: "10.48550/arXiv.2203.17082"
arxiv: "https://arxiv.org/abs/2203.17082"
code: "https://github.com/MCG-NJU/MeMOT"
citations: 250+
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
  - memory
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[MOTR (2022)]]"
  - "[[TransTrack (2021)]]"
  - "[[TrackFormer (2021)]]"
predecessors:
  - "[[TransTrack (2021)]]"
successors:
  - "[[MOTRv2 (2023)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**MeMOT** introduced a **memory-augmented transformer framework** for multi-object tracking (MOT). Unlike MOTR, which relied on persistent queries alone, MeMOT maintained an **explicit memory of track features** over time, improving robustness in long-term occlusions and crowded scenes.

# Key Idea
> Enhance transformer-based MOT with an **external memory bank** that stores and updates track features across frames, allowing better long-term reasoning beyond query persistence.

# Method
- **Base architecture**: Transformer encoder–decoder for detection and tracking.  
- **Memory bank**:  
  - Stores track embeddings (appearance + spatial features).  
  - Dynamically updated as frames progress.  
- **Tracking**: New detections matched against memory for association.  
- **Output**: Joint detection + identity tracking.  

# Results
- Outperformed many transformer and CNN-based baselines on MOT17, MOT20.  
- Stronger long-term identity preservation than MOTR.  
- More robust in crowded occlusions (DanceTrack).  

# Why it Mattered
- Showed the importance of **explicit memory** for MOT, complementing transformer queries.  
- Advanced the state of transformer-based trackers.  
- Inspired later improvements (MOTRv2, memory-augmented video transformers).  

# Architectural Pattern
- Transformer with detection head.  
- External memory bank for track features.  
- Memory–query interaction for association.  

# Connections
- Related to **MOTR (2022)** but adds explicit memory.  
- Successor to **TransTrack/TrackFormer**.  
- Predecessor to MOTRv2-style refinements.  

# Implementation Notes
- More compute-intensive than MOTR due to memory updates.  
- Memory bank size and update strategy critical.  
- PyTorch code and pretrained models available.  

# Critiques / Limitations
- High memory and compute costs.  
- Memory management adds complexity.  
- Performance depends on robust embedding quality.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What memory modules are in deep learning.  
- Why storing past features helps track objects through occlusion.  
- Comparison: Kalman filter memory vs neural memory.  
- Applications: person tracking in video, autonomous vehicles.  

## Postgraduate-Level Concepts
- Transformer–memory interactions in sequence modeling.  
- Trade-offs: query persistence vs explicit memory banks.  
- Long-term temporal reasoning in MOT.  
- Potential for scaling to video transformers and multimodal tracking.  

---

# My Notes
- MeMOT was a **clever extension of MOTR**: queries + explicit memory.  
- Brings MOT closer to how humans track: short-term focus + long-term memory.  
- Open question: How to balance **memory size vs efficiency** for real-time tracking?  
- Possible extension: Use **memory compression or hierarchical memory** for large-scale video MOT.  

---
