---
title: "GMFlow: Learning Optical Flow via Global Matching (2022)"
aliases:
  - GMFlow
  - Global Matching Flow
authors:
  - Xiaoyu Xu
  - Zheyu Zhuang
  - Yifan Liu
  - Chunhua Shen
  - Baosheng Yu
year: 2022
venue: "CVPR"
doi: "10.1109/CVPR52688.2022.01609"
arxiv: "https://arxiv.org/abs/2111.13680"
code: "https://github.com/haofeixu/gmflow"
citations: ~500+
dataset:
  - FlyingChairs
  - FlyingThings3D
  - Sintel
  - KITTI
tags:
  - paper
  - optical-flow
  - transformers
  - matching
fields:
  - vision
  - motion-estimation
related:
  - "[[RAFT (2020)]]"
  - "[[GMA (2021)]]"
  - "[[FlowFormer (2022)]]"
predecessors:
  - "[[RAFT (2020)]]"
successors:
  - "[[GMFlow+ (2023)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**GMFlow** introduced a transformer-based optical flow framework that formulates flow estimation as a **global matching problem**. Instead of recurrent refinement (RAFT) or iterative attention (GMA), GMFlow performs **dense global matching in a single forward pass** using transformers, making it faster and simpler.

# Key Idea
> Predict optical flow by **directly matching features across the whole image** using transformer attention, avoiding iterative refinement loops.

# Method
- **Feature extraction**: CNN backbone → high-resolution features.  
- **Transformer encoder-decoder**:  
  - Models long-range dependencies across the two feature maps.  
  - Cross-attention identifies correspondences globally.  
- **Global matching**: Each pixel in image1 attends to all pixels in image2.  
- **Flow prediction**: Matches converted into dense displacement vectors.  

# Results
- Achieved strong performance on **Sintel** and **KITTI**.  
- Faster inference than RAFT (no recurrent refinement).  
- Competitive accuracy with FlowFormer while being simpler.  

# Why it Mattered
- Showed that **global matching with transformers** can rival recurrent refinement.  
- Simplified architecture with fewer moving parts.  
- Paved the way for **real-time transformer-based flow** models.  

# Architectural Pattern
- CNN feature extractor.  
- Transformer encoder-decoder for cross-image matching.  
- One-shot flow prediction (no recurrence).  

# Connections
- Related to RAFT’s correlation volume but replaces it with transformer matching.  
- Parallel development to **FlowFormer (2022)**.  
- Extended by GMFlow+ (2023) with better multi-scale handling.  

# Implementation Notes
- Faster than RAFT/GMA due to non-recurrent design.  
- Works well for large displacements (global matching).  
- Requires strong positional encoding for stable training.  

# Critiques / Limitations
- Single-pass design can miss fine details refined iteratively in RAFT.  
- High memory usage for large images due to dense attention.  
- Performance still lags behind FlowFormer++ on hardest benchmarks.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Optical flow as matching problem.  
- Transformers for cross-attention between two feature maps.  
- Trade-off: iterative refinement vs one-shot prediction.  

## Postgraduate-Level Concepts
- Global matching vs correlation volume.  
- Attention efficiency and memory cost in dense prediction.  
- Extensions to hierarchical/multi-scale transformers.  

---

# My Notes
- GMFlow is the **one-shot counterpart** to RAFT/FlowFormer: direct matching via transformers.  
- Great for speed-sensitive tasks like **real-time video editing**.  
- Open question: Can GMFlow be combined with RAFT’s iterative refinement for hybrid speed-accuracy tradeoffs?  
- Possible extension: Use GMFlow as initialization, then refine with diffusion-like iterative updates.  

---
