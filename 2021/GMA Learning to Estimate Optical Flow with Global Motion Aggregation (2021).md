---
title: "GMA: Learning to Estimate Optical Flow with Global Motion Aggregation (2021)"
aliases:
  - GMA
  - Global Motion Aggregation for Optical Flow
authors:
  - Shihao Jiang
  - Dylan Campbell
  - Yiran Zhong
  - Yuchao Dai
  - Hongdong Li
  - Richard Hartley
year: 2021
venue: "ICCV"
doi: "10.1109/ICCV48922.2021.00445"
arxiv: "https://arxiv.org/abs/2104.02409"
code: "https://github.com/zacjiang/GMA"
citations: ~600+
dataset:
  - FlyingChairs
  - FlyingThings3D
  - Sintel
  - KITTI
tags:
  - paper
  - optical-flow
  - attention
  - motion-estimation
fields:
  - vision
  - motion-estimation
related:
  - "[[RAFT (2020)]]"
  - "[[FlowFormer (2022)]]"
predecessors:
  - "[[RAFT (2020)]]"
successors:
  - "[[FlowFormer (2022)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**GMA** extended RAFT by introducing a **global motion aggregation module** that leverages **self-attention** to capture long-range dependencies in the correlation volume. This allowed the model to aggregate motion cues across the entire image, improving robustness in challenging scenarios such as large displacements, occlusions, and textureless regions.

# Key Idea
> Enhance RAFT’s recurrent refinement with **attention-driven global aggregation** of motion cues, letting the model reason beyond local neighborhoods.

# Method
- **Feature extraction**: Same as RAFT (CNN encoder).  
- **All-pairs correlation volume**: Dense 4D correlation computed as in RAFT.  
- **Global Motion Aggregation (GMA) module**:  
  - Attention mechanism aggregates correlations globally across the image.  
  - Captures consistent motion patterns, even far apart.  
- **Recurrent update operator**: Similar to RAFT’s GRU, but updated with GMA-enhanced features.  

# Results
- Improved accuracy over RAFT on **Sintel** (both clean and final passes).  
- Achieved better performance on **KITTI benchmarks**, especially in occlusion-heavy regions.  
- Stronger generalization to real-world motion compared to RAFT.  

# Why it Mattered
- Showed the value of **global context in optical flow**.  
- Pioneered the use of **attention mechanisms** in flow estimation, leading to transformer-based successors like FlowFormer.  
- Demonstrated that adding global reasoning improved robustness without discarding RAFT’s efficient refinement.  

# Architectural Pattern
- RAFT backbone → correlation volume → GMA attention module → recurrent GRU refinement.  

# Connections
- Successor to **RAFT**.  
- Predecessor to transformer-based optical flow models (e.g., FlowFormer).  
- Related to attention-based dense prediction networks in vision.  

# Implementation Notes
- Training pipeline similar to RAFT (FlyingChairs → FlyingThings3D → Sintel/KITTI).  
- GMA adds modest compute overhead compared to RAFT.  
- Code and pretrained weights released (PyTorch).  

# Critiques / Limitations
- Attention module increases memory and latency compared to RAFT.  
- Still limited by correlation volume’s memory footprint for very high resolutions.  
- Later transformers (FlowFormer, GMFlow) surpassed GMA in both performance and efficiency.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Optical flow basics.  
- Attention for global context.  
- How RAFT iterative refinement works.  

## Postgraduate-Level Concepts
- Incorporating global reasoning into dense prediction tasks.  
- Trade-offs between local recurrence (RAFT) and global attention (GMA).  
- Role of inductive bias in motion estimation (local vs global).  

---

# My Notes
- GMA is the **natural next step after RAFT**: add global attention to overcome local limits.  
- Feels like a bridge toward **transformer-based flow models**.  
- Open question: Could GMA-style global aggregation stabilize **video diffusion models** for long-range temporal coherence?  
- Possible extension: Combine GMA with hierarchical patch-based attention (as in Swin) for efficient high-res flow.  

---
