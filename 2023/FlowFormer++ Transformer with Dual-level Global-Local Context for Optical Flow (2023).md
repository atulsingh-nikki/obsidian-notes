---
title: "FlowFormer++: Transformer with Dual-level Global-Local Context for Optical Flow (2023)"
aliases:
  - FlowFormer++
  - Improved FlowFormer
authors:
  - Zhicheng Huang
  - Jiahui Zhang
  - Liang Pan
  - Haiyu Zhao
  - Shuai Yi
  - Hongsheng Li
  - Xizhou Zhu
year: 2023
venue: "ICCV"
doi: "10.1109/ICCV51070.2023.00222"
arxiv: "https://arxiv.org/abs/2303.01237"
code: "https://github.com/drinkingcoder/FlowFormer-Official"
citations: ~200+
dataset:
  - FlyingChairs
  - FlyingThings3D
  - Sintel
  - KITTI
tags:
  - paper
  - optical-flow
  - transformers
  - motion-estimation
  - dual-context
fields:
  - vision
  - dense-prediction
  - motion-estimation
related:
  - "[[FlowFormer (2022)]]"
  - "[[GMFlow+ (2023)]]"
predecessors:
  - "[[FlowFormer (2022)]]"
successors:
  - "[[Future Optical Flow Transformers (2024+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**FlowFormer++** improves upon FlowFormer by incorporating **dual-level global-local context reasoning** within a transformer architecture. It strengthens both **global motion reasoning** (for large displacements) and **local detail refinement** (for sharp boundaries), setting new SOTA on multiple optical flow benchmarks.

# Key Idea
> Enrich the transformer’s correlation encoding with **both global motion context and local detail refinement**, bridging the gap between large-scale matching and fine-grained accuracy.

# Method
- **Dual-level context**:  
  - **Global context**: Transformer-based cost volume encoder captures long-range dependencies across entire frames.  
  - **Local context**: Dedicated local refinement branch sharpens motion boundaries and small-scale details.  
- **Flow decoder**: Merges global + local cues iteratively to predict flow.  
- **Training**: Same synthetic-to-real schedule (Chairs → Things3D → Sintel/KITTI).  

# Results
- Outperformed FlowFormer, RAFT, GMA, and GMFlow+ on **Sintel (clean/final)** and **KITTI**.  
- Achieved **new SOTA** optical flow accuracy in 2023.  
- Balanced global robustness with local precision.  

# Why it Mattered
- Addressed FlowFormer’s limitation: global reasoning but weaker fine-grained detail capture.  
- Demonstrated that **global + local dual context** is key for optical flow.  
- Solidified transformer dominance in optical flow research.  

# Architectural Pattern
- Transformer encoder over correlation features (global).  
- Local refinement branch (boundary-aware, fine motion).  
- Decoder fuses both for final prediction.  

# Connections
- Direct successor to **FlowFormer (2022)**.  
- Parallel improvement to **GMFlow+ (2023)**.  
- Likely influenced **2024+ optical flow foundation models**.  

# Implementation Notes
- Higher compute cost than GMFlow/RAFT but best accuracy.  
- Requires careful balancing of global and local supervision.  
- Code and pretrained weights publicly released.  

# Critiques / Limitations
- More complex than FlowFormer, less efficient for real-time use.  
- Memory-intensive at high resolutions.  
- Still depends on synthetic pretraining for generalization.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Optical flow as dense motion estimation.  
- Why both global and local context matter.  
- Transformers as global context aggregators.  

## Postgraduate-Level Concepts
- Dual-branch architectures for context fusion.  
- Trade-offs between accuracy and efficiency in dense prediction.  
- Lessons for general correspondence tasks (e.g., stereo, video tracking).  

---

# My Notes
- FlowFormer++ feels like the **"Swin Transformer moment"** for optical flow: efficient global-local fusion.  
- Likely to inspire **general dense correspondence backbones** for vision tasks.  
- Open question: Can FlowFormer++ dual-context reasoning be applied to **video diffusion models** for long-range temporal consistency + sharp frame-level detail?  
- Possible extension: Integrate into **foundation video editing backbones** for motion-aware consistency.  

---
