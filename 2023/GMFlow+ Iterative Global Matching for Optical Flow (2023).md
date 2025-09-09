---
title: "GMFlow+: Iterative Global Matching for Optical Flow (2023)"
aliases:
  - GMFlow+
  - Iterative GMFlow
authors:
  - Xiaoyu Xu
  - Zheyu Zhuang
  - Yifan Liu
  - Chunhua Shen
  - Baosheng Yu
year: 2023
venue: "CVPR"
doi: "10.1109/CVPR52729.2023.01058"
arxiv: "https://arxiv.org/abs/2212.02515"
code: "https://github.com/haofeixu/gmflow"
citations: ~150+
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
  - refinement
fields:
  - vision
  - motion-estimation
related:
  - "[[GMFlow (2022)]]"
  - "[[RAFT (2020)]]"
  - "[[FlowFormer (2022)]]"
predecessors:
  - "[[GMFlow (2022)]]"
successors:
  - "[[FlowFormer++ (2023)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**GMFlow+** improves upon GMFlow by combining **global matching with iterative refinement**. It retains GMFlow’s transformer-based global matching, but introduces an iterative update mechanism to refine correspondences, leading to higher accuracy while remaining efficient.

# Key Idea
> Marry **global transformer-based matching** (GMFlow) with **iterative refinement** (RAFT-style), achieving both robustness to large displacements and fine-grained precision.

# Method
- **Global matching (coarse)**: Transformer encoder-decoder performs dense global feature matching across the two images.  
- **Iterative refinement (fine)**: Multiple refinement stages progressively improve the flow estimate, especially for details and occlusions.  
- **Multi-scale design**: Coarse-to-fine matching allows handling both global motion and local details.  
- **Architecture**: CNN + transformer backbone, with refinement loops over flow predictions.  

# Results
- Achieved **new SOTA or near-SOTA** on Sintel and KITTI in 2023.  
- Outperformed GMFlow significantly, narrowing the gap with FlowFormer++.  
- Strong balance of **speed, accuracy, and robustness**.  

# Why it Mattered
- Demonstrated that **hybrid architectures** (matching + refinement) outperform pure one-shot or pure recurrent methods.  
- Proved global matching benefits from iterative updates, echoing RAFT’s success.  
- Strengthened transformer-based flow estimation’s position as the new paradigm.  

# Architectural Pattern
- Transformer global matching → initial flow.  
- Iterative refinement loop → polished flow.  
- Multi-scale coarse-to-fine integration.  

# Connections
- Direct successor to **GMFlow (2022)**.  
- Hybrid between **RAFT’s iterative refinement** and **GMFlow’s transformer matching**.  
- Predecessor to **FlowFormer++ (2023)**.  

# Implementation Notes
- Training pipeline: Chairs → Things3D → Sintel/KITTI.  
- Multi-scale supervision improves convergence.  
- Released pretrained weights (PyTorch).  

# Critiques / Limitations
- More complex than GMFlow; loses some of the simplicity advantage.  
- Refinement adds inference cost vs GMFlow.  
- Still memory-heavy at very high resolutions.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Optical flow basics.  
- Coarse-to-fine strategies for vision tasks.  
- Why iterative refinement improves predictions.  

## Postgraduate-Level Concepts
- Hybridizing RAFT-style recurrence with transformer-based matching.  
- Trade-offs: accuracy vs latency in flow models.  
- Multi-scale training for dense correspondence.  

---

# My Notes
- GMFlow+ feels like the **middle ground**: global matching (fast, robust) + refinement (precise).  
- A strong balance of theory and practice — like RAFT but modernized with transformers.  
- Open question: Could GMFlow+ serve as the **motion backbone for video diffusion models**, enforcing temporal stability?  
- Possible extension: Use GMFlow+ initialization + diffusion refinement for **video inpainting and editing**.  

---
