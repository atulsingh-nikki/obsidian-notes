---
title: "DeAOT: Decoupling Object Association and Propagation for Transformer-based Video Object Segmentation (2022)"
aliases:
  - DeAOT
authors:
  - Zongxin Yang
  - Yunchao Wei
  - Yi Yang
year: 2022
venue: "ECCV"
doi: "10.48550/arXiv.2207.03318"
arxiv: "https://arxiv.org/abs/2207.03318"
code: "https://github.com/z-x-yang/DeAOT"
citations: 200+
dataset:
  - DAVIS 2017
  - YouTube-VOS
  - Long-VOS
tags:
  - paper
  - video-object-segmentation
  - segmentation
  - transformers
fields:
  - vision
  - segmentation
  - video-understanding
related:
  - "[[AOT (2021)]]"
  - "[[STM (2019)]]"
  - "[[XMem (2022)]]"
predecessors:
  - "[[AOT (2021)]]"
successors:
  - "[[RDeAOT (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**DeAOT** improved upon **AOT** by **decoupling object association and propagation** within the transformer framework, making video object segmentation (VOS) significantly more **efficient and scalable** for long videos and many objects.

# Key Idea
> Split the transformer into two roles:  
> - **Association module**: Uses identity tokens to link objects across frames.  
> - **Propagation module**: Updates segmentation masks efficiently.  
This separation reduces complexity while preserving accuracy.

# Method
- **Decoupled design**:  
  - **Association transformer**: Maintains identity consistency with tokens.  
  - **Propagation module**: Efficiently updates segmentation masks frame by frame.  
- **Hierarchical processing**: Allows scaling to long videos and multiple objects.  
- **Training**: End-to-end with standard VOS datasets.  

# Results
- Outperformed AOT on **DAVIS 2017**, **YouTube-VOS**, and **Long-VOS**.  
- More efficient and scalable to longer videos.  
- Reduced computational cost while improving robustness.  

# Why it Mattered
- Solved efficiency bottlenecks of AOT.  
- Made transformer-based VOS practical for real-world use.  
- Extended token-based VOS into long-sequence scenarios.  

# Architectural Pattern
- Transformer with **decoupled association + propagation**.  
- Identity tokens maintained, but propagation handled separately.  

# Connections
- Successor to **AOT (2021)**.  
- Complementary to **XMem (2022)** (memory efficiency).  
- Predecessor to **RDeAOT (2023)** (refined decoupling).  

# Implementation Notes
- Runs faster than AOT, closer to STM/XMem efficiency.  
- Supports multiple objects naturally.  
- PyTorch code and pretrained models available.  

# Critiques / Limitations
- Still semi-supervised (first-frame mask required).  
- Token-based approach may still bottleneck with very large object counts.  
- Less biologically intuitive than XMem’s memory design.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Review: transformers in video segmentation.  
- Why splitting a problem (association vs propagation) makes things simpler.  
- Example: assigning IDs to objects separately, then updating their masks.  

## Postgraduate-Level Concepts
- Decoupling design in deep learning architectures.  
- Identity tokens for association vs propagation-specific modules.  
- Comparison: STM (memory) vs AOT (tokens) vs DeAOT (decoupled tokens).  
- Extensions: handling real-time, interactive VOS.  

---

# My Notes
- DeAOT = **AOT made efficient**.  
- The decoupling insight is elegant: don’t overload the transformer with both roles.  
- Open question: Can decoupling be extended to **multimodal segmentation (video + text + audio)**?  
- Possible extension: **RDeAOT** pushes this further, worth adding next.  

---
