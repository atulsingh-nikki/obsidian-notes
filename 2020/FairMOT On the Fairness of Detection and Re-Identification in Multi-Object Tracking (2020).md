---
title: "FairMOT: On the Fairness of Detection and Re-Identification in Multi-Object Tracking (2020)"
aliases:
  - FairMOT
authors:
  - Yifu Zhang
  - Chunyu Wang
  - Xinggang Wang
  - Wenjun Zeng
year: 2020
venue: "IJCV (originally preprint 2020)"
doi: "10.1007/s11263-021-01513-4"
arxiv: "https://arxiv.org/abs/2004.01888"
code: "https://github.com/ifzhang/FairMOT"
citations: 3500+
dataset:
  - MOT16
  - MOT17
  - MOT20
  - CrowdHuman
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - detection
  - re-identification
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[CenterTrack (2020)]]"
  - "[[DeepSORT (2017)]]"
  - "[[ByteTrack (2022)]]"
  - "[[BoT-SORT (2023)]]"
predecessors:
  - "[[DeepSORT (2017)]]"
  - "[[CenterTrack (2020)]]"
successors:
  - "[[ByteTrack (2022)]]"
  - "[[OC-SORT (2023)]]"
  - "[[BoT-SORT (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**FairMOT** unified **object detection and re-identification (ReID)** in a **single network** for multi-object tracking (MOT). Unlike prior trackers that optimized detection at the expense of ReID (or vice versa), FairMOT designed a balanced architecture that treated both tasks fairly, leading to state-of-the-art MOT performance.

# Key Idea
> Build a **two-branch network** (detection + ReID) on a **shared backbone**, ensuring fairness between tasks so both detection accuracy and identity preservation are strong.

# Method
- **Backbone**: High-resolution network (HRNet) to maintain spatial detail.  
- **Two branches**:  
  - **Detection branch** → predicts object centers and box sizes.  
  - **ReID branch** → predicts identity embeddings for each detected object.  
- **Fairness design**: Balanced loss and architecture so ReID isn’t suppressed by detection learning.  
- **Training**: Jointly optimize detection loss + ReID embedding loss.  

# Results
- Achieved **state-of-the-art MOT accuracy** on MOT16, MOT17, MOT20.  
- Strong identity preservation with low ID switches.  
- Outperformed both detection-only (ByteTrack-style) and appearance-only trackers.  

# Why it Mattered
- First truly **fair joint detection + ReID tracker**.  
- Unified pipeline replaced two-stage “detector + separate ReID model” setups.  
- Widely adopted in research and autonomous driving.  

# Architectural Pattern
- Shared HRNet backbone.  
- Dual branches: detection + ReID embedding.  

# Connections
- Built upon **DeepSORT** (added ReID, but separate).  
- Contemporary of **CenterTrack** (joint detection + motion).  
- Predecessor to **ByteTrack** (2022) and **BoT-SORT** (2023).  

# Implementation Notes
- HRNet backbone computationally heavier than SORT-based trackers.  
- Requires ReID dataset for training embeddings.  
- Official PyTorch implementation widely used.  

# Critiques / Limitations
- More complex training pipeline than ByteTrack.  
- ReID branch adds computational cost.  
- Struggles in extreme occlusion despite improvements.  

---

# Educational Connections

## Undergraduate-Level Concepts
- MOT basics: detection + identity tracking.  
- What re-identification (ReID) is and why it matters.  
- Difference between detection-only vs detection + ReID trackers.  
- Role of multi-branch networks in handling multiple tasks.  

## Postgraduate-Level Concepts
- Balancing multi-task learning (fairness in optimization).  
- HRNet backbone benefits for spatial tasks.  
- Trade-offs in **joint vs separate ReID models**.  
- Extensions of FairMOT principles to transformer-based MOT.  

---

# My Notes
- FairMOT = **DeepSORT made end-to-end**.  
- Very influential in unifying detection + ReID under one roof.  
- It shows MOT evolving along two branches: **CenterTrack (motion focus)** and **FairMOT (appearance focus)**, later merged by **ByteTrack/BoT-SORT**.  
- Open question: Can **transformer MOT models** inherit FairMOT’s fairness principle for multi-task learning?  

---
