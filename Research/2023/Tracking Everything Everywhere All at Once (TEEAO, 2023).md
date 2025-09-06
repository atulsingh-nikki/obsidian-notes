---
title: Tracking Everything Everywhere All at Once (TEEAO, 2023)
aliases:
  - TEEAO
  - Tracking Everything
authors:
  - Gedas Bertasius
  - Heng Wang
  - Deva Ramanan
  - et al.
year: 2023
venue: arXiv
doi: 10.48550/arXiv.2306.05422
arxiv: https://arxiv.org/abs/2306.05422
code: https://github.com/facebookresearch/track-anything
citations: 250+
dataset:
  - COCO
  - DAVIS
  - YouTube-VOS
  - LaSOT
  - TAO
tags:
  - paper
  - tracking
  - multi-object-tracking
  - video-object-segmentation
  - foundation-model
fields:
  - vision
  - tracking
  - segmentation
  - video-understanding
related:
  - "[[Segment Anything (SAM, 2023)]]"
  - "[[SAM 2 Segment Anything in Images and Videos (2024)|SAM 2]]"
  - "[[XMem Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model (2022)|XMem]]"
  - "[[RDeAOT Recurrent Decoupling for Efficient Video Object Segmentation (2023)|RDeAOT (2023)]]"
predecessors:
  - "[[Segment Anything (SAM, 2023)]]"
  - "[[Generic Object Tracking Benchmarks]]"
successors: []
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**Tracking Everything Everywhere All at Once (TEEAO)** proposed a **unified foundation model for tracking** in videos, aiming to handle **any object, any domain, and any task** (MOT, VOS, object tracking, etc.). It generalized tracking the way **SAM generalized segmentation**.

# Key Idea
> Build a **task-agnostic tracking model** that can “track anything” given a prompt (e.g., bounding box, mask, point), inspired by the **Segment Anything Model (SAM)** philosophy.

# Method
- **Promptable tracking**: Input = points, boxes, or masks.  
- **Backbone**: Transformer-based encoder (ViT).  
- **Tracking head**: Predicts object positions/masks across frames.  
- **Training**: Jointly trained on heterogeneous datasets (tracking, segmentation, detection).  
- **Generalization**: Learns to track without task-specific tuning.  

# Results
- Worked across **multiple tracking benchmarks** (DAVIS, YouTube-VOS, LaSOT, TAO).  
- Outperformed specialized models in **cross-dataset generalization**.  
- First serious attempt at a **tracking foundation model**.  

# Why it Mattered
- Shift from task-specific trackers (MOT, VOS, single-object tracking) → **universal tracker**.  
- Analogous to SAM in segmentation, but for **temporal consistency**.  
- Foundation approach → train once, apply everywhere.  

# Architectural Pattern
- ViT-based backbone.  
- Prompt encoder.  
- Temporal transformer for tracking.  

# Connections
- Inspired by **SAM (2023)**.  
- Parallel to **SAM 2 (2024)** in extending foundation models to video.  
- Related to VOS baselines (STM, XMem, RDeAOT) but broader in scope.  

# Implementation Notes
- Large training mixture required (multi-task training).  
- Handles **MOT + VOS + generic tracking** with the same model.  
- Open-source code and models available.  

# Critiques / Limitations
- Still less accurate than specialized trackers in their own domains.  
- Computationally heavy (ViT backbone).  
- Prompt design critical — not fully language-promptable yet.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What “tracking” means: following an object over time.  
- Why prompts (box, point, mask) help specify what to track.  
- Example: click a dog in frame 1 → TEEAO tracks it across the video.  

## Postgraduate-Level Concepts
- Foundation model training: merging datasets from different tasks.  
- Temporal transformers for video tracking.  
- Generalization vs specialization trade-offs.  
- Potential for multimodal (vision + language) promptable tracking.  

---

# My Notes
- TEEAO = **SAM for tracking**.  
- Big conceptual leap: one model, many tracking tasks.  
- Open question: Will foundation trackers **replace MOT/VOS specialists** or just complement them?  
- Possible extension: Add **language prompts** → “track the red car” → foundation-level VOS/MOT.  

---
