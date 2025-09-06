---
title: "Segment Anything (SAM, 2023)"
aliases:
  - SAM
  - Segment Anything Model
authors:
  - Alexander Kirillov
  - Eric Mintun
  - Nikhila Ravi
  - Hanzi Mao
  - Chloe Rolland
  - Laura Gustafson
  - Tete Xiao
  - Spencer Whitehead
  - Alexander Berg
  - Wan-Yen Lo
  - Piotr Dollar
  - Ross Girshick
year: 2023
venue: "arXiv / Meta AI Research"
doi: "10.48550/arXiv.2304.02643"
arxiv: "https://arxiv.org/abs/2304.02643"
code: "https://segment-anything.com/"
citations: 8000+
dataset:
  - SA-1B (11M images, 1.1B masks)
tags:
  - paper
  - segmentation
  - foundation-model
  - vision
fields:
  - vision
  - segmentation
  - foundation-models
related:
  - "[[Mask R-CNN (2017)]]"
  - "[[DETR (2020)]]"
  - "[[XMem (2022)]]"
  - "[[RDeAOT (2023)]]"
predecessors:
  - "[[Mask R-CNN (2017)]]"
  - "[[Vision Transformers (2020)]]"
successors:
  - "[[SAM 2 (2024)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Segment Anything (SAM)** is a **foundation model for image segmentation**, trained on the massive **SA-1B dataset** (11M images, 1.1B masks). It can **segment any object in any image with zero-shot prompts**, marking a paradigm shift toward foundation models in vision.

# Key Idea
> Train a **promptable segmentation model** on a **web-scale dataset**, enabling zero-shot generalization to unseen objects and domains.

# Method
- **Backbone**: Vision Transformer (ViT-Huge).  
- **Promptable interface**:  
  - Input prompts = points, boxes, or text (later extensions).  
  - Model outputs masks aligned to prompt.  
- **Training dataset**: SA-1B, largest segmentation dataset ever created.  
- **Outputs**: High-quality masks, often at interactive speeds.  

# Results
- Strong **zero-shot transfer** across domains (medical, scientific, natural images).  
- Outperformed task-specific segmentation models on unseen tasks.  
- Enabled **interactive segmentation tools** with high usability.  

# Why it Mattered
- First **segmentation foundation model**.  
- Dataset scale (SA-1B) unprecedented in segmentation.  
- Kickstarted a wave of foundation-style segmentation research (e.g., SAM 2, MedSAM).  

# Architectural Pattern
- ViT-based encoder.  
- Prompt encoder + mask decoder.  
- Unified segmentation interface via prompts.  

# Connections
- Successor to **Mask R-CNN (2017)** and **DETR (2020)** in segmentation lineage.  
- Complementary to **VOS models** like STM, XMem, RDeAOT.  
- Successor: **SAM 2 (2024)** added video segmentation + memory.  

# Implementation Notes
- Open-sourced model + dataset.  
- Runs efficiently on GPU; interactive speeds possible.  
- Used as a pre-processing tool in pipelines (labeling, mask generation).  

# Critiques / Limitations
- Struggles with **fine boundaries** (e.g., hair, transparency).  
- Requires **powerful hardware** (ViT-Huge).  
- Dataset biases from internet images.  
- Not directly optimized for **video segmentation** (addressed in SAM 2).  

---

# Educational Connections

## Undergraduate-Level Concepts
- What segmentation means (pixels → objects).  
- Why prompts (points/boxes) help guide segmentation.  
- How a giant dataset can make a model more general.  
- Example: clicking on a cat in an image → SAM gives the mask instantly.  

## Postgraduate-Level Concepts
- Promptable interfaces in vision foundation models.  
- Dataset scale vs annotation quality trade-offs.  
- SAM vs task-specific segmentation: generalization vs specialization.  
- Foundation models unifying vision tasks (classification, detection, segmentation).  

---

# My Notes
- SAM = **“GPT moment” for segmentation**.  
- Proved that segmentation can scale to **foundation model status**.  
- Open question: How to extend SAM to **video and multimodal tasks** robustly? (→ SAM 2, VOS + language).  
- Possible extension: SAM as the **front-end for generative editing pipelines**.  

---
