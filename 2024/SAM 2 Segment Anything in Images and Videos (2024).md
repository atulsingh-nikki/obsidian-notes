---
title: "SAM 2: Segment Anything in Images and Videos (2024)"
aliases:
  - SAM 2
  - Segment Anything Model 2
authors:
  - Meta AI Research (FAIR team)
  - Alexander Kirillov
  - Eric Mintun
  - Nikhila Ravi
  - Piotr Dollar
  - Ross Girshick
  - et al.
year: 2024
venue: arXiv
doi: 10.48550/arXiv.2408.00762
arxiv: https://arxiv.org/abs/2408.00762
code: https://ai.meta.com/research/publications/sam-2/
citations: 200+
dataset:
  - SA-V (Segment Anything Video dataset, 51M masks)
tags:
  - paper
  - segmentation
  - foundation-model
  - vision
  - video-object-segmentation
  - memory-networks
fields:
  - vision
  - segmentation
  - video-understanding
related:
  - "[[Segment Anything (SAM, 2023)]]"
  - "[[Video Object Segmentation using Space-Time Memory Networks (STM, 2019)|Space-Time Memory Networks]]"
  - "[[XMem Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model (2022)|XMem]]"
predecessors:
  - "[[Segment Anything (SAM, 2023)]]"
successors: []
impact: ⭐⭐⭐⭐⭐
status: reading
---

# Summary
**SAM 2** extends the **Segment Anything Model (SAM)** to handle **both images and videos**. It introduces **memory mechanisms** for video object segmentation (VOS), unifying **image segmentation, video tracking, and interactive segmentation** under one foundation model.

# Key Idea
> Add a **memory system** to SAM’s promptable segmentation framework, enabling the model to propagate object masks across video frames with interactive editing.

# Method
- **Architecture**:  
  - ViT-based encoder (like SAM).  
  - Prompt encoder (points, boxes, text).  
  - Mask decoder.  
  - **Memory module** for video: stores past frame features + masks.  
- **Dataset**: SA-V — large-scale video segmentation dataset with 51M masks.  
- **Interface**: Still promptable (points, boxes), extended to video.  

# Results
- Strong performance on **DAVIS 2017**, **YouTube-VOS**, and long VOS datasets.  
- Outperformed STM/XMem/RDeAOT in zero-shot generalization.  
- First **foundation-level VOS model** trained at web scale.  

# Why it Mattered
- Brought **foundation models to video segmentation**, not just images.  
- Unified static and temporal segmentation under one interface.  
- Marked a step toward **multimodal foundation vision models** (image + video + text prompts).  

# Architectural Pattern
- SAM encoder–decoder.  
- Extended with memory retrieval for video frames.  

# Connections
- Successor to **SAM (2023)**.  
- Related to **STM (2019)**, **XMem (2022)**, **RDeAOT (2023)** (VOS baselines).  
- Potential bridge to multimodal editing systems.  

# Implementation Notes
- Requires large compute for training.  
- Open-sourced with pretrained models and dataset (SA-V).  
- Usable for interactive segmentation in images + videos.  

# Critiques / Limitations
- Heavy model (ViT-H backbone).  
- Annotation quality of SA-V not as precise as DAVIS/YouTube-VOS.  
- Focused on generalization, not task-specific fine-grained accuracy.  

---

# Educational Connections

## Undergraduate-Level Concepts
- SAM was for images; SAM 2 adds **videos with memory**.  
- What “memory” means in segmentation: reusing past masks to guide new frames.  
- Example: selecting a person in the first frame → SAM 2 tracks them across the video.  

## Postgraduate-Level Concepts
- Memory-augmented transformers for VOS.  
- Dataset scaling: SA-1B (images) → SA-V (videos).  
- SAM 2 vs STM/XMem/RDeAOT: generalist foundation vs task-specific specialists.  
- Implications for **interactive video editing tools**.  

---

# My Notes
- SAM 2 = **the “foundation model” version of STM/XMem**.  
- Big shift: one model for **both image and video segmentation**.  
- Open question: Will SAM 2 replace specialized VOS models in practice, or will **lightweight task-specific methods still dominate**?  
- Possible extension: SAM 2 + diffusion → **promptable generative video editing**.  

---
