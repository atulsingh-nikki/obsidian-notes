---
title: "Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling (2021)"
aliases:
  - ClipBERT
  - Sparse Sampling for Video-Language
authors:
  - Jie Lei
  - Linjie Li
  - Luowei Zhou
  - Zhe Gan
  - Tamara L. Berg
  - Mohit Bansal
  - Jingjing Liu
year: 2021
venue: "CVPR (Best Paper Honorable Mention)"
doi: "10.1109/CVPR46437.2021.00456"
arxiv: "https://arxiv.org/abs/2102.06183"
code: "https://github.com/jayleicn/ClipBERT"
citations: 800+
dataset:
  - MSR-VTT
  - ActivityNet Captions
  - COCO Captions (pretraining)
tags:
  - paper
  - video-language
  - transformer
  - sparse-sampling
  - efficient-learning
fields:
  - vision
  - language
  - multimodal
related:
  - "[[VideoBERT (2019)]]"
  - "[[ViLBERT (2019)]]"
  - "[[TimeSformer (2021)]]"
predecessors:
  - "[[VideoBERT (2019)]]"
  - "[[ViLBERT (2019)]]"
successors:
  - "[[Video-Language Pretraining Models (2022+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**ClipBERT** proposed a more **efficient video-and-language learning framework** that uses **sparse temporal sampling** of video frames. Instead of processing full video sequences, it demonstrated that **end-to-end training on only a few sampled frames** is highly effective for tasks like video question answering and video retrieval.

# Key Idea
> Sparse temporal sampling (a few frames per video) + end-to-end multimodal transformer training is sufficient for strong video-language performance, improving efficiency without losing accuracy.

# Method
- **Sparse frame sampling**: Sample a small set of frames (e.g., 8–16) instead of dense sequences.  
- **Frame-level encoding**: Use CNN/transformer backbone for image features.  
- **Multimodal fusion**: Jointly encode text (BERT) and visual features.  
- **End-to-end training**: Train vision and language encoders together rather than freezing vision backbone.  
- **Applications**: Video QA, retrieval, captioning.  

# Results
- Achieved SOTA or competitive results on video QA and retrieval benchmarks.  
- Reduced computation cost drastically compared to dense sampling models.  
- Showed scalability of sparse sampling to large datasets.  

# Why it Mattered
- Challenged assumption that dense temporal modeling is always necessary.  
- Made **video-language models more practical** by reducing GPU cost.  
- Opened the door for efficient pretraining strategies in multimodal learning.  

# Architectural Pattern
- Sparse frame sampling → visual encoder → multimodal transformer.  
- End-to-end optimization of visual + language features.  

# Connections
- Builds on **VideoBERT/ViLBERT** multimodal transformers.  
- Predecessor to **efficient video-language models** (2022+).  
- Complementary to video transformers like **TimeSformer**.  

# Implementation Notes
- Works surprisingly well with very few frames.  
- Efficiency-critical: good for resource-limited training.  
- Open-source implementation widely used.  

# Critiques / Limitations
- Temporal fine-grained details may be missed (good for coarse tasks, weaker for fine motion).  
- Not directly optimized for very long videos.  
- Sparse frame strategy may not capture rare events.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Why video tasks are expensive compared to image tasks.  
- Idea of sampling: using fewer frames to reduce computation.  
- Basics of multimodal learning: combining text and images.  
- Difference between frame-level vs sequence-level processing.  

## Postgraduate-Level Concepts
- Sparse sampling as a design trade-off in video-language modeling.  
- End-to-end training vs feature freezing in multimodal setups.  
- Comparison to dense temporal transformers (e.g., TimeSformer).  
- Implications for scaling to internet-scale video-text datasets.  

---

# My Notes
- ClipBERT’s message is simple but powerful: **less is more**.  
- Proved that expensive dense temporal encoders aren’t always necessary.  
- Open question: Can sparse sampling ideas extend to **long video reasoning** (minutes/hours)?  
- Possible extension: Combine sparse ClipBERT backbone with **video diffusion models** for efficient generative video understanding.  

---
