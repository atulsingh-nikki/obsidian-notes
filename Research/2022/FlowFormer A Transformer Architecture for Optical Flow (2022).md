---
title: "FlowFormer: A Transformer Architecture for Optical Flow (2022)"
aliases:
  - FlowFormer
  - Transformer Optical Flow
authors:
  - Zhicheng Huang
  - Jiahui Zhang
  - Liang Pan
  - Haiyu Zhao
  - Shuai Yi
  - Hongsheng Li
  - Xizhou Zhu
year: 2022
venue: "ECCV"
doi: "10.1007/978-3-031-20065-6_6"
arxiv: "https://arxiv.org/abs/2203.16194"
code: "https://github.com/drinkingcoder/FlowFormer-Official"
citations: ~400+
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
fields:
  - vision
  - motion-estimation
  - dense-prediction
related:
  - "[[RAFT (2020)]]"
  - "[[GMA (2021)]]"
  - "[[GMFlow (2022)]]"
predecessors:
  - "[[GMA (2021)]]"
successors:
  - "[[GMFlow (2022)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**FlowFormer** introduced the first **pure transformer-based architecture for optical flow estimation**. Instead of RAFT’s recurrent GRU refinement or GMA’s attention modules, FlowFormer uses a **transformer-based cost volume encoder** that captures both global motion context and local details, setting new SOTA on multiple benchmarks.

# Key Idea
> Replace recurrent refinement with a **transformer encoder-decoder over the correlation volume**, allowing global context aggregation in a feedforward fashion.

# Method
- **Cost Volume Encoder (CVE)**:  
  - Constructs a multi-scale cost volume from feature maps.  
  - Processes it with a transformer encoder to model global dependencies.  
- **Memory tokens**: Store motion context and propagate it through transformer layers.  
- **Decoder**: Predicts flow iteratively from encoded cost features.  
- **Training**: Standard multi-stage schedule (Chairs → Things3D → Sintel/KITTI).  

# Results
- Achieved **new SOTA** on Sintel (clean/final) and KITTI at the time.  
- Outperformed RAFT and GMA with better accuracy and competitive runtime.  
- Demonstrated transformers’ strength in dense correspondence tasks.  

# Why it Mattered
- Marked the shift from RAFT-style recurrent architectures to **transformer-based flow estimation**.  
- Proved transformers are effective for **dense motion tasks**, not just classification or detection.  
- Opened path to later works like GMFlow and FlowFormer++ with more efficient transformer designs.  

# Architectural Pattern
- CNN/transformer hybrid feature extractor.  
- Transformer encoder over cost volume (global + local aggregation).  
- Feedforward decoder for flow prediction.  

# Connections
- Builds on RAFT’s all-pairs correlation idea.  
- Extends GMA’s attention with a full transformer backbone.  
- Precedes GMFlow and FlowFormer++ refinements.  

# Implementation Notes
- Requires more compute than RAFT but scales better with large datasets.  
- Memory tokens critical for stable training.  
- Released official code with pretrained weights.  

# Critiques / Limitations
- Higher compute/memory cost than RAFT for high-res inputs.  
- Training stability depends on careful token/memory design.  
- Real-time applications still favor lighter models.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Optical flow basics.  
- Cost volume as a similarity representation.  
- Transformers for sequence modeling.  

## Postgraduate-Level Concepts
- Transformer encoders applied to dense prediction.  
- Memory tokens for propagating context.  
- Comparison of recurrence vs transformers in motion estimation.  

---

# My Notes
- FlowFormer feels like the **ViT moment for optical flow**: transformers step in, outperform recurrences.  
- Strong candidate backbone for **video consistency in editing/diffusion**.  
- Open question: Can FlowFormer scale to **long video clips** with hierarchical temporal attention?  
- Possible extension: Integrate FlowFormer into **video foundation models** as the motion reasoning module.  

---
