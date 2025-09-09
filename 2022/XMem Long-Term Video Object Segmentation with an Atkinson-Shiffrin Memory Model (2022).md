---
title: "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model (2022)"
aliases:
  - XMem
  - X-Memory Networks
authors:
  - Ho Kei Cheng
  - Alexander Schwing
  - Yuxuan Wei
  - Seoung Wug Oh
  - Ning Xu
  - et al.
year: 2022
venue: "ECCV"
doi: "10.48550/arXiv.2207.07115"
arxiv: "https://arxiv.org/abs/2207.07115"
code: "https://github.com/hkchengrex/XMem"
citations: 400+
dataset:
  - DAVIS 2017
  - YouTube-VOS
  - Long-VOS
tags:
  - paper
  - video-object-segmentation
  - segmentation
  - memory-networks
  - recurrent
fields:
  - vision
  - segmentation
  - video-understanding
related:
  - "[[STM (2019)]]"
  - "[[AOT (2021)]]"
predecessors:
  - "[[STM (2019)]]"
successors: []
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**XMem** improved **Space-Time Memory (STM)** by introducing a **hierarchical memory system** inspired by the **Atkinson–Shiffrin cognitive memory model**, enabling **scalable long-term video object segmentation (VOS)**. It maintained high accuracy while drastically reducing memory growth, making long videos tractable.

# Key Idea
> Introduce a **multi-level memory system** (short-term cache + long-term store) to handle long videos efficiently, balancing accuracy and computational cost.

# Method
- **Atkinson–Shiffrin inspired memory**:  
  - **Working memory**: Actively updated with recent frames.  
  - **Long-term memory**: Stores compressed representations of older frames.  
  - **Selection module**: Decides what to keep, update, or discard.  
- **Query mechanism**: Combines both memory types to guide segmentation.  
- **Efficiency**: Keeps memory size sublinear in video length.  

# Results
- State-of-the-art on **DAVIS 2017**, **YouTube-VOS**, and **Long-VOS**.  
- Successfully handled long sequences (>10k frames).  
- Outperformed STM while using far less memory.  

# Why it Mattered
- Made **long-term video object segmentation practical**.  
- Introduced cognitive memory modeling into VOS.  
- Widely adopted as the new baseline for semi-supervised VOS.  

# Architectural Pattern
- CNN/Transformer encoder backbone.  
- Hierarchical memory (short-term + long-term).  
- Space-time attention for segmentation.  

# Connections
- Successor to **STM (2019)**.  
- Complementary to **AOT (2021)** (transformer-based VOS).  
- Influenced efficient memory designs in video transformers.  

# Implementation Notes
- Much more memory-efficient than STM.  
- Open-source PyTorch code widely used.  
- Can scale to real-time applications.  

# Critiques / Limitations
- Complexity in managing memory hierarchy.  
- Still semi-supervised VOS (first-frame mask required).  
- Struggles when appearance changes are extreme without strong memory features.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Why storing **all past frames** isn’t scalable for long videos.  
- Idea of **short-term vs long-term memory**.  
- Real-world analogy: remembering yesterday’s details vs a summary of last month.  
- Example: tracking a person across a 5-minute video.  

## Postgraduate-Level Concepts
- Memory compression and retrieval in neural networks.  
- Attention over multi-level memory banks.  
- Cognitive science inspiration (Atkinson–Shiffrin model) in AI.  
- Extensions: real-time interactive segmentation, multimodal memory.  

---

# My Notes
- XMem = **STM made scalable** → solved the memory explosion problem.  
- Cognitive memory analogy is elegant: short-term vs long-term.  
- Open question: Can this memory architecture generalize to **video understanding tasks beyond segmentation** (e.g., captioning, VQA)?  
- Possible extension: Combine XMem’s efficiency with **diffusion-based video editing** for practical creative tools.  

---
