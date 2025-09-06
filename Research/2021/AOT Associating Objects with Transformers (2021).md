---
title: "AOT: Associating Objects with Transformers (2021)"
aliases:
  - AOT
  - Associating Objects with Transformers
authors:
  - Zongxin Yang
  - Yunchao Wei
  - Yi Yang
year: 2021
venue: ICCV
doi: 10.1109/ICCV48922.2021.00660
arxiv: https://arxiv.org/abs/2106.02638
code: https://github.com/z-x-yang/AOT
citations: 700+
dataset:
  - DAVIS 2017
  - YouTube-VOS
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
  - "[[Video Object Segmentation using Space-Time Memory Networks (STM, 2019)|Space-Time Memory Networks]]"
  - "[[XMem Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model (2022)|XMem]]"
predecessors:
  - "[[Video Object Segmentation using Space-Time Memory Networks (STM, 2019)|Space-Time Memory Networks]]"
successors:
  - "[[DeAOT (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**AOT** introduced a **transformer-based framework** for semi-supervised Video Object Segmentation (VOS). It pioneered the use of **object-specific tokens** within a transformer to maintain **instance-level identity** across frames, enabling robust multi-object tracking and segmentation.

# Key Idea
> Assign each object an **identity token** in a transformer, and let attention propagate features and identity information across frames for robust multi-object segmentation.

# Method
- **Identity tokens**:  
  - Each tracked object assigned a unique token.  
  - Tokens carry object identity across frames.  
- **Transformer backbone**: Processes video frames and identity tokens jointly.  
- **Association**: Tokens link to pixels via attention, enabling identity-consistent segmentation.  
- **Training**: Supervised with ground-truth masks from DAVIS/YTVOS.  

# Results
- Outperformed STM on **DAVIS 2017** and **YouTube-VOS**.  
- Strong multi-object segmentation performance.  
- More robust to occlusions and re-appearance than memory-only models.  

# Why it Mattered
- First major **transformer-based VOS** method.  
- Introduced the idea of **identity tokens**, later widely adopted.  
- Marked the shift from memory networks (STM) → token-based transformers.  

# Architectural Pattern
- Transformer encoder–decoder.  
- Identity tokens for each object.  
- Pixel–token attention for segmentation.  

# Connections
- Successor to **STM (2019)** (memory-based).  
- Complementary to **XMem (2022)** (efficient memory).  
- Successor: **DeAOT (2022)** (more efficient, scalable transformer design).  

# Implementation Notes
- Transformer-based, heavier than CNN memory models.  
- Handles multiple objects naturally via tokens.  
- PyTorch implementation open-sourced.  

# Critiques / Limitations
- Computationally expensive compared to STM/XMem.  
- Token capacity limits number of objects.  
- Still semi-supervised (first-frame mask required).  

---

# Educational Connections

## Undergraduate-Level Concepts
- What a transformer is and how it uses attention.  
- Idea of assigning each object a unique “token” to track it.  
- Why tokens help distinguish objects even if they look similar.  
- Example: tracking two dogs of the same color in a video.  

## Postgraduate-Level Concepts
- Token-based object representation in transformers.  
- Attention as implicit association between tokens and pixels.  
- Comparison: STM (memory-based) vs AOT (token-based).  
- Extensions: DeAOT (efficient), fast token updating, multimodal tokens.  

---

# My Notes
- AOT was the **transformer revolution for VOS**: tokens instead of explicit memory.  
- Identity tokens elegantly solved **multi-object consistency**.  
- Open question: Can AOT-scale token approaches handle **very long videos** without hierarchical memory like XMem?  
- Possible extension: Combine **AOT’s tokens** with **XMem’s efficient memory** for the best of both worlds.  

---
