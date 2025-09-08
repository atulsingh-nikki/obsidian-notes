---
title: Real-Time High-Resolution Background Matting (2021)
aliases:
  - Background Matting v2
  - Real-Time Background Matting
authors:
  - Shanchuan Lin
  - Andrey Ryabtsev
  - Soumyadip Sengupta
  - Brian Curless
  - Steve Seitz
  - Ira Kemelmacher-Shlizerman
year: 2021
venue: CVPR (Best Paper Honorable Mention)
doi: 10.1109/CVPR46437.2021.00346
arxiv: https://arxiv.org/abs/2012.07810
code: https://github.com/PeterL1n/RobustVideoMatting
citations: 1100+
dataset:
  - VideoMatte240K
  - High-res matting benchmarks
tags:
  - paper
  - matting
  - background-replacement
  - real-time
  - high-resolution
fields:
  - vision
  - graphics
  - video-processing
related:
  - "[[Deep Image Matting (2017)]]"
  - "[[MODNet Mobile Portrait Matting (2020)|MODNet (2020)]]"
  - "[[Robust Video Matting (RVM, 2021)|Robust Video Matting (2021)]]"
predecessors:
  - "[[Deep Image Matting (2017)]]"
  - "[[MODNet Mobile Portrait Matting (2020)|MODNet (2020)]]"
successors:
  - "[[Robust Video Matting (RVM, 2021)|Robust Video Matting (2021)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**Real-Time High-Resolution Background Matting** presented a **lightweight method for high-quality background replacement** in real-time. It required a single auxiliary background image to guide matting and achieved **state-of-the-art accuracy** at high resolutions while remaining efficient enough for live video.

# Key Idea
> Use an **auxiliary clean background image** plus a lightweight network to produce high-resolution alpha mattes and foregrounds in real time.

# Method
- **Auxiliary background capture**: Single static background photo provided.  
- **Inputs**: Foreground frame + background reference.  
- **Network**: Lightweight encoder–decoder that predicts alpha matte + foreground image.  
- **Training**: Large-scale synthetic dataset (VideoMatte240K) with ground-truth mattes.  
- **Efficiency**: Designed for real-time inference on high-res video.  

# Results
- State-of-the-art performance on matting benchmarks.  
- Ran in real time at HD/4K resolutions.  
- Produced high-fidelity alpha mattes suitable for video conferencing, film, AR.  

# Why it Mattered
- First to combine **real-time performance** with **high-resolution matting quality**.  
- Practical applications in video conferencing (Zoom, Teams), AR/VR, and filmmaking.  
- Public dataset + code significantly advanced matting research.  

# Architectural Pattern
- Guidance from clean background.  
- Encoder–decoder matting network.  
- Foreground refinement module.  

# Connections
- Builds on **Deep Image Matting (2017)** and **MODNet (2020)**.  
- Predecessor to **Robust Video Matting (2021)** (no background required).  
- Related to segmentation but more fine-grained (hair, edges).  

# Implementation Notes
- Needs a clean background capture once before use.  
- Lightweight and deployable for live apps.  
- Dataset and pretrained models made available.  

# Critiques / Limitations
- Assumes access to a clean background capture.  
- Struggles if background is not static or changes.  
- Later models (Robust Video Matting) removed this requirement.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What is **image matting** and why it’s harder than segmentation (requires per-pixel transparency).  
- Alpha matte: the transparency layer separating foreground and background.  
- Why having a **clean background image** makes matting easier.  
- Applications: background replacement in Zoom/Teams, green-screen removal.  

## Postgraduate-Level Concepts
- Encoder–decoder networks for matting.  
- Training with large synthetic datasets (VideoMatte240K).  
- Differences between **matting vs segmentation** in detail (hair, fine edges).  
- Evolution from background-assisted to background-free real-time matting.  

---

# My Notes
- This work was **hugely practical** — directly usable in consumer video apps.  
- Auxiliary background trick = simple but powerful.  
- Open question: How to keep quality while removing the need for reference backgrounds? (solved partly by Robust Video Matting).  
- Possible extension: Combine matting with **real-time relighting & depth** for AR/VR compositing.  

---
