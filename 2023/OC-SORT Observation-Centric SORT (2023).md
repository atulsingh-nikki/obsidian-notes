---
title: "OC-SORT: Observation-Centric SORT (2023)"
aliases:
  - OC-SORT
  - Observation-Centric SORT
authors:
  - Jiarui Cao
  - Xinshuo Weng
  - Rawal Khirodkar
  - Andreas Geiger
  - Kris Kitani
year: 2023
venue: CVPR
doi: 10.48550/arXiv.2203.14360
arxiv: https://arxiv.org/abs/2203.14360
code: https://github.com/noahcao/OC_SORT
citations: 400+
dataset:
  - MOT17
  - MOT20
  - BDD100K
  - DanceTrack
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - motion-modeling
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[DeepSORT (2017)]]"
  - "[[BoT-SORT (2023)]]"
  - "[[ByteTrack Multi-Object Tracking By Associating Every Detection Box (2022)|ByteTrack MOT]]"
predecessors:
  - "[[SORT (2016)]]"
  - "[[ByteTrack (2022)]]"
successors:
  - "[[Next-Gen MOT Transformers (2024+)]]"
impact: ⭐⭐⭐⭐☆
status: read
---

# Summary
**OC-SORT** proposed a new observation-centric update for **multi-object tracking (MOT)** that improved robustness in **occlusion-heavy and crowded scenes**. Unlike ByteTrack, which primarily leveraged low-confidence detections, OC-SORT focused on **refining motion prediction and association** using observation cues.

# Key Idea
> Replace the traditional **state-centric Kalman filter update** in SORT with an **observation-centric update**, making tracking more robust to mismatches during occlusions and irregular motions.

# Method
- **Tracking base**: Builds on SORT/ByteTrack pipeline.  
- **Observation-centric update**:  
  - Instead of relying only on motion prediction, integrates **new observations** into the state update more directly.  
- **Motion refinement**: Better handling of velocity changes, abrupt direction shifts.  
- **Association**: Still uses high- and low-confidence detections, but pairing is stabilized by improved motion estimation.  

# Results
- Outperformed ByteTrack in **occlusion-heavy benchmarks** like DanceTrack.  
- Competitive performance on MOT17, MOT20, and BDD100K.  
- More stable long-term tracking under crowded conditions.  

# Why it Mattered
- Addressed one of ByteTrack’s weaknesses: struggles under **frequent occlusion**.  
- Introduced a principled motion refinement strategy, keeping MOT simple yet robust.  
- Became a **new strong baseline** for MOT research.  

# Architectural Pattern
- Detector → OC-SORT association.  
- Observation-centric state update instead of state-centric.  

# Connections
- Successor to **ByteTrack** (2022).  
- Related to **BoT-SORT (2023)** (appearance + motion fusion).  
- Competitor to transformer-based trackers but lighter and faster.  

# Implementation Notes
- Plug-and-play replacement for SORT/ByteTrack.  
- Still real-time capable.  
- Public PyTorch implementation available.  

# Critiques / Limitations
- Still relies on Kalman filter backbone (hand-crafted).  
- No deep appearance embedding like DeepSORT or BoT-SORT.  
- May underperform in identity switches in extremely crowded scenes.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Basics of **multi-object tracking (MOT)**.  
- What occlusion is and why it breaks tracking.  
- Difference between motion prediction and observation updates.  
- Why refining state updates makes trackers more robust.  

## Postgraduate-Level Concepts
- Observation-centric vs state-centric filtering in tracking.  
- Trade-offs: simplicity (OC-SORT) vs complexity (transformer-based trackers).  
- Handling long-term occlusions in MOT benchmarks.  
- Implications for autonomous driving and surveillance systems.  

---

# My Notes
- OC-SORT is a **natural evolution of ByteTrack**: fixing motion/occlusion weaknesses.  
- Feels like the last big step before **transformer-based end-to-end MOT** takes over.  
- Open question: Can OC-SORT principles be **merged with appearance embeddings** (BoT-SORT) for best of both worlds?  
- Possible extension: Apply OC-SORT to **3D MOT (LiDAR + camera fusion)** for autonomous driving.  

---
