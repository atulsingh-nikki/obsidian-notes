---
title: "SORT: Simple Online and Realtime Tracking (2016)"
aliases:
  - SORT
  - Simple Online and Realtime Tracking
authors:
  - Alex Bewley
  - Zongyuan Ge
  - Lionel Ott
  - Fabio Ramos
  - Ben Upcroft
year: 2016
venue: "ICIP"
doi: "10.1109/ICIP.2016.7533003"
arxiv: "https://arxiv.org/abs/1602.00763"
code: "https://github.com/abewley/sort"
citations: 6000+
dataset:
  - MOT16
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - detection
  - kalman-filter
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[DeepSORT (2017)]]"
  - "[[ByteTrack (2022)]]"
  - "[[OC-SORT (2023)]]"
  - "[[BoT-SORT (2023)]]"
predecessors: []
successors:
  - "[[DeepSORT (2017)]]"
  - "[[ByteTrack (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**SORT** introduced a **simple, real-time multi-object tracker** based on combining **Kalman filter motion prediction** with **Hungarian assignment** for data association. It relied purely on bounding box detections and motion cues, making it lightweight and fast, though prone to ID switches in crowded scenes.

# Key Idea
> MOT can be done in real-time with a minimal pipeline: detection + Kalman filter + Hungarian matching. Simplicity enables scalability and deployment.

# Method
- **Detections**: Uses bounding boxes from an object detector.  
- **Motion model**: Kalman filter predicts each object’s next state (position, velocity).  
- **Association**: Hungarian algorithm matches predictions with detections using IoU.  
- **Update**: Tracks updated or initialized; unmatched tracks terminated.  

# Results
- Ran at real-time speeds (>60 FPS).  
- Outperformed many complex trackers of the time on MOT16 benchmark.  
- Became the new lightweight baseline for MOT.  

# Why it Mattered
- Reframed MOT as **detection + association**, moving away from handcrafted pipelines.  
- Influential baseline for later trackers: DeepSORT, ByteTrack, OC-SORT, BoT-SORT.  
- Proved that **simple, efficient methods** could be highly competitive.  

# Architectural Pattern
- Detector → Kalman filter → Hungarian association → track update.  

# Connections
- Predecessor to **DeepSORT (2017)** (added appearance features).  
- Indirect predecessor to **ByteTrack (2022)** (better handling of low-confidence detections).  
- Foundation for modern MOT pipelines.  

# Implementation Notes
- Requires only bounding box detections; no appearance model.  
- Extremely fast and easy to implement.  
- Public code made it a standard baseline.  

# Critiques / Limitations
- Struggles with occlusion and crowded scenes.  
- Frequent **identity switches** without appearance cues.  
- Assumes fairly smooth motion (simple Kalman model).  

---

# Educational Connections

## Undergraduate-Level Concepts
- What multi-object tracking (MOT) is.  
- Basics of Kalman filter: predict and update.  
- Hungarian algorithm for assignment.  
- Why SORT is so fast and lightweight.  

## Postgraduate-Level Concepts
- Trade-offs of motion-only vs motion + appearance tracking.  
- ID switches as a key MOT failure mode.  
- SORT as a foundation for detection-based MOT pipelines.  
- Influence on modern MOT with deep detectors and transformers.  

---

# My Notes
- SORT was the **turning point**: MOT simplified to “detection + association.”  
- Hugely influential despite simplicity — shows value of **strong baselines**.  
- Open question: Can modern MOT return to SORT’s **simplicity + efficiency** but with transformer power?  
- Possible extension: A **SORT-Transformer hybrid** leveraging end-to-end learned association.  

---
