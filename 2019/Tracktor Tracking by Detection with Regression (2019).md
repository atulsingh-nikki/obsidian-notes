---
title: "Tracktor: Tracking by Detection with Regression (2019)"
aliases:
  - Tracktor
  - Tracking by Regression
authors:
  - Philipp Bergmann
  - Tim Meinhardt
  - Laura Leal-Taixé
year: 2019
venue: "ICCV"
doi: "10.1109/ICCV.2019.00370"
arxiv: "https://arxiv.org/abs/1903.05625"
code: "https://github.com/phil-bergmann/tracking_wo_bnw"
citations: 2000+
dataset:
  - MOT16
  - MOT17
tags:
  - paper
  - tracking
  - multi-object-tracking
  - regression
  - detection
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[SORT (2016)]]"
  - "[[DeepSORT (2017)]]"
  - "[[ByteTrack (2022)]]"
  - "[[OC-SORT (2023)]]"
predecessors:
  - "[[SORT (2016)]]"
successors:
  - "[[CenterTrack (2020)]]"
  - "[[FairMOT (2020)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**Tracktor** reframed multi-object tracking (MOT) as a **bounding box regression task** using modern object detectors. Instead of explicit motion models (e.g., Kalman filters), it reused the **regression head of a detector** (Faster R-CNN) to propagate bounding boxes across frames, making tracking more integrated with detection.

# Key Idea
> A detector’s bounding box regression head can be reused to **predict object positions in the next frame**, effectively performing tracking without explicit motion models.

# Method
- **Detector-based tracker**: Uses Faster R-CNN trained for detection.  
- **Tracking by regression**:  
  - For each tracked box, run it through the detector’s regression head in the next frame.  
  - Adjust box location → gives predicted position.  
- **Association**: Combine regression predictions with detection outputs via IoU matching.  
- **No explicit motion model**: Regression replaces Kalman filter.  

# Results
- Achieved strong results on **MOT16 and MOT17** benchmarks.  
- Showed that simply reusing detectors could rival dedicated trackers.  
- Pushed MOT field towards closer integration with detection.  

# Why it Mattered
- Broke from SORT’s **motion model + assignment** paradigm.  
- Highlighted the growing dominance of **detectors in MOT pipelines**.  
- Inspired later joint detection-and-tracking methods like **CenterTrack (2020)** and **FairMOT (2020)**.  

# Architectural Pattern
- Detector (Faster R-CNN) → regression head → box propagation + detection association.  

# Connections
- Parallel evolution to the **SORT/DeepSORT/ByteTrack** family.  
- Predecessor to **CenterTrack (2020)** (joint detection and tracking).  
- Related to **FairMOT (2020)** (multi-task detection + ReID).  

# Implementation Notes
- Requires a strong detector with regression head (Faster R-CNN).  
- Simpler than training a separate motion model.  
- Public code released, widely used baseline.  

# Critiques / Limitations
- Dependent on detector quality — weak detectors → poor tracking.  
- Struggles with long-term occlusions (no explicit appearance model).  
- Not real-time due to Faster R-CNN backbone.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What regression heads in detectors do.  
- How bounding boxes can be “moved forward” across frames.  
- Difference between detection-based and motion-based tracking.  
- Simplicity: reuse existing components instead of building new models.  

## Postgraduate-Level Concepts
- Detector–tracker unification: implications for MOT design.  
- Comparison with Kalman filter–based methods.  
- Trade-offs: regression vs explicit motion/appearance modeling.  
- Influence on later **end-to-end joint detection + tracking** networks.  

---

# My Notes
- Tracktor was a **paradigm shift**: MOT ≠ motion models, but could just be **detector regression + association**.  
- Showed the field moving from **separate tracking modules → integrated detection-tracking pipelines**.  
- Open question: Is regression-only MOT enough for **3D tracking** (LiDAR + cameras)?  
- Possible extension: Combine regression-based tracking with **transformers** for richer temporal reasoning.  

---
