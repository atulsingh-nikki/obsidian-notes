---
title: "CenterTrack: Tracking Objects as Points (2020)"
aliases:
  - CenterTrack
  - Tracking Objects as Points
authors:
  - Xingyi Zhou
  - Vladlen Koltun
  - Philipp Krähenbühl
year: 2020
venue: "ECCV"
doi: "10.48550/arXiv.2004.01177"
arxiv: "https://arxiv.org/abs/2004.01177"
code: "https://github.com/xingyizhou/CenterTrack"
citations: 2500+
dataset:
  - MOT17
  - KITTI
  - COCO
  - BDD100K
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - detection
  - end-to-end
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[Tracktor (2019)]]"
  - "[[FairMOT (2020)]]"
  - "[[ByteTrack (2022)]]"
predecessors:
  - "[[Tracktor (2019)]]"
successors:
  - "[[FairMOT (2020)]]"
  - "[[TransTrack (2021)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**CenterTrack** extended the **CenterNet detector** into a joint **detection + tracking network**, predicting both object centers and their offsets across frames. This allowed **end-to-end multi-object tracking (MOT)** without relying on separate motion models or regression-only propagation like Tracktor.

# Key Idea
> Represent objects as **center points** and jointly predict their **locations and motion offsets** in a single network, achieving unified detection and tracking.

# Method
- **Base model**: Built on **CenterNet**, which detects objects as center keypoints.  
- **Tracking extension**:  
  - Input: Current frame + previous frame.  
  - Network outputs: object centers + box sizes + **offset vectors** pointing to object’s previous location.  
- **Association**: Tracks formed by linking detections using predicted offsets.  
- **Training**: Joint loss for detection + tracking.  

# Results
- Outperformed Tracktor and other baselines on **MOT17, KITTI**.  
- Real-time performance (30 FPS).  
- Strong performance in autonomous driving datasets.  

# Why it Mattered
- First **end-to-end MOT framework** — no handcrafted motion models or post-hoc association.  
- Bridged detection and tracking into a single architecture.  
- Influenced later trackers like **FairMOT (2020)** and **TransTrack (2021)**.  

# Architectural Pattern
- CenterNet → add offset prediction → direct track association.  

# Connections
- Successor to **Tracktor (2019)** (detector-based tracking).  
- Predecessor to **FairMOT (2020)** (joint detection + ReID).  
- Related to **TransTrack (2021)** (transformer-based detection + tracking).  

# Implementation Notes
- Runs real-time on modern GPUs.  
- Input must include pairs of frames for offset estimation.  
- Public PyTorch code widely adopted.  

# Critiques / Limitations
- Lacks explicit appearance modeling → ID switches in occlusions.  
- Offset-based linking less robust in crowded scenes.  
- Struggles with long-term tracking.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What object detection is (finding bounding boxes).  
- How tracking can be integrated directly into detection.  
- Idea of using **center points** as anchors for tracking.  
- Applications: self-driving cars, sports analytics, surveillance.  

## Postgraduate-Level Concepts
- Joint detection + tracking vs two-stage pipelines.  
- Offset regression as temporal linking.  
- Comparison with appearance-based association methods.  
- Extension to transformer-based architectures (TransTrack, MOTR).  

---

# My Notes
- CenterTrack was the **true step into end-to-end MOT**.  
- Showed motion offsets could replace explicit Kalman filters.  
- Open question: Can center-based tracking scale well to **3D tracking with LiDAR**?  
- Possible extension: Fuse CenterTrack ideas with **transformers for global temporal context**.  

---
