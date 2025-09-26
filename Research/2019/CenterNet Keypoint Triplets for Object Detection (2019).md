---
title: "CenterNet: Keypoint Triplets for Object Detection (2019)"
aliases:
  - CenterNet
  - Keypoint Triplet Detection
  - CornetNet_2019
authors:
  - Xingyi Zhou
  - Dequan Wang
  - Philipp Krähenbühl
year: 2019
venue: ICCV
doi: 10.1109/ICCV.2019.00666
arxiv: https://arxiv.org/abs/1904.07850
code: https://github.com/xingyizhou/CenterNet
citations: 5000+
dataset:
  - COCO
tags:
  - paper
  - object-detection
  - anchor-free
  - keypoints
fields:
  - vision
  - detection
related:
  - "[[CornerNet Detecting Objects as Paired Keypoints (2018)|CornerNet]]"
  - "[[FCOS Fully Convolutional One-Stage Object Detection (2019)|FCOS]]"
predecessors:
  - "[[CornerNet Detecting Objects as Paired Keypoints (2018)|CornerNet]]"
successors:
  - "[[DETR End-to-End Object Detection with Transformers (2020)|DETR]]"
impact: ⭐⭐⭐⭐☆
status: read
---

# Summary
CenterNet proposed an **anchor-free object detector** that represents each object by a **triplet of keypoints**:  
- object center  
- top-left corner  
- bottom-right corner  

This design enables precise detection while simplifying the pipeline, building upon **CornerNet** by adding the center keypoint for better localization.

# Key Idea
> Represent objects as **keypoint triplets** (center + corners) and detect them with heatmap-based keypoint prediction networks.

# Method
- Backbone CNN predicts three heatmaps: centers, top-left corners, bottom-right corners.  
- Each heatmap indicates the probability of a pixel being that keypoint.  
- Bounding boxes formed by pairing corners with corresponding centers.  
- Added **center heatmap pooling** to verify object presence.  
- Losses: focal loss for keypoints + regression for offsets/sizes.  

# Results
- Achieved **state-of-the-art anchor-free detection** on COCO at the time.  
- Improved over CornerNet in accuracy and efficiency.  
- Showed competitive mAP compared to anchor-based detectors like RetinaNet.  

# Why it Mattered
- Strengthened the case for **anchor-free object detection**.  
- Demonstrated that simple keypoint-based formulations are effective.  
- Influenced later transformer-based set prediction methods (DETR).  

# Architectural Pattern
- Fully convolutional backbone.  
- Heatmap-based keypoint detection.  
- Center validation for bounding boxes.  

# Connections
- **Contemporaries**: FCOS (anchor-free dense regression).  
- **Influence**: DETR (end-to-end set prediction), keypoint-based human pose detectors.  

# Implementation Notes
- Requires careful matching of corners and centers.  
- Center pooling boosts precision by validating boxes.  
- Still requires non-max suppression (NMS).  

# Critiques / Limitations
- More complex than FCOS (multiple heatmaps, pairing).  
- Matching keypoints can be ambiguous for crowded scenes.  
- Slower than regression-only anchor-free detectors.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1904.07850)  
- [Official PyTorch code](https://github.com/xingyizhou/CenterNet)  
- [Detectron2 reimplementations available]  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Geometry**: Bounding boxes from corners + centers.  
- **Probability**: Heatmaps as spatial likelihood distributions.  
- **Linear Algebra**: Offset regression for subpixel accuracy.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Keypoint heatmap detectors.  
- **Computer Vision**: Anchor-free detection vs regression-based methods.  
- **Research Methodology**: Evaluating detection via mAP.  
- **Advanced Optimization**: Focal loss adaptation for sparse keypoints.  

---

# My Notes
- CenterNet felt like the **keypoint cousin of FCOS**: both anchor-free, different philosophies (heatmaps vs regression).  
- Relevant for **video detection** where center-based consistency helps with temporal stability.  
- Open question: Can keypoint-triplet formulation extend naturally to **3D detection** (center + cuboid corners)?  
- Possible extension: Use center-triplet cues in **diffusion-based object tracking/editing**.  

---
