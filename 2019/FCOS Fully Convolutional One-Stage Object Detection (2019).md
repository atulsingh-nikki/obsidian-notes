---
title: "FCOS: Fully Convolutional One-Stage Object Detection (2019)"
aliases: 
  - FCOS
  - Fully Convolutional Object Detector
authors:
  - Zhi Tian
  - Chunhua Shen
  - Hao Chen
  - Tong He
year: 2019
venue: "ICCV"
doi: "10.1109/ICCV.2019.00664"
arxiv: "https://arxiv.org/abs/1904.01355"
code: "https://github.com/tianzhi0549/FCOS"
citations: 6000+
dataset:
  - COCO
tags:
  - paper
  - object-detection
  - anchor-free
  - one-stage
fields:
  - vision
  - detection
related:
  - "[[YOLO (2016)]]"
  - "[[RetinaNet (2017)]]"
  - "[[CenterNet (2019)]]"
predecessors:
  - "[[RetinaNet (2017)]]"
successors:
  - "[[ATSS (2020)]]"
  - "[[DETR (2020)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
FCOS proposed a **fully convolutional, anchor-free one-stage detector**, eliminating anchor boxes and proposal generation. It reformulated object detection as **per-pixel prediction** of classification and bounding box regression, simplifying design while achieving state-of-the-art performance.

# Key Idea
> Remove anchors and treat object detection as **dense prediction**: each pixel in feature maps predicts whether it belongs to an object and regresses distances to bounding box edges.

# Method
- **Anchor-free detection**: Each location predicts:  
  - Class scores.  
  - 4 distances (to left, right, top, bottom of object bounding box).  
- **FPN backbone** (Feature Pyramid Network) for multi-scale detection.  
- **Center-ness branch**: Extra head predicts how close pixel is to object center → downweights low-quality predictions.  
- **Loss**: Classification loss (focal loss) + regression loss (IoU/GIoU) + center-ness loss.  

# Results
- Outperformed anchor-based detectors (RetinaNet, Faster R-CNN) on COCO.  
- Simpler design, fewer hyperparameters than anchor-based methods.  
- Showed anchor-free detectors are viable and competitive.  

# Why it Mattered
- One of the **first anchor-free detectors** to reach and surpass anchor-based methods.  
- Eliminated anchor hyperparameter tuning (sizes, ratios).  
- Influenced subsequent dense prediction and transformer-based detectors (ATSS, DETR).  

# Architectural Pattern
- Fully convolutional, dense prediction.  
- FPN backbone.  
- Center-ness branch for quality scoring.  

# Connections
- **Contemporaries**: CenterNet (keypoint-based detection, 2019).  
- **Influence**: ATSS (2020, adaptive training sample selection), DETR (set prediction).  

# Implementation Notes
- Simpler than anchor-based pipelines.  
- Center-ness prediction critical for suppressing low-quality boxes.  
- Still requires NMS (non-max suppression) for final outputs.  

# Critiques / Limitations
- Struggles with very small objects compared to some anchor-based methods.  
- Dense prediction leads to many negative samples (handled with focal loss).  
- Still convolution-based; transformers soon redefined detection (DETR).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1904.01355)  
- [Official PyTorch implementation](https://github.com/tianzhi0549/FCOS)  
- [Detectron2 implementation](https://github.com/facebookresearch/detectron2)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Geometry**: Bounding box regression as distances from pixel to edges.  
- **Linear Algebra**: Feature maps → per-pixel prediction.  
- **Probability**: Classification confidence and focal loss.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Anchor-free detection heads.  
- **Computer Vision**: Dense prediction and center-ness quality estimation.  
- **Research Methodology**: Benchmarking detectors on COCO.  
- **Advanced Optimization**: Handling class imbalance with focal loss.  

---

# My Notes
- FCOS was a **game-changer** → simple, anchor-free, strong results.  
- Relevant for **video object detection/editing pipelines**, where anchors are clunky.  
- Open question: Can FCOS-style dense regression integrate with **transformer DETR-like architectures**?  
- Possible extension: Anchor-free **video diffusion detectors** for spatio-temporal editing.  

---
