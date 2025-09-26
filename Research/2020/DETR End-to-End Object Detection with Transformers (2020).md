---
title: "DETR: End-to-End Object Detection with Transformers (2020)"
aliases:
  - DETR
  - End-to-End Transformer Detection
authors:
  - Nicolas Carion
  - Francisco Massa
  - Gabriel Synnaeve
  - Nicolas Usunier
  - Alexander Kirillov
  - Sergey Zagoruyko
year: 2020
venue: ECCV
doi: 10.1007/978-3-030-58452-8_13
arxiv: https://arxiv.org/abs/2005.12872
code: https://github.com/facebookresearch/detr
citations: 16,000+
dataset:
  - COCO
tags:
  - paper
  - object-detection
  - transformer
  - anchor-free
fields:
  - vision
  - detection
related:
  - "[[CornerNet Detecting Objects as Paired Keypoints (2018)]]"
  - "[[FCOS Fully Convolutional One-Stage Object Detection (2019)|FCOS]]"
  - "[[CenterNet Keypoint Triplets for Object Detection (2019)|CornetNet_2019]]"
predecessors:
  - "[[CornerNet Detecting Objects as Paired Keypoints (2018)|CornerNet]]"
  - "[[FCOS Fully Convolutional One-Stage Object Detection (2019)|FCOS]]"
successors:
  - "[[Deformable DETR (2020)]]"
  - "[[DN-DETR (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---


# Summary
DETR reframed object detection as a **direct set prediction problem** using **transformers**, eliminating hand-crafted components like anchors, region proposals, and non-max suppression (NMS). It outputs a fixed-size set of predictions, trained end-to-end with bipartite matching.

# Key Idea
> Treat detection as a **set prediction task**, using transformers to globally reason over image features and predict bounding boxes directly.

# Method
- **Backbone**: CNN (ResNet) extracts features.  
- **Transformer encoder-decoder**: Processes image features as tokens.  
- **Object queries**: Learnable embeddings act as slots for predictions.  
- **Hungarian matching**: Bipartite assignment between predictions and ground truth.  
- **Loss**: Combination of classification, L1, and GIoU losses.  

# Results
- Achieved competitive mAP on **COCO** compared to Faster R-CNN and RetinaNet.  
- Removed NMS and anchor-box hyperparameters.  
- Demonstrated end-to-end simplicity but required long training schedules.  

# Why it Mattered
- First successful integration of **transformers into detection**.  
- Unified detection with sequence modeling.  
- Laid the foundation for a wave of DETR variants improving efficiency and convergence.  

# Architectural Pattern
- CNN backbone → transformer encoder-decoder → object queries → set predictions.  

# Connections
- **Contemporaries**: FCOS (dense regression), CenterNet (keypoints).  
- **Influence**: Deformable DETR, Conditional DETR, DN-DETR, transformer-based segmentation.  

# Implementation Notes
- Needs large datasets and long training schedules (~500 epochs).  
- Fixed-size prediction set includes "no object" class for unused slots.  
- Robust but less efficient for small object detection.  

# Critiques / Limitations
- Slow convergence compared to CNN-based detectors.  
- Struggles with small objects and crowded scenes.  
- Requires significant compute resources.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/2005.12872)  
- [Official code (PyTorch)](https://github.com/facebookresearch/detr)  
- [Detectron2 DETR implementations available]  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Attention as matrix multiplications.  
- **Probability**: Hungarian matching ensures one-to-one assignments.  
- **Geometry**: Bounding box regression (L1, IoU).  

## Postgraduate-Level Concepts
- **Neural Network Design**: Encoder-decoder with object queries.  
- **Computer Vision**: Anchor-free, end-to-end detection.  
- **Research Methodology**: Evaluating end-to-end simplifications.  
- **Advanced Optimization**: Training stability in bipartite matching losses.  

---

# My Notes
- DETR was the **transformer breakthrough** for vision detection.  
- Relevant for **video editing pipelines**: could treat frames as sequences and queries as object tracks.  
- Open question: Can DETR-scale models integrate **diffusion priors for dense prediction**?  
- Possible extension: Use DETR-like queries in **video diffusion editing for object consistency**.  
