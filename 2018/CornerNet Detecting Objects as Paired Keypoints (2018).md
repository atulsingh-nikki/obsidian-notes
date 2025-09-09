---
title: "CornerNet: Detecting Objects as Paired Keypoints (2018)"
aliases: 
  - CornerNet
  - Keypoint-based Object Detection
authors:
  - Hei Law
  - Jia Deng
year: 2018
venue: "ECCV"
doi: "10.1007/978-3-030-01264-9_1"
arxiv: "https://arxiv.org/abs/1808.01244"
code: "https://github.com/princeton-vl/CornerNet"
citations: 4000+
dataset:
  - MS COCO
tags:
  - paper
  - object-detection
  - anchor-free
  - keypoints
fields:
  - vision
  - detection
related:
  - "[[CenterNet (2019)]]"
  - "[[FCOS (2019)]]"
predecessors:
  - "[[YOLO (2016)]]"
  - "[[RetinaNet (2017)]]"
successors:
  - "[[CenterNet (2019)]]"
  - "[[DETR (2020)]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
CornerNet proposed an **anchor-free object detection** method that represents each object as a **pair of keypoints**: its top-left and bottom-right corners. These are detected using **heatmaps**, and grouped with learned embeddings to form bounding boxes.

# Key Idea
> Detect object **corners directly as keypoints** and group them using embeddings to form bounding boxes, avoiding anchor boxes.

# Method
- **Heatmap prediction**:  
  - Network outputs heatmaps for top-left and bottom-right corners.  
  - Each heatmap indicates the probability of a pixel being a corner.  
- **Embedding vectors**: Learned embeddings ensure that paired corners belong to the same object.  
- **Backbone**: Hourglass network for high-resolution keypoint prediction.  
- **Training losses**:  
  - Focal loss for keypoint detection.  
  - Pull-push loss for embedding consistency.  

# Results
- Achieved strong performance on **MS COCO**, competitive with anchor-based detectors.  
- Outperformed RetinaNet at similar settings.  
- Demonstrated the viability of **anchor-free, keypoint-based detection**.  

# Why it Mattered
- One of the first successful **anchor-free detectors**.  
- Showed corners/keypoints can replace anchor-based bounding box regression.  
- Paved the way for CenterNet (adding center keypoint) and DETR (set prediction).  

# Architectural Pattern
- Keypoint heatmaps for corner detection.  
- Embeddings for grouping corners.  
- Hourglass backbone for multi-scale representation.  

# Connections
- **Contemporaries**: RetinaNet (anchor-based one-stage).  
- **Influence**: CenterNet (triplets), DETR (end-to-end anchor-free).  

# Implementation Notes
- Hourglass backbone computationally heavy.  
- Embedding matching adds complexity.  
- Heatmap quality strongly impacts detection accuracy.  

# Critiques / Limitations
- Matching corners is ambiguous in crowded scenes.  
- Slower than regression-based anchor-free detectors (e.g., FCOS).  
- Large backbone (hourglass) limits scalability.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1808.01244)  
- [Official PyTorch implementation](https://github.com/princeton-vl/CornerNet)  
- [Reimplementations in Detectron2, MMDetection]  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Geometry**: Bounding boxes from corner coordinates.  
- **Probability**: Heatmaps for spatial likelihood.  
- **Linear Algebra**: Embeddings for grouping.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Keypoint detection via hourglass networks.  
- **Computer Vision**: Anchor-free vs anchor-based methods.  
- **Research Methodology**: Evaluating detection via mAP benchmarks.  
- **Advanced Optimization**: Pull-push embedding loss for keypoint grouping.  

---

# My Notes
- CornerNet was a **turning point**: first major anchor-free competitor to anchor-based detectors.  
- Important for **object editing tasks** where bounding boxes should emerge from keypoints naturally.  
- Open question: Could modern **transformers replace embeddings** for corner pairing?  
- Possible extension: Extend corner-pair detection to **3D cuboid keypoints** for AR/VR editing.  

---
