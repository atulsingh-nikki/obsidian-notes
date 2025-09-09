---
title: "RepPoints: Point Set Representation for Object Detection (2019)"
aliases:
  - RepPoints
  - Point Set Object Detection
authors:
  - Ze Yang
  - Shaohui Liu
  - Han Hu
  - Liwei Wang
  - Stephen Lin
year: 2019
venue: "ICCV"
doi: "10.1109/ICCV.2019.00943"
arxiv: "https://arxiv.org/abs/1904.11490"
code: "https://github.com/microsoft/RepPoints"
citations: ~1200+
dataset:
  - COCO
  - PASCAL VOC
tags:
  - paper
  - object-detection
  - anchor-free
  - representation-learning
fields:
  - vision
  - detection
related:
  - "[[FCOS (2019)]]"
  - "[[CenterNet (2019)]]"
  - "[[DETR (2020)]]"
predecessors:
  - "[[Anchor-based Object Detectors (Faster R-CNN, RetinaNet)]]"
successors:
  - "[[RepPointsV2 (2020)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**RepPoints** introduced a new representation for objects in detection: a **set of adaptive, deformable points** instead of rigid bounding boxes. These points capture both the **object’s spatial extent and geometry**, enabling more precise localization, especially for non-rigid or irregularly shaped objects.

# Key Idea
> Replace fixed bounding boxes with **learned point sets** that adapt to object shapes, serving as both detection representation and feature sampling locations.

# Method
- **Point Set Representation**: Each object represented by k adaptive points.  
- **Learning**:  
  - Points predicted from features.  
  - Refined iteratively to better align with object shape.  
- **Detection Head**:  
  - Uses point sets to regress bounding boxes (for evaluation) and classify objects.  
  - Point features aggregated via deformable sampling.  
- **Training Loss**: Combination of point-based regression loss and detection classification loss.  

# Results
- Achieved **state-of-the-art detection performance** on COCO at time of publication.  
- Outperformed RetinaNet and Faster R-CNN baselines.  
- Showed stronger localization, especially for deformable/non-rigid objects.  

# Why it Mattered
- Moved object detection **beyond rigid bounding boxes**.  
- Inspired later flexible representations (RepPointsV2, deformable DETR).  
- Demonstrated point-based object modeling as a promising paradigm.  

# Architectural Pattern
- CNN backbone → feature maps → point set prediction → deformable point feature aggregation → detection/classification outputs.  

# Connections
- Related to **anchor-free detectors** like FCOS and CenterNet.  
- Preceded point/deformable representations in DETR-style architectures.  
- Links to deformable convolution (DCNv2).  

# Implementation Notes
- Number of points (k) is a hyperparameter (default 9 or 25).  
- Training requires balancing point regression and classification objectives.  
- Code released by Microsoft Research.  

# Critiques / Limitations
- Final evaluation still relies on bounding boxes, not pure point sets.  
- Additional computation for point refinement and deformable sampling.  
- Gains diminish for rigid objects where bounding boxes suffice.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Object detection basics (bounding boxes).  
- Idea of feature sampling and deformable points.  
- Regression vs classification tasks.  

## Postgraduate-Level Concepts
- Anchor-free detection paradigms.  
- Deformable convolution and feature alignment.  
- Representations beyond bounding boxes in detection.  

---

# My Notes
- RepPoints felt like a **conceptual leap**: from boxes to point sets.  
- Strong link to **pose estimation and non-rigid object modeling**.  
- Open question: Can point sets be extended to **3D object detection and reconstruction**?  
- Possible extension: Integrate RepPoints into **video object editing pipelines** for more flexible tracking than bounding boxes.  

---
