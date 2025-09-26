---
title: "YOLO9000: Better, Faster, Stronger (2017)"
aliases:
  - YOLOv2
  - YOLO9000
authors:
  - Joseph Redmon
  - Ali Farhadi
year: 2017
venue: CVPR
doi: 10.1109/CVPR.2017.690
arxiv: https://arxiv.org/abs/1612.08242
code: https://pjreddie.com/darknet/yolo/
citations: 13,000+
dataset:
  - PASCAL VOC
  - COCO
  - ImageNet (for YOLO9000 joint training)
tags:
  - paper
  - object-detection
  - yolo
  - real-time
fields:
  - vision
  - detection
related:
  - "[[YOLOv3 (2018)]]"
  - "[[YOLO You Only Look Once — Unified, Real-Time Object Detection|YOLO (2016)]]"
predecessors:
  - "[[YOLO You Only Look Once — Unified, Real-Time Object Detection|YOLO (2016)]]"
successors:
  - "[[YOLOv3 (2018)]]"
  - "[[Scaled-YOLOv4]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
YOLO9000 (YOLOv2) improved upon the original YOLO with architectural and training refinements, achieving **state-of-the-art speed/accuracy tradeoff** in real-time object detection. It also introduced **YOLO9000**, trained jointly on detection (COCO, VOC) and classification (ImageNet) data, enabling recognition of **over 9000 categories**.

# Key Idea
> Improve YOLO’s accuracy without sacrificing speed, and extend detection to thousands of categories by **jointly training on detection + classification datasets**.

# Method
- **Architectural improvements (YOLOv2)**:  
  - New backbone: **Darknet-19** (19 conv layers + 5 max-pooling).  
  - Batch normalization for stabilization and better generalization.  
  - High-resolution classifier pretraining.  
  - Anchor boxes (borrowed from Faster R-CNN).  
  - Dimension clusters via k-means to optimize anchor shapes.  
  - Multi-scale training (network trained to adapt to varying input resolutions).  
- **YOLO9000 joint training**:  
  - Combined COCO detection data with ImageNet classification data.  
  - WordTree ontology used to unify labels across datasets.  
  - Enabled detection of **9000+ categories**, including unseen detection classes.  

# Results
- Faster and more accurate than **YOLOv1**, SSD, and Faster R-CNN (at similar speeds).  
- 76.8 mAP on VOC 2007, 48.1 mAP on VOC 2012.  
- Ran in **40–90 FPS** depending on resolution.  
- YOLO9000 could detect categories **not present in detection datasets**, thanks to joint training.  

# Why it Mattered
- Proved that **real-time detection could be accurate**.  
- Introduced **joint classification + detection training** → generalized detection to unseen categories.  
- Paved the way for YOLOv3/v4/v5 and the entire YOLO family still dominant today.  

# Architectural Pattern
- Fully convolutional detector with anchors.  
- Multi-scale feature learning.  
- Joint detection-classification training.  

# Connections
- **Contemporaries**: SSD (2016), Faster R-CNN (2015).  
- **Influence**: YOLOv3 (2018), YOLOv4/v5/v7, Ultralytics YOLO family.  

# Implementation Notes
- Dimension clusters (anchor priors) crucial for accuracy.  
- Multi-scale training improved robustness.  
- WordTree ontology enabled semantic consistency across datasets.  

# Critiques / Limitations
- Still less accurate than slower two-stage detectors (Faster R-CNN).  
- Anchor-based design adds complexity.  
- Long-tail detection (rare classes in 9000) remained weak.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1612.08242)  
- [Official Darknet code](https://pjreddie.com/darknet/yolo/)  
- [PyTorch YOLOv2 implementation](https://github.com/marvis/pytorch-yolo2)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Convolutions, k-means clustering for anchors.  
- **Probability & Statistics**: Object classification, softmax hierarchies.  
- **Optimization Basics**: Multi-task loss (detection + classification).  

## Postgraduate-Level Concepts
- **Neural Network Design**: Anchor-based detection, joint dataset training.  
- **Computer Vision**: Large-vocabulary detection.  
- **Research Methodology**: Dataset unification (WordTree ontology).  
- **Advanced Optimization**: Balancing classification vs detection gradients.  

---

# My Notes
- Important step for **scalable detection** → category expansion without new detection labels.  
- Connects to my interests in **large-vocabulary vision models** (e.g., grounding in video editing).  
- Open question: Can **vision-language models (e.g., CLIP)** replace hierarchical ontologies for joint training?  
- Possible extension: Combine YOLO-style dense detection with **diffusion-based refinement** for high-quality masks.  
