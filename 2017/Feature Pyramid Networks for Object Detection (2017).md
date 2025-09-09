---
title: Feature Pyramid Networks for Object Detection (2017)
aliases:
  - FPN
  - Feature Pyramid Networks
authors:
  - Tsung-Yi Lin
  - Piotr Dollár
  - Ross Girshick
  - Kaiming He
  - Bharath Hariharan
  - Serge Belongie
year: 2017
venue: CVPR
doi: 10.1109/CVPR.2017.106
arxiv: https://arxiv.org/abs/1612.03144
code: https://github.com/facebookresearch/detectron2
citations: 20,000+
dataset:
  - MS COCO
tags:
  - paper
  - object-detection
  - feature-pyramids
  - multi-scale
fields:
  - vision
  - detection
related:
  - "[[HyperNet]]"
  - "[[Mask R-CNN]]"
predecessors:
  - "[[HyperNet]]"
  - "[[Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks|Faster R-CNN (2015)]]"
successors:
  - "[[RetinaNet]]"
  - "[[PANet]]"
impact: ⭐⭐⭐⭐⭐
status: read
---
# Summary
Feature Pyramid Networks (FPN) introduced a **top-down pathway with lateral connections** to efficiently build multi-scale feature pyramids for object detection. It improved detection of **small objects** while keeping computational cost low, becoming a **default backbone** for many detection and segmentation models.

# Key Idea
> Combine high-resolution low-level features with semantically strong high-level features through a top-down and lateral fusion pathway, creating rich multi-scale feature maps.

# Method
- **Bottom-up pathway**: Standard CNN backbone (e.g., ResNet) generates hierarchical feature maps.  
- **Top-down pathway**: Higher-level semantic features are upsampled.  
- **Lateral connections**: Merge upsampled features with corresponding lower-level feature maps.  
- Produces a **pyramid of feature maps** (P2–P5), each with rich semantics at multiple scales.  
- Used in both RPN and detection heads.  

# Results
- Improved accuracy of **Faster R-CNN** on MS COCO, especially for small objects.  
- Achieved strong results without significantly increasing computational overhead.  
- Generalized well across detection, segmentation, and keypoint tasks.  

# Why it Mattered
- Made **multi-scale feature representation standard** in detection.  
- Simpler and more efficient than HyperNet or image pyramids.  
- Became a core component in frameworks like **Mask R-CNN, RetinaNet, Detectron2**.  

# Architectural Pattern
- **Top-down + lateral fusion** architecture.  
- Multi-scale feature pyramid as backbone.  
- Modular and widely adaptable.  

# Connections
- **Contemporaries**: SSD (2016), YOLOv2.  
- **Influence**: RetinaNet (2017), PANet (2018), Cascade R-CNN, Detectron2 standard.  

# Implementation Notes
- Usually paired with ResNet backbones.  
- Needs careful alignment of feature map resolutions when merging.  
- Pyramid levels can be extended for larger/smaller object scales.  

# Critiques / Limitations
- Still two-stage; not as lightweight as single-stage detectors (YOLO).  
- Pyramid features are hand-designed; later approaches use learned attention (e.g., NAS-FPN).  
- May lose very fine detail for extremely small/thin objects.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1612.03144)  
- [Detectron2 implementation](https://github.com/facebookresearch/detectron2)  
- [PyTorch FPN tutorial](https://pytorch.org/vision/stable/_modules/torchvision/ops/feature_pyramid_network.html)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Convolutions, upsampling.  
- **Signals & Systems**: Multi-scale representations.  
- **Optimization Basics**: End-to-end joint training with detection heads.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Top-down + lateral skip fusion.  
- **Computer Vision**: Multi-scale detection challenges.  
- **Research Methodology**: Benchmarking on MS COCO small/medium/large subsets.  
- **Advanced Optimization**: Balancing pyramid levels in training.  

---

# My Notes
- Extremely relevant for **multi-scale object selection in video** workflows.  
- Open question: Can pyramid fusion be replaced by **transformer hierarchies** (e.g., Swin Transformer)?  
- Possible extension: Use **temporal pyramid networks** for video segmentation/tracking.  
