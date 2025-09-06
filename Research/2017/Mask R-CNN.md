---
title: Mask R-CNN
aliases:
  - MaskR-CNN (2017)
authors:
  - Kaiming He
  - Georgia Gkioxari
  - Piotr Dollár
  - Ross Girshick
year: 2017
venue: ICCV 2017
dataset:
  - MS COCO 2017
  - PASCAL VOC
tags:
  - computer-vision
  - instance-segmentation
  - object-detection
  - two-stage-detector
  - mask-prediction
  - roi-align
arxiv: https://arxiv.org/abs/1703.06870
related:
  - "[[R-CNN (2014)]]"
  - "[[Fast R-CNN (2015)]]"
  - "[[Faster R-CNN (2015)]]"
  - "[[YOLO (2016)]]"
  - "[[SSD (2016)]]"
  - "[[Panoptic Segmentation]]"
  - "[[ROI Align]]"
---

# Summary
Mask R-CNN extended Faster R-CNN by adding a **parallel branch for predicting object masks**, enabling **instance segmentation** (detecting *what* and *where* each object is, with pixel-level masks). It also introduced **ROIAlign** for precise feature extraction, solving misalignment issues from ROI Pooling. Mask R-CNN became the **go-to framework for detection + segmentation** tasks.

# Key Idea (one-liner)
> Add a mask prediction branch to Faster R-CNN and fix ROI misalignment with ROIAlign → unified detection, classification, and segmentation.

# Method
- **Base detector**: Faster R-CNN backbone (e.g., ResNet-50/101 + FPN).
- **ROIAlign**: replaces ROI Pooling, preserves spatial alignment via bilinear interpolation.
- **Mask head**: small FCN (fully convolutional network) that predicts a binary mask for each ROI.
- **Multi-task loss**: combines classification loss + bounding-box regression + mask loss.
- **Training**: end-to-end, mask branch trained only on positive ROIs.

# Results
- State-of-the-art performance on **MS COCO 2017** (instance segmentation + detection).
- Flexible framework → extended to panoptic segmentation, keypoint detection (Pose Estimation).
- Mask R-CNN backbone widely adopted in research & production.

# Why it Mattered
- First practical **instance segmentation** framework.
- Introduced **ROIAlign**, later reused in many architectures.
- Set new benchmarks for COCO detection/segmentation.
- Extensible design → opened path to panoptic segmentation, pose estimation.

# Architectural Pattern
- [[Two-Stage Detectors]] → proposals (RPN) + classification/regression.
- [[ROI Align]] → precise feature alignment.
- [[Mask Branch]] → FCN mask prediction per ROI.
- [[Multi-task Loss]] → joint optimization of detection and segmentation.

# Connections
- **Predecessors**:
  - [[Faster R-CNN (2015)]] → detection backbone.
- **Successors**:
  - [[Panoptic FPN (2019)]] → unified semantic + instance segmentation.
  - [[Detectron2 (Facebook AI)]] → production-grade implementation.
- **Alternatives**:
  - [[YOLO (2016)]] + extensions → faster, less precise.
  - [[FCIS (2016)]] → earlier segmentation attempt, less flexible.

# Implementation Notes
- Common backbones: ResNet-50/101 + FPN.
- Pretrained models widely available (Detectron2, MMDetection).
- Inference slower than YOLO/SSD, but higher accuracy and segmentation.
- Still widely used as a baseline in academic research.

# Critiques / Limitations
- Slower than one-stage detectors.
- Memory intensive (multiple branches).
- Mask prediction limited to class-specific binary masks (later generalized in panoptic segmentation).

# Repro / Resources
- Paper: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Code: Detectron, Detectron2, MMDetection.
- Dataset: [[MS COCO]]

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Convolutions in backbone and mask head.
  - Tensor slicing and alignment in ROIAlign.
  
- **Probability & Statistics**
  - Softmax classification.
  - IoU for detection evaluation.
  - Binary cross-entropy for mask loss.

- **Calculus**
  - Backprop through bilinear interpolation (ROIAlign).
  - Multi-task loss optimization.

- **Signals & Systems**
  - Convolutions for mask prediction.
  - Pooling vs interpolation effects on signals.

- **Data Structures**
  - ROIs as structured proposals.
  - Masks as binary pixel-level arrays.

- **Optimization Basics**
  - Multi-task loss balancing.
  - SGD with momentum for backbone and mask branch.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Training stability with multi-task loss.
  - Balancing classification, regression, and segmentation heads.
  - End-to-end fine-tuning with shared features.

- **Numerical Methods**
  - ROIAlign → bilinear interpolation (numerical precision).
  - Runtime trade-offs between accuracy and speed.

- **Machine Learning Theory**
  - Multi-task learning (shared backbone, specialized heads).
  - Instance segmentation vs semantic segmentation.
  - Generalization to panoptic tasks.

- **Computer Vision**
  - Unified detection + segmentation pipeline.
  - Benchmark leader on MS COCO.
  - Extensions to human pose estimation.

- **Neural Network Design**
  - Multi-branch design: detection + mask.
  - Feature Pyramid Networks (FPN) as backbone improvement.
  - Modularity → extendable to other tasks.

- **Transfer Learning**
  - Pretrained backbones (ImageNet/COCO).
  - Fine-tuning for domain-specific segmentation tasks.

- **Research Methodology**
  - Ablation: ROIAlign vs ROI Pooling.
  - Benchmarking on MS COCO.
  - Strong baselines for future segmentation research.
