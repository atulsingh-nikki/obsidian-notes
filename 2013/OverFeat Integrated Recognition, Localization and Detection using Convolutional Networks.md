---
title: "OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks"
aliases:
  - OverFeat (2013)
authors:
  - Pierre Sermanet
  - David Eigen
  - Xiang Zhang
  - Michaël Mathieu
  - Rob Fergus
  - Yann LeCun
year: 2013
venue: ICLR 2014 (arXiv 2013)
dataset:
  - ImageNet (ILSVRC 2013 classification + localization + detection tasks)
tags:
  - computer-vision
  - cnn
  - object-detection
  - object-localization
  - image-classification
  - sliding-window
arxiv: https://arxiv.org/abs/1312.6229
related:
  - "[[R-CNN (2014)]]"
  - "[[YOLO (2016)]]"
  - "[[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]]"
  - "[[Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks|Faster R-CNN (2015)]]"
---

# Summary
OverFeat extended convolutional neural networks from **classification** to **localization and detection**, showing CNNs could serve as a **unified architecture** for multiple vision tasks. It combined sliding-window detection with CNN feature maps, achieving strong performance on ImageNet 2013 challenges.

# Key Idea (one-liner)
> Use a single CNN trained for classification and extend it with regression and sliding-window strategies to perform localization and detection.

# Method
- **Base CNN**:
  - Similar to [[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]] (convolutions + pooling + fully connected).
- **Recognition**:
  - Standard classification on ImageNet categories.
- **Localization**:
  - Regression head predicts bounding box coordinates.
- **Detection**:
  - Sliding-window approach over multiple scales.
  - CNN applied efficiently to overlapping regions by sharing convolutional computations.
- **Training**:
  - Multi-task loss for classification + localization.
  - Scale augmentation for robust detection.

# Results
- **ILSVRC 2013**:
  - Won 1st place in localization.
  - Top results in detection and classification tasks.
- Demonstrated CNNs could generalize beyond classification into structured tasks.
- Efficiency: shared convolutional computation sped up sliding-window detection.

# Why it Mattered
- Showed CNNs could be **multi-task learners**: classification + localization + detection.
- Anticipated later unified frameworks (Faster R-CNN, YOLO).
- Sliding-window CNN detection → precursor to region-based detectors.
- Helped establish CNNs as the standard toolkit for computer vision.

# Architectural Pattern
- [[Convolutional Neural Networks]] → feature extraction backbone.
- [[Sliding Window Detection]] → scanning with CNN features.
- [[Multi-Task Learning]] → classification + bounding-box regression.
- [[Shared Convolutions]] → efficient reuse of features.

# Connections
- **Predecessors**:
  - [[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]] — classification breakthrough.
- **Successors**:
  - [[Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN)|R-CNN (2014)]] — region proposals with CNN features.
  - [[Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks|Faster R-CNN (2015)]] — end-to-end detection with RPNs.
  - [[YOLO You Only Look Once — Unified, Real-Time Object Detection|YOLO (2016)]] — single-shot detection without sliding windows.
- **Influence**:
  - Bridged classification and detection with CNNs.
  - Inspired architectures combining multiple outputs from one backbone.

# Implementation Notes
- Sliding-window detection was still compute-heavy compared to later methods.
- Regression head made CNN output bounding boxes directly.
- Multi-scale pyramid important for handling object sizes.
- Implemented efficiently using shared convolution layers.

# Critiques / Limitations
- Sliding-window detection slower than proposal-based or single-shot methods.
- Bounding box regression less precise than later frameworks.
- Computational demands high for 2013 hardware.
- Superseded quickly by region-proposal-based methods.

# Repro / Resources
- Paper: [arXiv:1312.6229](https://arxiv.org/abs/1312.6229)
- Dataset: [[ImageNet]]
- Early Caffe/Torch implementations (not official).

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Convolutions across image regions.
  - Regression for bounding boxes as linear mapping.

- **Probability & Statistics**
  - Softmax classification.
  - IoU (Intersection-over-Union) metric for evaluation.

- **Calculus**
  - Backpropagation for regression + classification losses.
  - Gradient-based optimization of CNN weights.

- **Signals & Systems**
  - Sliding windows = signal scanning.
  - Multi-scale pyramid = frequency scaling.

- **Data Structures**
  - Feature maps as tensors.
  - Bounding boxes as coord
