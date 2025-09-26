---
title: Fast R-CNN (2015)
aliases:
  - Fast R-CNN
  - Fast R-CNN (2015)
authors:
  - Ross Girshick
year: 2015
venue: ICCV
doi: 10.48550/arXiv.1504.08083
arxiv: https://arxiv.org/abs/1504.08083
code: https://github.com/rbgirshick/fast-rcnn
citations: 42,000+
dataset:
  - PASCAL VOC
  - Preliminary MS COCO
tags:
  - paper
  - object-detection
  - cnn
  - roi-pooling
fields:
  - vision
  - detection
related:
  - '[[Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN)|R-CNN (2014)]]'
  - "[[SPPnet (2014)]]"
predecessors:
  - "[[SPPnet (2014)]]"
  - '[[Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN)|R-CNN]]'
successors:
  - "[[Mask R-CNN]]"
  - "[[Fast R-CNN (2015)|Fast R-CNN]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
Fast R-CNN introduced an **efficient, single-stage training** pipeline for region-based object detection. It dramatically sped up both training and inference over R-CNN and SPPnet, while improving accuracy on PASCAL VOC benchmarks :contentReference[oaicite:1]{index=1}.

# Key Idea
> Share convolutional feature computation across all object proposals via an **RoI pooling layer**, enabling faster and unified optimization of classification and bounding-box regression.

# Method
- Entire image processed through convolutional backbone (e.g., VGG16).
- **RoI pooling** extracts fixed-size features per proposal directly from shared conv feature maps.
- Branches into two sibling heads:
  - Softmax classifier (object vs. background + class categories)
  - Bounding-box regressor
- Uses a **multi-task loss**, merging classification and localization into one end-to-end training :contentReference[oaicite:2]{index=2}.

# Results
- Faster than R-CNN by **~9× in training and ~213× at testing**, with higher detection mAP on PASCAL VOC 2012 :contentReference[oaicite:3]{index=3}.
- Outperformed SPPnet while reducing disk storage (no feature caching required) :contentReference[oaicite:4]{index=4}.

# Why it Mattered
- Cleaned up the object detection pipeline: replaced multi-stage patch-CNN-SVM-regressor architecture with a single, trainable network.
- RoI pooling became a foundation for detectors like Faster R-CNN and Mask R-CNN.

# Architectural Pattern
- Single-pass convolution feature extraction.
- RoI pooling for region-level representation.
- Multi-task detection head (classification + regression).

# Connections
- Builds on: R-CNN and SPPnet.
- Succeded by: Faster R-CNN (adding RPN), Mask R-CNN (instance segmentation).

# Implementation Notes
- Enabled end-to-end fine-tuning including convolutional layers.
- RoI pooling critical for efficiency but introduces quantization; later replaced with RoIAlign in Mask R-CNN.

# Critiques / Limitations
- Reliance on external region proposals (e.g., selective search).
- RoI pooling quantization artifacts.
- Still two-stage detection; objectness proposals separate from classification.

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1504.08083) :contentReference[oaicite:5]{index=5}
- [GitHub code (Caffe)](https://github.com/rbgirshick/fast-rcnn)

---

# Educational Connections

## Undergraduate-Level
- Convolutional features + region proposals → classification & bounding-box regression.
- Structure of multi-task loss functions.
- Efficiency gains from shared computation.

## Postgraduate-Level
- RoI pooling enables spatial cropping within feature space.
- Insights into designing efficient detection architectures.
- Foundation of detection cascades and region-based detectors.

---

# My Notes
- One of those papers that changed detection forever — lean, fast, and smart.
- RoI pooling logic still resonates across segmentation and detection networks.
- Open question: Could the concept of in-feature-region extraction be generalized for video editing tasks?
- Extension idea: Use RoI-like feature modulation within diffusion U-Nets to guide region-specific edits.

---

Let me know if you want a note on **Faster R-CNN next**, or maybe the shift to single-stage detectors like YOLO or RetinaNet.
::contentReference[oaicite:6]{index=6}

---
