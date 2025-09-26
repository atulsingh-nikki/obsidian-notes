---
title: "DeepLab: Atrous Convolution and Semantic Image Segmentation (2017–2018)"
aliases:
  - DeepLab
  - DeepLabv2
  - DeepLabv3
  - DeepLabv3+
authors:
  - Liang-Chieh Chen
  - George Papandreou
  - Florian Schroff
  - Hartwig Adam
  - et al.
year: 2017–2018
venue: CVPR, ECCV, TPAMI
doi: 10.1109/TPAMI.2017.2699184
arxiv:
  - https://arxiv.org/abs/1606.00915
  - https://arxiv.org/abs/1706.05587
  - https://arxiv.org/abs/1802.02611
code: https://github.com/tensorflow/models/tree/master/research/deeplab
citations: 20000+
dataset:
  - PASCAL VOC
  - Cityscapes
  - COCO
tags:
  - paper
  - segmentation
  - semantic-segmentation
  - deep-learning
fields:
  - vision
  - segmentation
  - semantic-understanding
related:
  - "[[Mask R-CNN]]"
  - "[[PointRend Image Segmentation as Rendering (2020)|PointRend]]"
  - "[[Segment Anything (SAM, 2023)|Segment Anything Model]]"
  - "[[Fully Convolutional Networks for Semantic Segmentation (2015)|FCN]]"
predecessors:
  - "[[Fully Convolutional Networks for Semantic Segmentation (2015)|FCN]]"
successors:
  - "[[Segment Anything (SAM, 2023)|Segment Anything Model]]"
  - "[[SegFormer (2021)]]"
  - "[[HRNet (2020)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**DeepLab** is a family of semantic segmentation models that pioneered **atrous (dilated) convolutions** and **multi-scale context aggregation**. It became the dominant baseline for semantic segmentation tasks before the transformer and foundation model era.

# Key Idea
> Use **atrous (dilated) convolutions** to control receptive field size without losing resolution, and capture multi-scale context via **atrous spatial pyramid pooling (ASPP)**.

# Method
- **Atrous convolution**: Expands receptive field by skipping pixels in convolution kernels.  
- **ASPP (Atrous Spatial Pyramid Pooling)**: Parallel atrous convs at multiple dilation rates to capture context.  
- **CRF (Conditional Random Field)**: Post-processing step (DeepLabv1/v2) for boundary refinement.  
- **Encoder–decoder (DeepLabv3+)**: Adds decoder for sharper segmentation.  

# Results
- Achieved state-of-the-art on **PASCAL VOC**, **Cityscapes**, and **COCO**.  
- Strong at capturing context and object-level semantics.  
- Widely used as a baseline in segmentation research.  

# Why it Mattered
- Popularized **atrous convolutions** in vision.  
- ASPP became a staple in segmentation.  
- Served as the **default strong baseline** until transformers (ViT, SegFormer) and SAM.  

# Architectural Pattern
- CNN backbone (ResNet, Xception).  
- Atrous conv layers for multi-scale receptive fields.  
- ASPP for context aggregation.  
- Optional decoder for boundary refinement.  

# Connections
- Successor to **FCN (2015)**.  
- Contemporary with **Mask R-CNN (2017)** (instance segmentation).  
- Predecessor to **PointRend (2020)** and **SegFormer (2021)**.  

# Implementation Notes
- DeepLabv3+ is the most widely used.  
- TensorFlow implementation still used in applied pipelines.  
- Lightweight enough for practical deployment compared to transformers.  

# Critiques / Limitations
- Limited boundary sharpness (improved by PointRend).  
- Relies heavily on strong backbones (ResNet/Xception).  
- Less generalizable than transformer-based models.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What semantic segmentation is (class per pixel).  
- Why receptive field size matters.  
- How atrous convolution expands “field of view” without downsampling.  
- Example: recognizing both the wheel and the car by using larger receptive fields.  

## Postgraduate-Level Concepts
- Multi-scale context aggregation (ASPP).  
- Trade-offs between CRF post-processing and end-to-end learning.  
- Encoder–decoder segmentation designs.  
- DeepLab’s influence on modern transformer segmenters (SegFormer, SAM).  

---

# My Notes
- DeepLab = **the ResNet of segmentation** → became the workhorse baseline.  
- ASPP was an elegant fix for multi-scale context before transformers.  
- Open question: Are atrous convolutions still useful in the foundation model era, or fully replaced by attention?  
- Possible extension: Combine **ASPP-style multi-scale features** with **ViT backbones** for hybrid segmentation.  

---
