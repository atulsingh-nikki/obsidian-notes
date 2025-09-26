---
title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (2021)"
aliases:
  - Swin Transformer
  - Shifted Window Transformer
authors:
  - Ze Liu
  - Yutong Lin
  - Yue Cao
  - Han Hu
  - Yixuan Wei
  - Zheng Zhang
  - Stephen Lin
  - Baining Guo
year: 2021
venue: ICCV
doi: 10.1109/ICCV48922.2021.00986
arxiv: https://arxiv.org/abs/2103.14030
code: https://github.com/microsoft/Swin-Transformer
citations: 15,000+
dataset:
  - ImageNet-1k
  - COCO (detection/segmentation)
  - ADE20K (semantic segmentation)
tags:
  - paper
  - vision-transformer
  - image-classification
  - detection
  - segmentation
fields:
  - vision
  - transformers
  - dense-prediction
related:
  - "[[An Image is Worth 16x16 Words Transformers for Image Recognition at Scale (ViT 2020 2021)|Vision Transformer]]"
  - "[[DeiT Training Data-Efficient Image Transformers & Distillation through Attention (2021)|DeiT]]"
  - "[[ConvNeXt A ConvNet for the 2020s (2022)|ConvNeXt]]"
predecessors:
  - "[[An Image is Worth 16x16 Words Transformers for Image Recognition at Scale (ViT 2020 2021)|ViT]]"
  - "[[DeiT Training Data-Efficient Image Transformers & Distillation through Attention (2021)|DeiT]]"
successors:
  - "[[SwinV2 (2022)]]"
  - "[[ConvNeXt A ConvNet for the 2020s (2022)|ConvNeXt]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
The **Swin Transformer** introduced a **hierarchical transformer architecture with shifted window attention**, making transformers scalable and efficient for vision tasks beyond classification — including **detection and segmentation**. By limiting self-attention to non-overlapping windows and shifting them across layers, Swin achieves both **linear computational complexity** and **cross-window connections**.

# Key Idea
> Replace global self-attention with **local windowed attention + shifted windows** to balance efficiency and representation power, while building a hierarchical feature pyramid like CNNs.

# Method
- **Hierarchical design**: Feature maps progressively downsampled, like CNN pyramids.  
- **Window-based self-attention**: Self-attention restricted to local non-overlapping windows for efficiency.  
- **Shifted window mechanism**: Alternate layers shift the window partition, enabling cross-window interactions without costly global attention.  
- **Applications**: Swin backbone plugged into detection/segmentation frameworks (Mask R-CNN, Cascade R-CNN, Semantic FPN).  

# Results
- **ImageNet-1k**: Achieved SOTA accuracy, scaling smoothly with model size.  
- **COCO detection**: Surpassed CNN backbones (ResNet, EfficientNet).  
- **ADE20K segmentation**: Achieved new SOTA in semantic segmentation.  
- Became the **default transformer backbone** for dense prediction tasks.  

# Why it Mattered
- Solved ViT/DeiT’s inefficiency in high-resolution vision tasks.  
- Brought transformers into detection and segmentation pipelines.  
- Sparked a new wave of hierarchical transformer backbones.  

# Architectural Pattern
- Patch embedding → hierarchical stages → shifted window attention.  
- Pyramid structure akin to ResNet/FPN.  

# Connections
- Successor to **ViT/DeiT** (classification focus).  
- Predecessor to **ConvNeXt (2022)** (CNNs reimagined in transformer style).  
- Directly extended by **SwinV2 (2022)**.  

# Implementation Notes
- Efficient: O(N) complexity with image size.  
- Widely adopted as backbone in detection frameworks.  
- Public models available (Swin-T, Swin-S, Swin-B, Swin-L).  

# Critiques / Limitations
- Still heavier than CNNs for some small-scale tasks.  
- Hand-crafted shifted window design — later works explored learned patterns.  
- Strong reliance on large-scale pretraining for best results.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Difference between global and local self-attention.  
- How CNN-like pyramids improve scalability.  
- Why shifting windows allows cross-region interactions.  

## Postgraduate-Level Concepts
- Complexity analysis: global vs windowed attention.  
- Transformer backbones for dense prediction tasks.  
- Connections between CNN inductive biases and transformer flexibility.  

---

# My Notes
- Swin was the **breakthrough** that made transformers usable beyond classification.  
- It feels like the **ResNet of transformers for vision** — strong, scalable, widely adopted.  
- Open question: Can hierarchical transformers be extended to **video** with temporal windows?  
- Possible extension: Use shifted windows in **video diffusion editing** for temporal consistency.  

---
