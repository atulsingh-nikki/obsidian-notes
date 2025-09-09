---
title: "ConvNeXt: A ConvNet for the 2020s (2022)"
aliases:
  - ConvNeXt
  - Modernized CNN
authors:
  - Zhuang Liu
  - Hanzi Mao
  - Chao-Yuan Wu
  - Christoph Feichtenhofer
  - Trevor Darrell
  - Saining Xie
year: 2022
venue: "CVPR"
doi: "10.1109/CVPR52688.2022.01093"
arxiv: "https://arxiv.org/abs/2201.03545"
code: "https://github.com/facebookresearch/ConvNeXt"
citations: ~4000+
dataset:
  - ImageNet-1k
  - COCO
  - ADE20K
tags:
  - paper
  - convnet
  - vision-transformer
  - image-classification
  - detection
  - segmentation
fields:
  - vision
  - deep-learning
  - architectures
related:
  - "[[Swin Transformer (2021)]]"
  - "[[ViT (2020/2021)]]"
predecessors:
  - "[[ResNet (2016)]]"
successors:
  - "[[ConvNeXt V2 (2023)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**ConvNeXt** reimagined convolutional neural networks (CNNs) with **transformer-inspired design choices**, showing that with proper modernization, pure ConvNets can match or outperform Vision Transformers (ViT, Swin) on large-scale benchmarks. It revived CNNs as competitive backbones for classification, detection, and segmentation.

# Key Idea
> Apply design patterns from transformers (e.g., layer scaling, large kernels, normalization tweaks) to **modernize CNNs**, making them competitive in the transformer era.

# Method
- **Architectural updates**:  
  - Large kernel sizes (7×7 depthwise conv) to mimic global receptive fields.  
  - Inverted bottlenecks (like MobileNetV2).  
  - LayerNorm instead of BatchNorm.  
  - Simplified stage design with patchify stem (like ViTs).  
  - GELU activations instead of ReLU.  
- **Scaling strategy**: Models scaled from ConvNeXt-Tiny → ConvNeXt-Large.  
- **Training tricks**: Heavy augmentations (RandAugment, Mixup, CutMix), AdamW optimizer, cosine decay — matching ViT training protocols.  

# Results
- **ImageNet-1k**: ConvNeXt-L achieved 87.8% top-1 accuracy, rivaling Swin-L.  
- **COCO (detection)**: Strong performance as backbone in Mask R-CNN/RetinaNet.  
- **ADE20K (segmentation)**: On par with Swin Transformers.  
- Proved that CNNs are still competitive with transformers when modernized.  

# Why it Mattered
- Challenged the “transformers have replaced CNNs” narrative.  
- Showed that architecture + training recipe, not just attention, drive performance.  
- Gave practitioners a **familiar, efficient backbone** competitive with ViTs.  

# Architectural Pattern
- Stage-based CNN (like ResNet).  
- Transformer-inspired updates: LayerNorm, GELU, patchify stem, large kernels.  

# Connections
- Successor to ResNet (2016), but redesigned.  
- Competitor to Swin Transformer (2021).  
- Inspired ConvNeXt V2 (2023, self-supervised + masked autoencoder integration).  

# Implementation Notes
- Trains with transformer-style recipe (AdamW, augmentations).  
- Patchify stem replaces conv7×7 + pooling.  
- LayerNorm critical for stability.  

# Critiques / Limitations
- Still less parameter-efficient than transformers at very large scale.  
- Lacks flexibility of attention for multimodal or long-range reasoning.  
- Mainly a classification/detection backbone, less used for generative tasks.  

---

# Educational Connections

## Undergraduate-Level Concepts
- CNN vs Transformer architectural differences.  
- Why kernel size, normalization, and activations matter.  
- Training protocol influences results as much as architecture.  

## Postgraduate-Level Concepts
- Hybridization of CNN and transformer design patterns.  
- Inductive bias vs flexibility trade-offs.  
- Modern large-scale training recipes (AdamW, heavy augmentations).  

---

# My Notes
- ConvNeXt feels like a **reminder**: CNNs aren’t obsolete, they just needed a refresh.  
- Strong candidate backbone for **efficiency-focused deployments** where transformers are heavy.  
- Open question: Can ConvNeXt principles extend to **video backbones** with temporal kernels?  
- Possible extension: Hybrid **ConvNeXt + temporal attention** for video editing and diffusion models.  

---
