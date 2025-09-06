---
title: "DeiT: Training Data-Efficient Image Transformers & Distillation through Attention (2021)"
aliases:
  - DeiT
  - Data-Efficient Image Transformers
authors:
  - Hugo Touvron
  - Matthieu Cord
  - Matthijs Douze
  - Francisco Massa
  - Alexandre Sablayrolles
  - Hervé Jégou
year: 2021
venue: "ICML"
doi: "10.48550/arXiv.2012.12877"
arxiv: "https://arxiv.org/abs/2012.12877"
code: "https://github.com/facebookresearch/deit"
citations: ~5000+
dataset:
  - ImageNet-1k (no external data)
tags:
  - paper
  - vision-transformer
  - image-classification
  - distillation
fields:
  - vision
  - transformers
  - efficient-training
related:
  - "[[An Image is Worth 16x16 Words (ViT, 2020/2021)]]"
  - "[[Swin Transformer (2021)]]"
predecessors:
  - "[[ViT (2020/2021)]]"
successors:
  - "[[Swin Transformer (2021)]]"
  - "[[ConvNeXt (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**DeiT** made Vision Transformers (ViT) practical by showing they can be trained **data-efficiently** on **ImageNet-1k from scratch**, without requiring massive pretraining datasets. The key innovation was **distillation through attention**, where a CNN teacher guides the transformer student not only via class labels but also through token-level attention.

# Key Idea
> Transformers don’t inherently need billions of images — with the right training tricks (strong augmentation, optimization, and distillation), they can match CNNs on ImageNet-1k alone.

# Method
- **Training from scratch**: On ImageNet-1k (1.3M images), no JFT-300M pretraining.  
- **Distillation token**:  
  - Added alongside [CLS] token.  
  - Learns directly from a CNN teacher (e.g., RegNet, EfficientNet).  
  - Guides transformer’s representation learning.  
- **Regularization & Augmentation**: Mixup, CutMix, RandAugment, stochastic depth, repeated augmentation.  
- **Architectures**: DeiT-Tiny, DeiT-Small, DeiT-Base (lighter than ViT).  

# Results
- DeiT matched or outperformed ResNets on ImageNet with comparable compute.  
- Achieved strong results **without extra data**.  
- Distillation token gave consistent improvements over label-only supervision.  
- Proved transformers could be practical beyond Google-scale compute.  

# Why it Mattered
- Removed ViT’s biggest barrier: **data hunger**.  
- Democratized vision transformers (trainable by labs without huge private datasets).  
- Open-sourced pretrained models, accelerating adoption.  
- Set the stage for transformer variants (Swin, ConvNeXt).  

# Architectural Pattern
- Same as ViT (patch embeddings + transformer encoder).  
- Added **distillation token** to leverage teacher knowledge.  

# Connections
- Successor to **ViT**.  
- Predecessor to hierarchical transformers (Swin).  
- Connected to knowledge distillation literature (Hinton et al.).  

# Implementation Notes
- Strong reliance on augmentation & regularization.  
- Teacher network is critical for distillation benefits.  
- Smaller models (DeiT-Tiny) achieve good trade-offs for efficiency.  

# Critiques / Limitations
- Still computationally expensive compared to CNNs for small-scale tasks.  
- Teacher-student setup adds complexity.  
- Later architectures (Swin, ConvNeXt) improved scalability and efficiency further.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Transformers vs CNNs in vision.  
- Knowledge distillation basics (teacher → student).  
- Data augmentation strategies.  

## Postgraduate-Level Concepts
- Distillation tokens and attention-based supervision.  
- Efficient transformer training strategies.  
- Trade-offs between data, architecture, and supervision.  

---

# My Notes
- DeiT made ViT **accessible**: ImageNet-only training changed the game.  
- Feels like the “ResNet moment” for transformers: broad adoption.  
- Open question: Can distillation tokens be generalized to **video transformers** (teacher-guided temporal consistency)?  
- Possible extension: Distillation through attention for **diffusion models** — teacher guiding student with feature-level supervision.  

---
