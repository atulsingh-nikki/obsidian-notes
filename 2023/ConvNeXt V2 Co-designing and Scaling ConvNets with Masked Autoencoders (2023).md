---
title: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders (2023)"
aliases:
  - ConvNeXt V2
  - Modernized CNN with MAE
authors:
  - Sanghyun Woo
  - Alexander Kirillov
  - Alan Yuille
  - Saining Xie
year: 2023
venue: "CVPR"
doi: "10.1109/CVPR52729.2023.01947"
arxiv: "https://arxiv.org/abs/2301.00808"
code: "https://github.com/facebookresearch/ConvNeXt-V2"
citations: ~500+
dataset:
  - ImageNet-1k
  - ImageNet-22k
  - COCO
  - ADE20K
tags:
  - paper
  - convnet
  - self-supervised
  - mae
  - representation-learning
fields:
  - vision
  - architectures
  - pretraining
related:
  - "[[ConvNeXt (2022)]]"
  - "[[Masked Autoencoders (MAE, 2021)]]"
predecessors:
  - "[[ConvNeXt (2022)]]"
successors:
  - "[[Hybrid CNN-Transformer Pretraining (2023+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**ConvNeXt V2** extends **ConvNeXt** by integrating **masked autoencoder (MAE) pretraining** with CNNs, showing that convolutional models can benefit from self-supervised pretraining strategies originally designed for transformers. It also refines the ConvNeXt design for better scaling and robustness.

# Key Idea
> Co-design ConvNets with **MAE pretraining**, proving that CNNs can serve as strong backbones in the self-supervised learning era, not just transformers.

# Method
- **Masked Autoencoder with CNNs**:  
  - Randomly mask image patches.  
  - Train ConvNeXt encoder to reconstruct missing patches.  
  - Decoder lightweight compared to encoder.  
- **Architectural refinements**:  
  - Streamlined design with scaled depth/width.  
  - Normalization and activation tweaks.  
- **Scaling**: Trained from ConvNeXt-Tiny up to ConvNeXt-XL with MAE pretraining.  

# Results
- **ImageNet-1k**: Outperformed ConvNeXt baselines.  
- **ImageNet-22k pretraining + fine-tuning**: Competitive with transformer MAEs.  
- **COCO detection / ADE20K segmentation**: Strong improvements, setting new benchmarks for CNN-based backbones.  

# Why it Mattered
- Showed that CNNs remain relevant even in the **self-supervised foundation model era**.  
- Bridged **MAE-style SSL** with convolutional architectures.  
- Reinforced CNNs as viable alternatives to ViTs for efficiency-focused tasks.  

# Architectural Pattern
- ConvNeXt backbone encoder.  
- MAE decoder for patch reconstruction.  
- Refinements for scaling to large models.  

# Connections
- Extends ConvNeXt (2022).  
- Inspired by **Masked Autoencoders (ViT-based, He et al., 2021)**.  
- Competes with ViTs in SSL benchmarks.  

# Implementation Notes
- Pretraining requires large compute but scales well.  
- Lightweight decoder avoids compute bottlenecks.  
- Public code and pretrained models available.  

# Critiques / Limitations
- Still lacks long-range modeling flexibility of transformers.  
- Masked pretraining may not fully leverage CNN inductive biases.  
- ViTs still dominate in multimodal/foundation models.  

---

# Educational Connections

## Undergraduate-Level Concepts
- CNN vs Transformer backbones.  
- Masked autoencoder principle.  
- Why self-supervised pretraining matters.  

## Postgraduate-Level Concepts
- Co-design of architectures + pretraining methods.  
- CNN scaling laws vs transformer scaling laws.  
- Transfer learning to detection and segmentation.  

---

# My Notes
- ConvNeXt V2 is the **"CNN comeback" in the SSL era**.  
- Shows that pretraining recipes matter as much as architecture.  
- Open question: Can CNN-based MAEs extend to **video pretraining** (masked temporal modeling)?  
- Possible extension: Hybrid CNN-Transformer MAE backbones for efficient **video diffusion editing**.  

---
