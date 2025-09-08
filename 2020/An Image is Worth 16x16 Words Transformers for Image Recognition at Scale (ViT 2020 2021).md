---
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT, 2020/2021)"
aliases:
  - Vision Transformer
  - ViT
authors:
  - Alexey Dosovitskiy
  - Lucas Beyer
  - Alexander Kolesnikov
  - Dirk Weissenborn
  - Xiaohua Zhai
  - Thomas Unterthiner
  - Mostafa Dehghani
  - Matthias Minderer
  - Georg Heigold
  - Sylvain Gelly
  - Jakob Uszkoreit
  - Neil Houlsby
year: 2020 (arXiv), 2021 (ICLR)
venue: "ICLR 2021"
doi: "10.48550/arXiv.2010.11929"
arxiv: "https://arxiv.org/abs/2010.11929"
code: "https://github.com/google-research/vision_transformer"
citations: 20,000+
dataset:
  - ImageNet-21k (pretraining)
  - JFT-300M (internal large-scale dataset)
  - ImageNet-1k (fine-tuning)
tags:
  - paper
  - vision-transformer
  - architecture
  - image-classification
fields:
  - vision
  - deep-learning
  - transformers
related:
  - "[[Attention Is All You Need (2017)]]"
  - "[[DeiT (2021)]]"
  - "[[Swin Transformer (2021)]]"
predecessors:
  - "[[CNN-based Image Classification (ResNet, 2016)]]"
successors:
  - "[[DeiT (Data-efficient Image Transformers, 2021)]]"
  - "[[Swin Transformer (2021)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
This paper introduced the **Vision Transformer (ViT)**, a pure transformer architecture for image classification. By splitting images into non-overlapping patches (e.g., 16×16 pixels), treating them as tokens, and processing them with a standard transformer encoder, ViT demonstrated that transformers can **match and surpass CNNs** given enough data and compute.

# Key Idea
> Represent an image as a sequence of patch tokens, then apply a **transformer encoder** — no convolutions required.

# Method
- **Patch embeddings**: Image split into fixed-size patches (e.g., 16×16). Each patch is linearly projected to a vector.  
- **Positional embeddings**: Added to preserve spatial order.  
- **Transformer encoder**: Stacked multi-head self-attention + feed-forward layers.  
- **Classification token [CLS]**: Aggregates information for classification output.  
- **Training**:  
  - Pretrain on large datasets (JFT-300M, ImageNet-21k).  
  - Fine-tune on downstream tasks (ImageNet-1k).  

# Results
- On ImageNet, ViT surpassed state-of-the-art CNNs (ResNet, EfficientNet) when pretrained on massive datasets.  
- Showed scaling benefits: performance improves smoothly with model size and dataset scale.  
- With smaller datasets, CNNs outperformed ViT due to inductive bias advantages.  

# Why it Mattered
- Broke the **CNN dominance** in vision.  
- Showed that transformers are a **general-purpose architecture** across language and vision.  
- Sparked a wave of vision transformer research (DeiT, Swin, ConvNeXt).  

# Architectural Pattern
- Tokenization (patch embedding).  
- Transformer encoder (multi-head self-attention).  
- CLS token for classification.  

# Connections
- Inspired by NLP’s BERT/transformer models.  
- Predecessor to **DeiT** (data-efficient training), **Swin Transformer** (hierarchical windows), and hybrid CNN-transformer architectures.  

# Implementation Notes
- Requires massive pretraining datasets to outperform CNNs.  
- ViT-B/16, ViT-L/16, ViT-H/14 are standard model variants.  
- Hugely scalable: performance scales with data and compute.  

# Critiques / Limitations
- Data hungry: poor performance without large-scale pretraining.  
- Computationally expensive.  
- Lacks convolutional inductive biases (translation equivariance, locality).  

---

# Educational Connections

## Undergraduate-Level Concepts
- How transformers process sequences (attention mechanism).  
- How images can be restructured into tokens.  
- Difference between CNN inductive biases and transformer flexibility.  

## Postgraduate-Level Concepts
- Scaling laws for deep learning (bigger data → better performance).  
- Trade-offs between inductive bias and flexibility.  
- Transfer learning and fine-tuning strategies.  

---

# My Notes
- ViT is **architectural revolution**: proved transformers are not just for language.  
- Relevant for video: treat frames or spatio-temporal patches as tokens → video transformers.  
- Open question: Can transformers replace CNNs in efficiency-critical tasks?  
- Possible extension: Patch-tokenization ideas integrated into **video diffusion models** for editing.  

---
