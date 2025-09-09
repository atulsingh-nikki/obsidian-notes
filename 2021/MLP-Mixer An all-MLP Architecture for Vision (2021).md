---
title: "MLP-Mixer: An all-MLP Architecture for Vision (2021)"
aliases:
  - MLP-Mixer
  - All-MLP Vision Model
authors:
  - Ilya Tolstikhin
  - Neil Houlsby
  - Alexander Kolesnikov
  - Lucas Beyer
  - Xiaohua Zhai
  - Thomas Unterthiner
  - Jessica Yung
  - Andreas Steiner
  - Daniel Keysers
  - Jakob Uszkoreit
  - Mario Lucic
  - Alexey Dosovitskiy
year: 2021
venue: "NeurIPS"
doi: "10.48550/arXiv.2105.01601"
arxiv: "https://arxiv.org/abs/2105.01601"
code: "https://github.com/google-research/vision_transformer"
citations: 896+
dataset:
  - ImageNet-1k
  - JFT-300M
tags:
  - paper
  - mlp
  - architecture
  - vision
  - deep-learning
fields:
  - vision
  - deep-learning
  - architectures
related:
  - "[[Vision Transformer (ViT, 2020)]]"
  - "[[ResMLP (2021)]]"
  - "[[gMLP (2021)]]"
predecessors:
  - "[[MLPs in Early Vision (Pre-CNN Era)]]"
successors:
  - "[[ResMLP (2021)]]"
  - "[[gMLP (2021)]]"
  - "[[ConvNeXt (2022)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**MLP-Mixer** presented the first **competitive image classification architecture built entirely from MLP layers**, without convolutions or self-attention. It showed that with sufficient scale and data, **simple MLP-based mixing of spatial and channel dimensions** can achieve performance comparable to CNNs and Transformers.

# Key Idea
> Separate **spatial (token-mixing)** and **channel (feature-mixing)** interactions, and model them using only MLPs, proving that convolutions or attention aren’t strictly necessary for high-performance vision models.

# Method
- **Input**: Images divided into non-overlapping patches (like ViT).  
- **Architecture**: Alternating blocks of:  
  - **Token-mixing MLP**: Operates across spatial positions for each channel.  
  - **Channel-mixing MLP**: Operates across feature channels for each token.  
- **Design**: Simple, uniform architecture (stack of Mixer layers).  
- **Training**: Large-scale JFT-300M dataset, transfer to ImageNet.  

# Results
- Achieved competitive ImageNet accuracy (comparable to ViT and ResNet when trained on large data).  
- Simpler architecture, easier to implement than attention.  
- Demonstrated scalability of MLP-based designs.  

# Why it Mattered
- Showed **attention is not the only alternative to convolution** for vision.  
- Sparked a wave of **all-MLP models** (ResMLP, gMLP).  
- Encouraged exploration of simplicity vs complexity in architectures.  

# Architectural Pattern
- Patch embeddings → stack of Mixer blocks.  
- Each block: Token-Mixing MLP + Channel-Mixing MLP.  
- Final classification head.  

# Connections
- Related to **Vision Transformer (ViT)** (patch embeddings, non-conv input).  
- Inspired **ResMLP, gMLP** as other all-MLP attempts.  
- Predecessor to hybrid architectures (ConvNeXt revisiting CNNs).  

# Implementation Notes
- Needs very large datasets (JFT-300M) for best performance.  
- Underperforms on smaller datasets without heavy regularization.  
- Extremely simple to implement (few lines of PyTorch/TF code).  

# Critiques / Limitations
- Not as data-efficient as CNNs.  
- Struggles with small datasets and transfer learning.  
- Largely overshadowed by Transformers + ConvNeXt in follow-ups.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Difference between CNNs, Transformers, and MLPs.  
- What patch embeddings are and why they simplify input.  
- Concept of **token-mixing vs channel-mixing**.  
- Why simple architectures can scale well with data.  

## Postgraduate-Level Concepts
- Comparison of **MLP-Mixer vs ViT vs CNNs** in inductive bias.  
- Data efficiency vs scalability trade-offs.  
- The role of architectural simplicity in foundation models.  
- Lessons: scaling laws apply beyond attention or convolution.  

---

# My Notes
- MLP-Mixer was a **provocative result**: even “dumb” MLPs can compete at scale.  
- A reminder that **inductive bias isn’t everything** if enough data + compute are available.  
- Open question: Can MLP simplicity be merged with modern efficiency tricks (like ConvNeXt)?  
- Possible extension: Use MLP-Mixer blocks in **multimodal or generative models** for lightweight mixing.  

---
