---
title: "Group Normalization (2018)"
aliases: 
  - GN
authors:
  - Yuxin Wu
  - Kaiming He
year: 2018
venue: "ECCV"
doi: "10.1007/978-3-030-01261-8_32"
arxiv: "https://arxiv.org/abs/1803.08494"
code: "https://github.com/ppwwyyxx/GroupNorm-reproduce"
citations: 6000+
dataset:
  - ImageNet
  - COCO (for detection/segmentation tasks)
tags:
  - paper
  - normalization
  - deep-learning
fields:
  - vision
  - optimization
  - architectures
related:
  - "[[Batch Normalization (2015)]]"
  - "[[Layer Normalization (2016)]]"
predecessors:
  - "[[Batch Normalization (2015)]]"
  - "[[Layer Normalization (2016)]]"
successors:
  - "[[Switchable Normalization (2019)]]"
  - "[[RMSNorm (2021)]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
Group Normalization (GN) is a normalization technique that normalizes feature channels by dividing them into **groups**. Unlike Batch Normalization (BN), GN’s computation is **independent of batch size**, making it particularly effective for tasks with small or variable batch sizes (e.g., detection, segmentation, video).

# Key Idea
> Normalize features by groups of channels, not across the batch, reducing dependency on batch statistics.

# Method
- Splits channels into **G groups** (e.g., 32 channels per group).  
- Computes mean and variance per group across spatial dimensions.  
- Normalizes each group independently:  
  $$
  \hat{x} = \frac{x - \mu_{group}}{\sqrt{\sigma^2_{group} + \epsilon}}
  $$  
- Group size is a hyperparameter; GN reduces to Layer Norm when G=1, and Instance Norm when G=channels.  

# Results
- Comparable accuracy to BN on ImageNet classification.  
- Outperformed BN on detection and segmentation (COCO), where batch sizes are typically small.  
- Stable across a wide range of tasks without needing large batches.  

# Why it Mattered
- Removed reliance on **large-batch training**, addressing a key limitation of BN.  
- Enabled effective training of detection/segmentation models on GPUs with limited memory.  
- Highlighted that **batch statistics aren’t essential** for normalization.  

# Architectural Pattern
- Per-group normalization across channels + spatial dimensions.  
- Fully deterministic during training and inference.  
- Generalizes BN, LayerNorm, and InstanceNorm.  

# Connections
- **Contemporaries**: Weight Normalization, Instance Normalization.  
- **Influence**: Switchable Normalization, RMSNorm in Transformers.  

# Implementation Notes
- Default choice: 32 groups.  
- Slightly higher computational cost than BN, but batch-size independence often worth it.  
- Works well in convolutional architectures and dense prediction tasks.  

# Critiques / Limitations
- Less effective than BN when very large batches are available.  
- Group size tuning may affect performance.  
- Not as dominant in NLP (LayerNorm is standard there).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1803.08494)  
- [PyTorch implementation](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)  
- [Reproduction repo](https://github.com/ppwwyyxx/GroupNorm-reproduce)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Mean/variance normalization per group.  
- **Probability & Statistics**: Variance estimation from small samples.  
- **Optimization Basics**: Stability from normalization techniques.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Normalization strategies in deep networks.  
- **Computer Vision**: Impact of batch size in dense prediction tasks.  
- **Research Methodology**: Benchmarking across classification vs detection.  
- **Advanced Optimization**: Trade-offs between BN, GN, LN, and IN.  

---

# My Notes
- Relevant for **video editing ML pipelines** where small batch sizes are common due to memory limits.  
- Open question: Can **normalization-free transformers (NFNets)** remove the need for GN-like methods?  
- Possible extension: Apply GN or hybrid norms in **diffusion U-Nets** where BN fails due to batch-size constraints.  

---
