---
title: "XGrad-CAM: Visual Explanations via Normalized Gradients (2020)"
aliases:
  - XGrad-CAM
  - Normalized Grad-CAM
authors:
  - Md Amirul Islam
  - Sen Jia
  - Neil D. B. Bruce
year: 2020
venue: "ECCV"
doi: "10.1007/978-3-030-58520-4_36"
arxiv: "https://arxiv.org/abs/2008.02312"
citations: 600+
tags:
  - paper
  - interpretability
  - explainability
  - deep-learning
  - visualization
fields:
  - computer-vision
  - deep-learning
  - explainable-ai
related:
  - "[[Grad-CAM (2017)]]"
  - "[[Score-CAM (2019)]]"
  - "[[Ablation-CAM (2020)]]"
  - "[[Layer-CAM (2021)]]"
predecessors:
  - "[[Grad-CAM (2017)]]"
successors:
  - "[[Layer-CAM (2021)]]"
impact: ⭐⭐⭐☆
status: "read"

---

# Summary
**XGrad-CAM (Islam et al., ECCV 2020)** improved upon Grad-CAM by introducing **gradient normalization**. It adjusted the weighting of feature maps using normalized gradients, making saliency maps more stable and faithful across inputs and classes.

# Key Idea
> Normalize gradients before weighting feature maps → prevents domination by extreme values and yields more reliable heatmaps.

# Method
- Forward pass: extract convolutional feature maps.  
- Backward pass: compute gradients of class score w.r.t. feature maps.  
- Normalize gradients → robust importance weights.  
- Weighted sum of feature maps → heatmap.  

# Results
- Reduced noise compared to Grad-CAM.  
- Better consistency across inputs and target classes.  
- More faithful localization of discriminative regions.  

# Why it Mattered
- Addressed instability in Grad-CAM caused by raw gradient values.  
- Improved reliability of saliency maps in multi-class settings.  
- Lightweight fix → easy adoption in existing Grad-CAM workflows.  

# Architectural Pattern
- Grad-CAM pipeline with gradient normalization step.  

# Connections
- Direct refinement of **Grad-CAM (2017)**.  
- Related to **Score-CAM** (gradient-free) and **Layer-CAM** (pixel-wise).  
- Predecessor to **hierarchical CAM variants**.  

# Implementation Notes
- Minimal code change from Grad-CAM.  
- No multiple forward passes (unlike Score-CAM).  
- Choice of normalization scheme affects results.  

# Critiques / Limitations
- Still gradient-dependent (can inherit gradient noise).  
- Limited improvement in low-resolution feature layers.  
- Less interpretable for highly abstract layers.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Grad-CAM highlights important regions but gradients can be noisy.  
- Normalizing gradients = avoid some pixels overpowering the map.  
- Example: dog classifier heatmap spreads across body, not just one hotspot.  

## Postgraduate-Level Concepts
- Gradient normalization for attribution.  
- Comparing attribution stability across Grad-CAM, Score-CAM, and XGrad-CAM.  
- Evaluation metrics for saliency faithfulness.  
- Broader link: normalization in explainability methods.  

---

# My Notes
- XGrad-CAM = **a cleaner Grad-CAM**.  
- Good trade-off: stable maps without heavy compute (unlike Score-CAM).  
- Open question: Can normalization + attention unify CAMs across CNNs and ViTs?  
- Possible extension: XGrad-CAM applied to multimodal transformers.  

---
