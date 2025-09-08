---
title: "FullGrad: A Class Activation Method for Deep Networks (2020)"
aliases:
  - FullGrad
  - Full Gradient Attribution
authors:
  - Suraj Srinivas
  - François Fleuret
year: 2020
venue: "NeurIPS"
doi: "10.5555/3495724.3497163"
arxiv: "https://arxiv.org/abs/1905.00780"
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
  - "[[Smooth Grad-CAM++ (2019)]]"
  - "[[XGrad-CAM (2020)]]"
predecessors:
  - "[[Grad-CAM (2017)]]"
successors:
  - "[[Transformer Attribution Maps (2020s)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**FullGrad (Srinivas & Fleuret, NeurIPS 2020)** proposed a **principled attribution method** that leverages **full gradients of the prediction w.r.t. inputs and intermediate biases**. Unlike CAM variants that focus on last-layer activations, FullGrad aggregates contributions across **all layers**, producing more complete and faithful explanations.

# Key Idea
> Attribution should include **all gradient paths**, not just final-layer activations. FullGrad combines input gradients with bias gradients across every layer to form a unified saliency map.

# Method
- Compute gradients of the output w.r.t.:  
  - Input features.  
  - Bias terms at each intermediate layer.  
- Aggregate contributions to form **FullGrad map**.  
- Normalize and visualize as a saliency heatmap.  

# Results
- More **faithful** than Grad-CAM variants.  
- Captured fine-grained details and global context together.  
- Worked consistently across architectures (CNNs, residual nets).  

# Why it Mattered
- First attribution method with a **clear theoretical grounding**.  
- Showed the role of **bias terms** in attributions.  
- Bridged gradient-based and CAM-style explanations.  

# Architectural Pattern
- Gradient + bias attribution across layers.  
- Summation → unified heatmap.  

# Connections
- Related to gradient × input methods (saliency maps).  
- Generalizes CAM beyond final-layer dependence.  
- Predecessor to transformer explainability approaches.  

# Implementation Notes
- More compute than Grad-CAM (gradients at all layers).  
- Produces saliency maps with less bias toward last-layer features.  
- Works well for CNNs with biases.  

# Critiques / Limitations
- Still gradient-dependent (susceptible to noise).  
- Visualization resolution tied to feature map structure.  
- Less popular in practice compared to Grad-CAM family.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Grad-CAM looks only at last layer → partial picture.  
- FullGrad = “look at the whole network,” not just the end.  
- Example: highlighting both global outline of a dog and fine details like ears.  

## Postgraduate-Level Concepts
- Attribution via input + bias gradients.  
- Faithfulness evaluation metrics for attribution methods.  
- Comparison with integrated gradients and Shapley-based methods.  
- Theoretical guarantees vs heuristic CAM variants.  

---

# My Notes
- FullGrad = **CAM with theory** → principled and complete.  
- Stronger than heuristic CAMs but less widely adopted.  
- Open question: How do FullGrad-style methods transfer to ViTs without biases?  
- Possible extension: Apply FullGrad ideas to multimodal transformers (bias attribution in cross-attention).  

---
