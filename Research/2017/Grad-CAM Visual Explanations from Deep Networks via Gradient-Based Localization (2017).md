---
title: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization (2017)"
aliases:
  - Grad-CAM
  - Gradient-weighted Class Activation Mapping
authors:
  - Ramprasaath R. Selvaraju
  - Michael Cogswell
  - Abhishek Das
  - Ramakrishna Vedantam
  - Devi Parikh
  - Dhruv Batra
year: 2017
venue: "ICCV"
doi: "10.1109/ICCV.2017.74"
arxiv: "https://arxiv.org/abs/1610.02391"
citations: 30,000+
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
  - "[[Learning Deep Features for Discriminative Localization (2016)]]"
  - "[[Zeiler & Fergus: Visualizing and Understanding Convolutional Networks (2014)]]"
  - "[[Score-CAM (2019)]]"
predecessors:
  - "[[Learning Deep Features for Discriminative Localization (2016)]]"
successors:
  - "[[Score-CAM (2019)]]"
  - "[[Transformer Attention Visualization (2020s)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Grad-CAM (Selvaraju et al., ICCV 2017)** introduced **gradient-weighted class activation mapping**, generalizing CAM to work with **any CNN architecture**. By backpropagating gradients, Grad-CAM identifies which image regions most strongly influence a model’s decision, producing class-discriminative heatmaps.

# Key Idea
> Use **gradients of the target class** flowing into convolutional feature maps to weight their importance, generating a **visual explanation** of what the model focuses on.

# Method
- Forward pass: compute feature maps at a convolutional layer.  
- Backward pass: compute gradients of class score w.r.t. these feature maps.  
- Compute weighted sum of feature maps using gradient-based weights.  
- Apply ReLU → generate localization heatmap.  

# Results
- Produced high-quality visual explanations across CNN architectures.  
- Showed class-discriminative focus (e.g., highlighting a dog, not background).  
- Used in applications: weakly supervised localization, debugging, bias detection.  

# Why it Mattered
- Became the **standard interpretability tool** for CNNs.  
- Enabled **post-hoc visualization** without architectural changes.  
- Sparked a wave of explainability research (Score-CAM, Eigen-CAM, etc.).  

# Architectural Pattern
- CNN → feature maps → gradients → weighted combination → heatmap.  

# Connections
- Direct successor to **CAM (2016)**.  
- Predecessor to **Score-CAM (2019)** and transformer attention visualizations.  
- Related to attribution methods (saliency maps, LRP).  

# Implementation Notes
- Works on pretrained models (no retraining required).  
- Flexible: applies to classification, captioning, VQA models.  
- Layer choice affects heatmap granularity.  

# Critiques / Limitations
- Resolution limited by feature map downsampling.  
- Can be noisy or ambiguous for overlapping objects.  
- Gradient saturation may weaken signals.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Gradients tell us **what the network cares about** for a prediction.  
- Heatmaps = highlight image regions driving the classification.  
- Example: dog classifier highlights dog’s face, not background grass.  

## Postgraduate-Level Concepts
- Gradient-based attribution vs perturbation-based methods.  
- Weakly supervised object localization using Grad-CAM.  
- Comparison to CAM: no GAP requirement, works on any CNN.  
- Extensions: Score-CAM, Eigen-CAM, transformer attention maps.  

---

# My Notes
- Grad-CAM = **the de facto standard** for CNN interpretability.  
- Big leap from CAM → generalization to any architecture.  
- Open question: For transformers, are attention maps “built-in Grad-CAMs,” or do we still need attribution tools?  
- Possible extension: Grad-CAM for multimodal models (CLIP, VLMs).  

---
