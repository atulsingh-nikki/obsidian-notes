---
title: "Learning Deep Features for Discriminative Localization (2016)"
aliases:
  - CAM
  - Class Activation Maps
  - Zhou et al. 2016
authors:
  - Bolei Zhou
  - Aditya Khosla
  - Àgata Lapedriza
  - Aude Oliva
  - Antonio Torralba
year: 2016
venue: "CVPR"
doi: "10.1109/CVPR.2016.319"
arxiv: "https://arxiv.org/abs/1512.04150"
citations: 20,000+
tags:
  - paper
  - deep-learning
  - explainability
  - visualization
  - cnn
fields:
  - computer-vision
  - deep-learning
  - interpretability
related:
  - "[[Zeiler & Fergus: Visualizing and Understanding Convolutional Networks (2014)]]"
  - "[[Grad-CAM (2017)]]"
  - "[[ResNet (2015)]]"
predecessors:
  - "[[Zeiler & Fergus: Visualizing and Understanding Convolutional Networks (2014)]]"
successors:
  - "[[Grad-CAM (2017)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Zhou et al. (CVPR 2016)** introduced **Class Activation Maps (CAMs)**, showing that global average pooling (GAP) in CNNs can localize discriminative image regions without explicit bounding box supervision. This was a key step in **weakly supervised object localization**.

# Key Idea
> By replacing fully connected layers with **global average pooling (GAP)**, CNNs retain spatial information, enabling the visualization of **discriminative regions** responsible for classification decisions.

# Method
- Modify CNN architecture by removing dense layers → add GAP before final softmax.  
- Compute **Class Activation Maps (CAMs)**:  
  - Weight feature maps by class scores.  
  - Overlay them on the image to highlight discriminative regions.  
- Train on image-level labels (no bounding boxes).  

# Results
- Localized objects with only image-level supervision.  
- Showed CNNs can implicitly learn to **attend to object regions**.  
- Demonstrated interpretability of deep features.  

# Why it Mattered
- Introduced **CAMs**, foundational for model interpretability in CV.  
- Sparked extensive follow-up work: Grad-CAM, Score-CAM, etc.  
- Proved GAP is effective for both classification and localization.  

# Architectural Pattern
- CNN backbone → GAP → softmax.  
- Discriminative region visualization via weighted sum of feature maps.  

# Connections
- Builds on visualization work (Zeiler & Fergus, 2014).  
- Predecessor to Grad-CAM (2017), which generalized CAM to many CNNs.  
- Influential in explainable AI (XAI).  

# Implementation Notes
- Requires GAP-based CNNs (e.g., modified AlexNet, GoogLeNet).  
- Works best with networks trained on classification tasks.  
- Visualization via heatmap overlay.  

# Critiques / Limitations
- Only works with GAP-structured CNNs (not generic).  
- Resolution of CAMs limited by CNN feature map downsampling.  
- Later methods improved generality and resolution.  

---

# Educational Connections

## Undergraduate-Level Concepts
- CNNs don’t just classify — they also learn *where* the object is.  
- GAP = averaging across feature map instead of dense layers.  
- CAM heatmaps = show what part of the image the CNN “looked at.”  

## Postgraduate-Level Concepts
- Weakly supervised localization: bounding boxes from classification labels.  
- Feature map weighting by class-specific softmax weights.  
- CAMs as precursors to explainable AI methods.  
- Limitations of CAM vs Grad-CAM.  

---

# My Notes
- CAM = **bridge between CNN classification and localization**.  
- Important both for **explainability** and **weakly supervised learning**.  
- Open question: How will CAM-like mechanisms evolve with transformers (attention maps are already interpretable)?  
- Possible extension: CAMs in multimodal models (vision-language grounding).  

---
