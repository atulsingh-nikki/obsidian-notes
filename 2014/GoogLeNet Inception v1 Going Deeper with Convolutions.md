---
title: Going Deeper with Convolutions (2015)
aliases:
  - Inception v1
  - GoogLeNet
  - GoogLeNet (2015)/Inception v1(2015)
authors:
  - Christian Szegedy
  - Wei Liu
  - Yangqing Jia
  - Pierre Sermanet
  - Scott Reed
  - Dragomir Anguelov
  - Dumitru Erhan
  - Vincent Vanhoucke
  - Andrew Rabinovich
year: 2015
venue: CVPR
doi: 10.1109/CVPR.2015.7298594
arxiv: https://arxiv.org/abs/1409.4842
code: https://github.com/tensorflow/models/tree/master/research/slim#inception-v1-googlenet
citations: 70,000+
dataset:
  - ImageNet (ILSVRC 2014 classification challenge)
tags:
  - paper
  - cnn
  - inception
  - image-classification
fields:
  - vision
  - deep-learning
related:
  - "[[Inception v2-v3 (2016)]]"
  - "[[ResNet (2015)]]"
predecessors:
  - "[[VGGNet (2014)]]"
  - "[[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]]"
successors:
  - "[[Inception v2-v3 (2016)]]"
  - "[[Inception v4 (2016)]]"
  - "[[ResNet (2015)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
This paper introduced the **Inception architecture (GoogLeNet)**, winner of the **ILSVRC 2014 classification challenge**. It achieved state-of-the-art accuracy with fewer parameters by using **multi-scale convolutional filters** and dimension reduction with 1×1 convolutions.

# Key Idea
> Instead of committing to a single filter size, use **parallel multi-scale convolutions (1×1, 3×3, 5×5)** within the same module, and concatenate outputs — enabling richer representations at low computational cost.

# Method
- **Inception module**:  
  - Parallel branches with 1×1, 3×3, 5×5 convolutions and pooling.  
  - Concatenation of outputs across channels.  
- **1×1 convolutions**: Used for **dimension reduction**, reducing computational cost.  
- **Architecture**:  
  - GoogLeNet = 22 layers deep with stacked inception modules.  
  - Used auxiliary classifiers at intermediate layers for regularization.  
- Optimized for both **depth and computational efficiency**.  

# Results
- Won **ILSVRC 2014** with **6.67% top-5 error**, surpassing [[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]] and VGG.  
- Far fewer parameters (~5M) compared to VGG (~138M).  
- Achieved strong performance on image classification and detection tasks.  

# Why it Mattered
- Broke the trade-off between **depth and efficiency**.  
- Established the **Inception family** of architectures, widely adopted in industry.  
- Inspired hybrid models combining multi-scale filters and residual connections.  

# Architectural Pattern
- Multi-branch module with different receptive fields.  
- Dimension reduction with 1×1 convolutions.  
- Deep, efficient stacking of inception modules.  

# Connections
- **Contemporaries**: VGGNet (2014).  
- **Influence**: Inception v2–v4, ResNet (2015), EfficientNet.  

# Implementation Notes
- 1×1 convolutions key for efficiency.  
- Auxiliary classifiers help gradient flow but later found unnecessary.  
- Inception modules improved hardware utilization for CNNs.  

# Critiques / Limitations
- Hand-designed module structure.  
- Later models (ResNet, EfficientNet) showed better scalability.  
- Complex compared to simple residual blocks.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1409.4842)  
- [TensorFlow slim implementation](https://github.com/tensorflow/models/tree/master/research/slim#inception-v1-googlenet)  
- [PyTorch implementations available]  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Convolutions, dimensionality reduction.  
- **Probability & Statistics**: Regularization effects of auxiliary classifiers.  
- **Optimization Basics**: Training deep CNNs with reduced parameters.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Multi-branch architectures.  
- **Computer Vision**: Image classification benchmarks.  
- **Research Methodology**: Designing for accuracy vs efficiency trade-offs.  
- **Advanced Optimization**: Gradient propagation in deep CNNs.  

---

# My Notes
- Inception was a turning point: **deep but efficient** networks.  
- Still relevant to video ML (multi-scale spatio-temporal filters).  
- Open question: Can **inception-like multi-scale reasoning** combine with transformers/diffusion?  
- Possible extension: Apply inception-style blocks in **video diffusion U-Nets** for better temporal resolution.  

---
