---
title: "A Guide to Convolution Arithmetic for Deep Learning (2016)"
aliases:
  - Convolution Arithmetic Guide
  - Dumoulin & Visin 2016
authors:
  - Vincent Dumoulin
  - Francesco Visin
year: 2016
venue: "arXiv preprint"
arxiv: "https://arxiv.org/abs/1603.07285"
citations: 6000+
tags:
  - survey
  - cnn
  - convolution
  - deep-learning
fields:
  - machine-learning
  - deep-learning
  - computer-vision
related:
  - "[[LeNet-5 (1998)]]"
  - "[[AlexNet (2012)]]"
  - "[[VGGNet (2014)]]"
  - "[[ResNet (2015)]]"
predecessors:
  - "[[Convolution basics in signal processing]]"
successors:
  - "[[Modern CNN tutorials (2020+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**Dumoulin & Visin (2016)** published a practical **guide to convolution arithmetic** in deep learning, explaining how convolutional layers, padding, stride, and pooling affect the **output dimensions** of CNNs. The paper became a canonical reference for practitioners and students.

# Key Idea
> Provide a clear and mathematical framework for computing the **input–output tensor dimensions** in convolutional and pooling layers.

# Content Overview
- **Convolution basics**: kernel size, stride, padding, dilation.  
- **Output shape formulas**:  
  - Without padding: `O = floor((I - K)/S) + 1`  
  - With padding: `O = floor((I + 2P - K)/S) + 1`  
- **Types of padding**: valid, same, full.  
- **Transposed convolutions** (a.k.a. deconvolutions).  
- **Pooling arithmetic** (max/average pooling).  
- Visual diagrams explaining stride and padding.  

# Results
- Clarified convolution shape math for the community.  
- Widely cited as a reference guide in deep learning courses and libraries.  
- Improved understanding of **deconvolutional layers** for generative models.  

# Why it Mattered
- Filled a major gap in practical CNN education.  
- Provided precise formulas + diagrams that remain standard in tutorials.  
- Made CNNs more accessible to newcomers in 2016 (pre-modern frameworks).  

# Connections
- Directly relevant to CNN architectures like LeNet, AlexNet, VGG, ResNet.  
- Influenced deep learning textbooks and PyTorch/TensorFlow docs.  
- Related to **deconvolution in GANs** and **segmentation upsampling**.  

# Implementation Notes
- Simple formulas but crucial for debugging CNN architectures.  
- Especially important when designing encoder–decoder networks.  
- Visual intuition remains unmatched.  

# Critiques / Limitations
- Purely educational, no new algorithms.  
- Assumes some familiarity with CNN basics.  
- Doesn’t cover modern variants (depthwise conv, dilated conv in detail).  

---

# Educational Connections

## Undergraduate-Level Concepts
- How padding/stride affect output image size.  
- Why pooling reduces spatial resolution.  
- Example: a 32×32 image with a 3×3 kernel and stride 1 → 30×30 output.  

## Postgraduate-Level Concepts
- Transposed convolutions for upsampling.  
- Dilated convolutions in semantic segmentation.  
- Encoder–decoder CNN design (U-Net, DeepLab).  
- Importance of convolution arithmetic in **GANs** and **autoencoders**.  

---

# My Notes
- This guide = **the cheat sheet every CNN student needed**.  
- Still cited today whenever output sizes get confusing.  
- Open question: how to extend such guides to **attention layers** (sequence length, patch embeddings)?  
- Possible extension: a unified “arithmetic guide” for CNNs + Transformers.  

---
