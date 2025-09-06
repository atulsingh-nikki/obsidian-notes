---
title: Gradient-Based Learning Applied to Document Recognition (1998)
aliases:
  - LeNet-5
  - LeCun 1998
authors:
  - Yann LeCun
  - Léon Bottou
  - Yoshua Bengio
  - Patrick Haffner
year: 1998
venue: Proceedings of the IEEE
doi: 10.1109/5.726791
url: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
citations: 60000+
dataset:
  - MNIST
tags:
  - paper
  - cnn
  - document-recognition
  - deep-learning
  - historical
fields:
  - vision
  - machine-learning
  - neural-networks
related:
  - "[[Backpropagation (Rumelhart et al., 1986)]]"
  - "[[Visualizing and Understanding Convolutional Networks (2014)]]"
  - "[[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]]"
predecessors:
  - "[[Backpropagation (Rumelhart et al., 1986)]]"
successors:
  - "[[AlexNet (2012)]]"
  - "[[VGGNet (2014)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
LeCun et al. (1998) introduced **LeNet-5**, a convolutional neural network (CNN) that achieved breakthrough performance in **handwritten digit recognition** (MNIST). This paper laid the foundation for modern deep learning by demonstrating the power of **gradient-based learning with CNNs**.

# Key Idea
> Use convolutional neural networks with local receptive fields, weight sharing, and subsampling (pooling) to efficiently learn visual features for document recognition.

# Method
- **LeNet-5 architecture**:  
  - Input: 32×32 grayscale images.  
  - 2 convolutional layers + subsampling (pooling).  
  - Fully connected layers + softmax.  
- **Training**: Stochastic gradient descent with backpropagation.  
- **Key innovations**:  
  - Local receptive fields (spatial hierarchies).  
  - Weight sharing (translation invariance).  
  - Subsampling → pooling for robustness.  

# Results
- Achieved state-of-the-art accuracy on **MNIST handwritten digits**.  
- Demonstrated scalability of gradient-based training for visual recognition.  
- Outperformed handcrafted features + traditional classifiers (SVM, KNN).  

# Why it Mattered
- First **successful demonstration of CNNs** at scale.  
- Proved gradient-based learning could handle **real-world vision tasks**.  
- Direct precursor to AlexNet (2012) and modern deep CNNs.  

# Architectural Pattern
- CNN (conv → pooling → fully connected → softmax).  
- Hierarchical feature extraction.  

# Connections
- Predecessor to **AlexNet (2012)**, which scaled CNNs with GPUs.  
- Related to early neural network interpretability and efficiency research.  
- Inspired virtually all modern deep learning architectures.  

# Implementation Notes
- Originally trained on digit recognition for bank checks and postal codes.  
- Required specialized hardware (AT&T Bell Labs’ DSPs).  
- The architecture was small by today’s standards (~60k parameters).  

# Critiques / Limitations
- Limited dataset size (MNIST-scale).  
- Handcrafted preprocessing still used (cropping, centering digits).  
- Couldn’t scale to large natural images due to compute/memory limits.  

---

# Educational Connections

## Undergraduate-Level Concepts
- CNN basics: convolution, pooling, fully connected layers.  
- Why weight sharing reduces parameters.  
- Example: detecting a handwritten “7” regardless of where it appears.  

## Postgraduate-Level Concepts
- Backpropagation through convolution + pooling layers.  
- Inductive biases: translation invariance vs generalization.  
- How LeNet-5 foreshadowed deep CNNs (AlexNet, ResNet).  
- Hardware limitations in the 1990s vs GPU era breakthroughs.  

---

# My Notes
- LeNet-5 = **the origin story of CNNs**.  
- Visionaries ahead of their time; ideas lay dormant until GPUs enabled AlexNet.  
- Open question: How much of CNN inductive bias (locality, weight sharing) will remain relevant in the transformer era?  
- Possible extension: Hybrid **conv-attention networks** that combine CNN efficiency with transformer flexibility.  

---
