---
title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
aliases:
  - EfficientNet (2019)
  - Compound Scaling CNN
authors:
  - Mingxing Tan
  - Quoc V. Le
year: 2019
venue: ICML 2019
doi: 10.48550/arXiv.1905.11946
arxiv: https://arxiv.org/abs/1905.11946
code: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
citations: 30000+
dataset:
  - ImageNet
  - CIFAR-100
  - Flowers-102
tags:
  - paper
  - deep-learning
  - computer-vision
  - efficient-models
  - cnn
fields:
  - vision
  - model-scaling
  - architecture-design
related:
  - "[[MobileNet (2017)]]"
  - "[[EfficientNetV2 (2021)]]"
  - "[[Deep Residual Learning for Image Recognition|ResNet (2015)]]"
  - "[[GoogLeNet / Inception (2014)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
EfficientNet introduced a **compound scaling method** that balances depth, width, and input resolution to build more efficient convolutional networks. It achieves state-of-the-art accuracy on ImageNet while using an order of magnitude fewer parameters and FLOPs than previous models.

# Key Idea
> Scale network depth, width, and resolution in a principled, compound way instead of arbitrary scaling in one dimension.

# Method
- **Baseline Architecture**: Developed a small EfficientNet-B0 using neural architecture search (NAS).
- **Compound Scaling**: Introduced coefficients to scale width, depth, and resolution together.
- **Family of Models**: EfficientNet-B0 → B7 created by scaling up with compound coefficients.
- **Activation**: Used [[Swish Self-Gated Activation Function]] instead of ReLU for smoother gradients.
- **Regularization**: Data augmentation, dropout, and stochastic depth.

# Results
- ImageNet top-1 accuracy: 84.4% (EfficientNet-B7) with far fewer FLOPs than ResNet/ResNeXt.
- Transfer learning: strong results across CIFAR-100, Flowers-102, and other benchmarks.
- Pareto frontier shift: higher accuracy with fewer parameters and computation.

# Why it Mattered
- Established a new scaling principle that influenced subsequent models.
- Became a foundation for mobile/edge deployment due to efficiency.
- Popularized Swish activation in large CNNs.
- Inspired EfficientNetV2 and influenced ViT design for parameter-efficiency.

# Architectural Pattern
- [[Convolutional Neural Networks]] → MBConv blocks (depthwise separable convolutions).
- [[Neural Architecture Search (NAS)]] → baseline design.
- [[Compound Scaling]] → systematic model scaling.
- [[Swish Activation]] → improved nonlinearity.

# Connections
- **Predecessors**: [[MobileNet (2017)]], [[ResNet (2015)]], [[Inception (2014)]].  
- **Contemporaries**: AmoebaNet, NASNet (NAS-driven models).  
- **Successors**: [[EfficientNetV2 (2021)]], MobileViT, ConvNeXt.  
- **Influence**: Deployment-friendly models for mobile/edge AI, widespread adoption in transfer learning.

# Implementation Notes
- Training: RMSProp with momentum, learning rate warmup and decay.
- Regularization: dropout, stochastic depth, AutoAugment.
- Pretrained models widely available in TensorFlow, PyTorch (`torchvision.models`).
- Scaling formula is simple and tunable.

# Critiques / Limitations
- NAS-generated baseline (EfficientNet-B0) is less interpretable.
- Scaling law tied to CNNs — less relevant with Transformers.
- Large variants (B7) still computationally heavy for edge devices.
- Later work (EfficientNetV2) improved training speed and efficiency.

# Repro / Resources
- Paper: [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Code: [TensorFlow TPU models](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- PyTorch: `torchvision.models.efficientnet_b0` through `b7`
- Dataset: [[ImageNet]]

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: matrix ops in convolutions, scaling by multipliers.  
- **Probability & Statistics**: dropout, stochastic depth regularization.  
- **Calculus**: backpropagation through depthwise convolutions.  
- **Signals & Systems**: depthwise separable convolutions as efficient filters.  
- **Data Structures**: feature maps, tensor scaling.  
- **Optimization Basics**: RMSProp optimizer, warmup schedules.  

## Postgraduate-Level Concepts
- **Advanced Optimization**: compound scaling vs single-dimension scaling.  
- **Numerical Methods**: FLOPs vs parameter trade-off analysis.  
- **Machine Learning Theory**: scaling laws for CNNs.  
- **Computer Vision**: state-of-the-art ImageNet performance with efficiency.  
- **Neural Network Design**: MBConv blocks, Swish activations.  
- **Transfer Learning**: EfficientNet pretrained models widely used.  
- **Research Methodology**: ablations on scaling strategies, Pareto frontier analysis.  

---

# My Notes
- Strong candidate for mobile deployment in vision tasks.  
- Useful baseline when balancing accuracy vs compute.  
- Compound scaling idea could inspire similar approaches in Transformers.  
