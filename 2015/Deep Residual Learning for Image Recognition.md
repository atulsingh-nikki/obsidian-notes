---
title: ResNet (Deep Residual Learning for Image Recognition)
aliases:
  - ResNet (2015)
authors:
  - Kaiming He
  - Xiangyu Zhang
  - Shaoqing Ren
  - Jian Sun
year: 2015
venue: CVPR 2016 (arXiv 2015, ILSVRC 2015 Winner)
dataset:
  - ImageNet (ILSVRC 2015)
  - CIFAR-10
tags:
  - computer-vision
  - cnn
  - residual-learning
  - deep-learning
  - image-classification
  - skip-connections
  - architecture
arxiv: https://arxiv.org/abs/1512.03385
related:
  - "[[GoogLeNet (2014)]]"
  - "[[DenseNet (2016)]]"
  - "[[Transformers in Vision]]"
  - "[[VGGNet (2014)]]"
  - "[[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]]"
---

# Summary
ResNet introduced **residual connections** that allowed training of extremely deep networks (up to 152 layers) without the degradation problem. By learning **residual functions** instead of direct mappings, the network optimized more easily, achieving breakthrough accuracy on ImageNet and winning ILSVRC 2015. ResNets became the **standard backbone** for computer vision tasks.

# Key Idea (one-liner)
> Learn residual functions \(F(x) = H(x) - x\), so the network instead learns \(y = F(x) + x\) via **skip connections**, making very deep networks trainable.

# Method
- **Residual Block**:
  - Input \(x\) → Conv → BN → ReLU → Conv → BN → Add(skip x) → ReLU.
- **Skip connections**: identity shortcuts bypassing 2–3 layers, enabling gradient flow.
- **Variants**: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152.
- **Bottleneck design**: 1×1 conv (reduce channels) → 3×3 conv → 1×1 conv (restore channels).
- **Optimization**: SGD with momentum, batch normalization, weight initialization.
- **Training depth**: successfully trained 152-layer networks on ImageNet.

# Results
- **ILSVRC 2015 Winner**: 3.57% top-5 error on ImageNet.
- Outperformed human-level accuracy on some classification benchmarks.
- Robust transferability: ResNet features became standard for detection, segmentation, recognition.

# Why it Mattered
- Solved the **degradation problem**: accuracy got worse with more layers (pre-ResNet).
- Proved extremely deep networks can be trained reliably.
- Residual learning became a **fundamental building block** in vision, NLP, and beyond (Transformers use skip connections everywhere).
- Established ResNet as a “default backbone” for CV research.

# Architectural Pattern
- [[Residual Block]] → skip/identity connection.
- [[Bottleneck Block]] → 1×1–3×3–1×1 design.
- [[Batch Normalization]] → stabilizes training.
- [[Deep Supervision via Skip Connections]] → better gradient flow.

# Connections
- **Predecessors**:
  - [[Very Deep Convolutional Networks for Large-Scale Image Recognition|VGGNet (2014)]] → deep but difficult to optimize.
  - [[GoogLeNet (2014)]] → modular multi-branch design.
- **Successors**:
  - [[ResNeXt (2017)]] → group convolutions.
  - [[DenseNet (2016)]] → dense skip connections.
  - [[Transformers in Vision (ViT, 2020)]] → residual connections form core design.
- **Influence**:
  - Every modern deep architecture (CNNs, Transformers, GANs) uses residual learning.

# Implementation Notes
- Widely available pretrained weights (18–152 layers).
- Still used as baselines and backbones in detection/segmentation pipelines.
- Efficient when paired with bottleneck design.

# Critiques / Limitations
- Increased depth = increased compute/memory.
- Performance saturates after certain depth (~1000 layers).
- Lacks efficiency compared to modern lightweight networks (MobileNet, EfficientNet).

# Repro / Resources
- Paper: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Dataset: [[ImageNet]]
- Official code: Microsoft Research (Caffe, later PyTorch/TensorFlow ports).
- Pretrained models: ResNet-18/34/50/101/152.

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Matrix multiplication in convolution layers.
  - Identity mapping (x passed directly via skip).
  
- **Probability & Statistics**
  - Softmax classifier for final layer.
  - Overfitting → regularization via deep capacity control.
  
- **Calculus**
  - Chain rule in backpropagation.
  - Vanishing gradients mitigated by identity shortcut.
  
- **Signals & Systems**
  - Identity connection as signal bypass → better gradient signal propagation.
  - Convolutional filters as feature extractors.

- **Data Structures**
  - Tensors for feature maps.
  - Block structures (residual modules).

- **Optimization Basics**
  - SGD with momentum.
  - Batch normalization.
  - Skip connections to stabilize optimization.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Degradation vs vanishing gradient problems.
  - Residual mappings as reparameterization trick.
  - Deeper networks enabled by improved gradient flow.

- **Numerical Methods**
  - Efficient implementation of skip connections.
  - Bottleneck design reduces FLOPs.
  - Depth scaling vs compute scaling.

- **Machine Learning Theory**
  - Residual learning as functional approximation.
  - Hypothesis space expansion with identity shortcuts.
  - Generalization in deeper architectures.

- **Computer Vision**
  - State-of-the-art classification on ImageNet.
  - Transferability to detection, segmentation, recognition.
  - ResNet as standard backbone.

- **Neural Network Design**
  - Residual block → template reused across tasks.
  - Bottleneck vs plain blocks.
  - Scalable family (18–152 layers).

- **Transfer Learning**
  - Pretrained ResNets widely used in downstream tasks.
  - Fine-tuning strategies vary by depth.
  - Features generalize across domains.

- **Research Methodology**
  - Systematic study of depth (18 vs 34 vs 50+).
  - Ablation on residual vs non-residual nets.
  - Benchmarks on ImageNet, CIFAR.
