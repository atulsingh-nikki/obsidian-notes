---
title: Fully Convolutional Networks for Semantic Segmentation (2015)
aliases:
  - FCN
  - FCN-8s
authors:
  - Jonathan Long
  - Evan Shelhamer
  - Trevor Darrell
year: 2015
venue: CVPR
doi: 10.1109/CVPR.2015.7298965
arxiv: https://arxiv.org/abs/1411.4038
code: https://github.com/shelhamer/fcn.berkeleyvision.org
citations: 30,000+
dataset:
  - PASCAL VOC
  - NYUDv2
  - SIFT Flow
tags:
  - paper
  - semantic-segmentation
  - deep-learning
  - cnn
fields:
  - vision
  - segmentation
related:
  - "[[SegNet]]"
  - "[[U-Net]]"
predecessors:
  - "[[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]]"
  - "[[OverFeat Integrated Recognition, Localization and Detection using Convolutional Networks]]"
successors:
  - "[[SegNet]]"
  - "[[DeepLab]]"
  - "[[U-Net Convolutional Networks for Biomedical Image Segmentation (2015)|UNet]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
This work introduced the **first end-to-end trainable convolutional network** for **pixel-wise semantic segmentation**. It repurposed classification CNNs ([[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]], VGG, GoogLeNet) into fully convolutional networks, enabling dense per-pixel predictions without fully connected layers.

# Key Idea
> Replace fully connected layers with convolutions, enabling dense per-pixel classification and end-to-end segmentation.

# Method
- Converted ImageNet-trained CNNs into **fully convolutional networks (FCNs)**.  
- Added **deconvolution (transpose convolution)** layers to upsample coarse outputs to input resolution.  
- Introduced **skip connections (FCN-16s, FCN-8s)** for finer spatial details.  
- Trained end-to-end on segmentation datasets with pixel-wise labels.  

# Results
- Achieved state-of-the-art performance on **PASCAL VOC 2011/2012**, **NYUDv2**, and **SIFT Flow**.  
- Demonstrated the viability of end-to-end deep learning for segmentation.  
- Outperformed classical methods like CRFs and region-based classifiers.  

# Why it Mattered
- First successful application of CNNs to **dense prediction tasks**.  
- Established a paradigm shift: **classification → dense prediction**.  
- Paved the way for architectures like **U-Net, DeepLab, PSPNet, Mask R-CNN**.  

# Architectural Pattern
- **Encoder–decoder** style, with upsampling via transposed convolutions.  
- Skip connections to combine deep semantic and shallow spatial features.  
- Set the foundation for many segmentation models that followed.  

# Connections
- **Contemporaries**: SegNet (2015), DeconvNet (2015).  
- **Influence**: Every modern segmentation architecture (DeepLab, U-Net, [[Mask R-CNN]]).  

# Implementation Notes
- Trained from pre-trained ImageNet classification models.  
- Need careful initialization for deconvolution filters.  
- Skip connections critical for fine boundary accuracy.  

# Critiques / Limitations
- Coarse boundaries compared to later CRF+CNN hybrids (e.g., DeepLab with CRF).  
- Computationally expensive at full resolution.  
- Limited robustness to small objects.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1411.4038)  
- [Original Project Page](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)  
- [Code (Caffe)](https://github.com/shelhamer/fcn.berkeleyvision.org)  
- [PyTorch reimplementation](https://github.com/wkentaro/pytorch-fcn)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Convolutions, transpose convolutions.  
- **Probability & Statistics**: Softmax for pixel-level classification.  
- **Optimization Basics**: Gradient descent training of deep nets.  
- **Signals & Systems**: Convolutional filtering and receptive fields.  

## Postgraduate-Level Concepts
- **Numerical Methods**: Efficient upsampling and interpolation strategies.  
- **Neural Network Design**: Encoder–decoder, skip connections.  
- **Computer Vision**: Semantic segmentation benchmarks and tasks.  
- **Research Methodology**: Transfer learning from classification to dense prediction.  

---

# My Notes
- Critical milestone in moving from **classification to structured prediction** in vision.  
- Links directly to my interest in **mask generation and object selection** for video editing.  
- Open question: Can diffusion models provide **per-pixel generation with uncertainty estimates** beyond FCNs?  
- Extension: Integrate with **temporal consistency losses** for video segmentation.  
