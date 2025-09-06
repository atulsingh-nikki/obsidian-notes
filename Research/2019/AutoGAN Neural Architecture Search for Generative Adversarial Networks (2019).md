---
title: "AutoGAN: Neural Architecture Search for Generative Adversarial Networks (2019)"
aliases: 
  - AutoGAN
  - NAS for GANs
authors:
  - Rui Gao
  - Yinpeng Chen
  - Mingyang Li
  - Zicheng Liu
  - Jingjing Liu
  - Jingdong Wang
  - Hanqing Lu
year: 2019
venue: "ICCV"
doi: "10.1109/ICCV.2019.00690"
arxiv: "https://arxiv.org/abs/1908.03835"
code: "https://github.com/TAMU-VITA/AutoGAN"
citations: 700+
dataset:
  - CIFAR-10
  - STL-10
tags:
  - paper
  - gan
  - architecture-search
  - nas
fields:
  - vision
  - generative-models
  - architecture
related:
  - "[[StyleGAN (2019)]]"
  - "[[BigGAN (2019)]]"
predecessors:
  - "[[Neural Architecture Search (NASNet, 2017)]]"
successors:
  - "[[Diffusion NAS models]]"
impact: ⭐⭐⭐☆
status: "read"
---

# Summary
AutoGAN applied **Neural Architecture Search (NAS)** to the design of **GAN architectures**, automatically discovering both **generator and discriminator** structures. Unlike hand-crafted designs (DCGAN, ResNet, StyleGAN), AutoGAN learned architectures tailored for stability and performance on image synthesis benchmarks.

# Key Idea
> Use **differentiable architecture search** to jointly optimize GAN architectures for both generator and discriminator, improving image quality and stability.

# Method
- **Search space**:  
  - Includes operations such as convolutions of different kernel sizes, upsampling, residual connections.  
  - Separate search spaces for generator and discriminator.  
- **Search algorithm**:  
  - Based on **differentiable NAS** (continuous relaxation of architecture choices).  
  - Trained adversarially to evaluate candidate architectures.  
- **Optimization**:  
  - Alternates between architecture parameter updates and network weight updates.  

# Results
- On **CIFAR-10**: Outperformed hand-designed GANs in FID and Inception Score.  
- On **STL-10**: Achieved competitive performance with far fewer parameters.  
- Demonstrated stable training dynamics from discovered architectures.  

# Why it Mattered
- First work to show NAS can discover **better GAN architectures** than human intuition.  
- Reduced need for costly trial-and-error GAN design.  
- Inspired further work in **autoML for generative modeling**.  

# Architectural Pattern
- Generator and discriminator searched independently.  
- Differentiable NAS used for efficient search.  
- Final architectures trained from scratch for evaluation.  

# Connections
- **Contemporaries**: StyleGAN, BigGAN.  
- **Influence**: AutoML applied to generative models, diffusion architecture search.  

# Implementation Notes
- Search computationally expensive (requires multiple GPUs).  
- Performance depends heavily on search space design.  
- Once discovered, architectures are efficient to train.  

# Critiques / Limitations
- Search space still hand-crafted; NAS did not fully escape manual bias.  
- Results strong on CIFAR/STL but not scaled to large datasets like ImageNet.  
- Later surpassed by StyleGAN and diffusion models in quality.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1908.03835)  
- [Official code (PyTorch)](https://github.com/TAMU-VITA/AutoGAN)  
- [Project page (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/autogan-neural-architecture-search-for-generative-adversarial-networks/)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Operations in convolutional architectures.  
- **Optimization**: Gradient descent for architecture parameters.  
- **Probability**: GAN loss as minimax optimization.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Automated architecture search.  
- **Generative Models**: Stability improvements in adversarial training.  
- **Research Methodology**: Designing and constraining search spaces.  
- **Advanced Optimization**: Differentiable architecture search with adversarial objectives.  

---

# My Notes
- AutoGAN feels like a **proof of concept**: NAS can indeed discover novel, stable GANs.  
- But diffusion-based models today rely less on NAS, more on **scaling laws and transformers**.  
- Open question: Would **NAS + diffusion backbones** discover even more efficient architectures?  
- Possible extension: Use NAS to optimize **video diffusion architectures** for editing tasks.  

---
