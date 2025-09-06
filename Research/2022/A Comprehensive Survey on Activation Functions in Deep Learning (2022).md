---
title: A Comprehensive Survey on Activation Functions in Deep Learning (2022)
aliases:
  - Activation Function Survey 2022
authors:
  - Jagtap, A.D.
  - Ravi, V.
  - et al.
year: 2022
venue: Neural Computing and Applications
doi: 10.1007/s00521-022-07630-y
arxiv: https://arxiv.org/abs/2202.00239
citations: 300+
tags:
  - survey
  - deep-learning
  - activation-functions
fields:
  - deep-learning
  - neural-networks
  - optimization
related:
  - "[[Backpropagation (Rumelhart et al., 1986)]]"
  - "[[Rectified Linear Unit (ReLU)]]"
  - "[[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]]"
  - "[[Gradient-Based Learning Applied to Document Recognition (1998)|LeNet-5]]"
impact: ⭐⭐⭐⭐☆
status: reading
---

# Summary
This survey provides a **comprehensive review of activation functions** used in deep learning, analyzing their properties, advantages, disadvantages, and applications. It covers **classical functions (Sigmoid, Tanh)**, **modern variants (ReLU, Leaky ReLU, ELU, GELU, Swish, Mish)**, and **domain-specific activations**.

# Key Idea
> Activation functions play a central role in learning dynamics by introducing **nonlinearity**, affecting convergence speed, gradient flow, and generalization. The survey categorizes them historically and experimentally.

# Content Overview
- **Classical activations**: Sigmoid, Tanh → smooth, but suffer from vanishing gradients.  
- **ReLU family**: ReLU, Leaky ReLU, PReLU, RReLU → efficient, sparse, but prone to “dying ReLU.”  
- **Advanced nonlinearities**: ELU, SELU, GELU, Swish, Mish → smoother gradients, better expressivity.  
- **Adaptive functions**: Learnable activations (APL, PReLU variants).  
- **Domain-specific**: Activations tuned for CNNs, RNNs, GANs, Transformers.  
- **Theoretical analysis**: Gradient flow, Lipschitz properties, and generalization links.  

# Results / Findings
- No single activation dominates across all tasks.  
- ReLU and its variants remain the **default choice** for efficiency.  
- GELU/Swish/Mish provide smoother convergence in transformers and large-scale models.  
- Domain-specific tuning often outperforms generic choices.  

# Why it Mattered
- First **comprehensive taxonomy** of activations in deep learning.  
- Helps practitioners select activations beyond defaults.  
- Highlights the **open research question**: designing principled activations vs empirical discovery.  

# Connections
- Builds on the role of **ReLU in AlexNet (2012)**.  
- Links to **SELU** (self-normalizing nets, 2017) and **GELU** (Transformers, 2018).  
- Survey complement to optimizer surveys (Adam, SGD, etc.).  

# Critiques / Limitations
- Mostly descriptive; limited theoretical depth on why activations work.  
- Rapidly evolving — newer variants may not be covered.  
- Lacks unifying principles for activation design.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What activation functions do (nonlinearity in neural nets).  
- Why Sigmoid/Tanh cause vanishing gradients.  
- Why ReLU is simple and effective.  
- Example: without nonlinearities, a neural net = linear regression.  

## Postgraduate-Level Concepts
- Activation design trade-offs: smoothness vs sparsity.  
- Gradient properties and Lipschitz continuity.  
- Role of activations in generalization vs expressivity.  
- Open problem: deriving activations from optimization theory.  

---

# My Notes
- Survey confirms: **ReLU is still king**, but GELU/Swish/Mish dominate in transformers.  
- Interesting that activations are **task- and architecture-dependent**, unlike universal optimizers.  
- Open question: Can we **automatically learn activations** (NAS-style) for each model?  
- Possible extension: meta-learning activations per layer for foundation models.  

---
