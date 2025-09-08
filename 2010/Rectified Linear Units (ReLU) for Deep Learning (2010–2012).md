---
title: "Rectified Linear Units (ReLU) for Deep Learning (2010–2012)"
aliases:
  - ReLU
  - Rectified Linear Unit
authors:
  - Vinod Nair
  - Geoffrey Hinton
  - Alex Krizhevsky
  - Ilya Sutskever
  - Geoffrey Hinton
year: 2010 (first intro), 2012 (popularized via AlexNet)
venue: "ICML 2010 (Nair & Hinton), NeurIPS 2012 (AlexNet)"
doi: 
  - "10.5555/3104322.3104425"   # Nair & Hinton ICML 2010
  - "10.1145/3065386"           # AlexNet retrospective
arxiv: "https://arxiv.org/abs/1207.0580"   # AlexNet
citations: 100000+
tags:
  - paper
  - activation-functions
  - deep-learning
  - cnn
fields:
  - machine-learning
  - deep-learning
  - neural-networks
related:
  - "[[LeNet-5 (1998)]]"
  - "[[AlexNet (2012)]]"
  - "[[Activation Function Survey (2022)]]"
predecessors:
  - "[[Sigmoid and Tanh Activations]]"
successors:
  - "[[Leaky ReLU (2015)]]"
  - "[[ELU (2016)]]"
  - "[[GELU (2018)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
The **Rectified Linear Unit (ReLU)** activation, defined as `f(x) = max(0, x)`, became the **default nonlinearity** for deep neural networks. Introduced by **Nair & Hinton (2010)** as part of Restricted Boltzmann Machines and popularized in **AlexNet (2012)**, ReLU solved the **vanishing gradient problem** that plagued Sigmoid/Tanh and enabled deep CNNs to train effectively.

# Key Idea
> Use a **simple piecewise linear function** that passes positive values unchanged and clips negatives to zero, enabling sparse activations and stable gradient flow.

# Method
- **Function**:  
  - `f(x) = max(0, x)`  
- **Properties**:  
  - Non-saturating gradient (constant for positive x).  
  - Computationally cheap.  
  - Introduces sparsity (many zeros).  
- **Variants**: Leaky ReLU, PReLU, RReLU, ELU, GELU, Mish.  

# Results
- Nair & Hinton (2010): Showed ReLU outperforms Sigmoid/Tanh in RBMs.  
- AlexNet (2012): Used ReLU in CNNs, achieving breakthrough ImageNet performance.  
- ReLU became the **standard activation** for deep nets.  

# Why it Mattered
- Removed a key barrier to training **deep networks**.  
- Enabled the **deep learning revolution** (ImageNet breakthrough).  
- Inspired a wave of research on activation functions.  

# Architectural Pattern
- Replace Sigmoid/Tanh with ReLU in CNNs.  
- Positive gradient flow maintains learning stability.  

# Connections
- Predecessor: Sigmoid/Tanh activations in LeNet-5.  
- Successors: Leaky ReLU, ELU, GELU (transformers).  
- Directly tied to AlexNet’s success.  

# Implementation Notes
- ReLU can cause “dying neurons” if inputs are always negative.  
- Variants (Leaky ReLU, PReLU) mitigate this.  
- Default choice in most architectures (ResNet, VGG, etc.).  

# Critiques / Limitations
- Unbounded positive values → risk of exploding activations.  
- Dead neurons if gradient stuck at zero.  
- Not smooth (non-differentiable at zero).  

---

# Educational Connections

## Undergraduate-Level Concepts
- Why nonlinearities are needed in neural networks.  
- Why Sigmoid/Tanh saturate and cause vanishing gradients.  
- How ReLU keeps gradients alive.  
- Example: a digit classifier where many neurons are inactive (sparse).  

## Postgraduate-Level Concepts
- Gradient dynamics of ReLU vs Sigmoid/Tanh.  
- Dead ReLU phenomenon and remedies (Leaky ReLU, ELU).  
- Activation choice as an inductive bias (sparsity, scale).  
- Role of ReLU in enabling large-scale CNN training.  

---

# My Notes
- ReLU = **the activation that unlocked deep learning**.  
- Remarkably simple: max(0, x).  
- Open question: Will smooth variants (GELU, Mish) replace ReLU in the long run, or will simplicity win out?  
- Possible extension: Learned adaptive activations that evolve per task.  

---
