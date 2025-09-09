---
title: "Empirical Evaluation of Rectified Activations in Convolutional Networks (2015)"
aliases:
  - PReLU
  - Parametric ReLU
  - Xu et al. 2015
authors:
  - Bing Xu
  - Naiyan Wang
  - Tianqi Chen
  - Mu Li
year: 2015
venue: "arXiv preprint"
arxiv: "https://arxiv.org/abs/1505.00853"
citations: 8000+
tags:
  - paper
  - activation-functions
  - cnn
fields:
  - machine-learning
  - deep-learning
  - neural-networks
related:
  - "[[ReLU (2010–2012)]]"
  - "[[Leaky ReLU (2015)]]"
  - "[[ELU (2016)]]"
  - "[[GELU (2018)]]"
predecessors:
  - "[[ReLU (2010–2012)]]"
successors:
  - "[[ELU (2016)]]"
  - "[[SELU (2017)]]"
  - "[[GELU (2018)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
Xu et al. (2015) conducted a systematic **empirical evaluation of rectified activation functions** and introduced the **Parametric Rectified Linear Unit (PReLU)**. By allowing the negative slope of the activation to be **learned**, PReLU improved accuracy in very deep CNNs, including ImageNet classification.

# Key Idea
> Generalize ReLU by introducing a **learnable parameter** for the slope of negative values, preventing the “dying ReLU” problem and adapting activation functions to the data.

# Method
- **ReLU**: `f(x) = max(0, x)`  
- **Leaky ReLU**: `f(x) = x if x > 0 else αx` (fixed α)  
- **PReLU**: same as Leaky ReLU, but **α is learnable** per channel.  
- **Empirical study**: compared ReLU, Leaky ReLU, and PReLU across deep CNNs.  

# Results
- PReLU consistently improved accuracy over ReLU and Leaky ReLU.  
- Helped train **very deep CNNs** (up to 30+ layers) on ImageNet.  
- Reduced risk of dead neurons compared to ReLU.  

# Why it Mattered
- Showed that even small changes in activation design impact training dynamics.  
- Popularized **learnable activation functions**.  
- Predecessor to adaptive activations like Swish, GELU, Mish.  

# Architectural Pattern
- CNN backbone (ResNet-style experiments).  
- Replace ReLU with PReLU.  

# Connections
- Direct successor to **ReLU** and **Leaky ReLU**.  
- Predecessor to smoother activations like **ELU (2016)** and **GELU (2018)**.  
- Related to adaptive activation research.  

# Implementation Notes
- Simple to implement: learnable α initialized at small value (e.g., 0.25).  
- Negligible increase in parameters (1 per channel).  
- Efficient in practice, no major compute overhead.  

# Critiques / Limitations
- Gains modest compared to architectural innovations (e.g., ResNet).  
- Risk of overfitting if α is too flexible.  
- Later superseded by smoother nonlinearities (ELU, GELU).  

---

# Educational Connections

## Undergraduate-Level Concepts
- ReLU vs Leaky ReLU vs PReLU.  
- Why allowing a small negative slope prevents “dead neurons.”  
- Example: a neuron always seeing negative inputs would be inactive in ReLU but still contribute in PReLU.  

## Postgraduate-Level Concepts
- Learning activation parameters jointly with weights.  
- Impact on gradient flow in very deep networks.  
- Comparison of PReLU vs Swish/GELU (smoothness vs piecewise linear).  
- Broader theme: **learned inductive biases** in deep learning.  

---

# My Notes
- PReLU = **small tweak, big impact** in 2015.  
- Was crucial in early deep CNNs before ResNet dominance.  
- Open question: Is there a principled way to derive optimal activations, or will they remain empirical hacks?  
- Possible extension: meta-learned per-layer activations in foundation models.  

---
