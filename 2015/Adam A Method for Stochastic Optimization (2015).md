---
title: "Adam: A Method for Stochastic Optimization (2015)"
aliases: 
  - Adam Optimizer
  - Adaptive Moment Estimation
authors:
  - Diederik P. Kingma
  - Jimmy Ba
year: 2015
venue: "ICLR"
doi: "10.48550/arXiv.1412.6980"
arxiv: "https://arxiv.org/abs/1412.6980"
code: "https://pytorch.org/docs/stable/generated/torch.optim.Adam.html"
citations: 150,000+
dataset:
  - Used across diverse ML benchmarks
tags:
  - paper
  - optimization
  - deep-learning
fields:
  - optimization
  - machine-learning
  - deep-learning
related:
  - "[[SGD with Momentum]]"
  - "[[RMSProp]]"
predecessors:
  - "[[AdaGrad]]"
  - "[[RMSProp]]"
successors:
  - "[[AMSGrad]]"
  - "[[Lion Optimizer]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
Adam (Adaptive Moment Estimation) is a **stochastic gradient descent optimization method** that computes **adaptive learning rates** for each parameter. It combines ideas from **momentum (1st moment)** and **RMSProp (2nd moment)** into a single, efficient algorithm.

# Key Idea
> Maintain exponentially decaying averages of past gradients and squared gradients, bias-correct them, and use them to adaptively scale updates.

# Method
- Maintains two running estimates for each parameter:  
  - **First moment (mₜ)**: mean of gradients (momentum).  
  - **Second moment (vₜ)**: uncentered variance of gradients.  
- Bias correction applied to counter initialization effects.  
- Update rule:  

  \[
  m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
  \]  
  \[
  v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
  \]  
  \[
  \hat{m_t} = m_t / (1-\beta_1^t), \quad \hat{v_t} = v_t / (1-\beta_2^t)
  \]  
  \[
  \theta_{t+1} = \theta_t - \alpha \cdot \hat{m_t} / (\sqrt{\hat{v_t}} + \epsilon)
  \]

- Default hyperparameters: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸.  

# Results
- Outperformed vanilla SGD and AdaGrad/RMSProp on benchmarks.  
- Became the **default optimizer** for deep learning frameworks.  
- Demonstrated robustness across models (CNNs, RNNs, VAEs).  

# Why it Mattered
- Extremely **easy to use**, requiring little hyperparameter tuning.  
- Robust across architectures and datasets.  
- Quickly became the **de facto standard optimizer** in deep learning research and practice.  

# Architectural Pattern
- Hybrid: combines **momentum** (first-order) and **adaptive scaling** (second-order).  
- Generalizable update rule used in almost all ML frameworks.  

# Connections
- **Contemporaries**: AdaDelta, RMSProp.  
- **Influence**: AMSGrad (fix for theoretical convergence), AdaBound, Rectified Adam (RAdam), Lion Optimizer.  

# Implementation Notes
- Often requires learning rate warm-up in large-scale training.  
- Can lead to poorer generalization compared to SGD with momentum in some cases (e.g., image classification).  
- Still widely used in practice due to stability.  

# Critiques / Limitations
- Convergence guarantees weaker than SGD with momentum.  
- Sometimes leads to worse **generalization** despite faster optimization.  
- Needs tuning of learning rate decay schedules for best results.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1412.6980)  
- [PyTorch Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)  
- [TensorFlow Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Vector updates in optimization.  
- **Probability & Statistics**: Variance estimation (2nd moment).  
- **Calculus**: Gradient-based optimization.  
- **Optimization Basics**: SGD, momentum.  

## Postgraduate-Level Concepts
- **Numerical Methods**: Bias correction, exponential moving averages.  
- **Machine Learning Theory**: Convergence analysis of adaptive methods.  
- **Research Methodology**: Benchmarking across architectures.  
- **Advanced Optimization**: Comparison with SGD generalization gaps.  

---

# My Notes
- Adam = my go-to for **training diffusion models and transformers**.  
- Open question: Can **newer optimizers (Lion, Sophia, Adafactor)** replace Adam in large-scale training?  
- Possible extension: Explore Adam variants specifically optimized for **video model training**.  
