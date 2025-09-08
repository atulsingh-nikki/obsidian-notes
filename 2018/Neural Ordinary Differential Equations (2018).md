---
title: "Neural Ordinary Differential Equations (2018)"
aliases: 
  - Neural ODEs
  - Continuous-depth Networks
authors:
  - Ricky T. Q. Chen
  - Yulia Rubanova
  - Jesse Bettencourt
  - David Duvenaud
year: 2018
venue: "NeurIPS"
doi: "10.48550/arXiv.1806.07366"
arxiv: "https://arxiv.org/abs/1806.07366"
code: "https://github.com/rtqichen/torchdiffeq"
citations: 10,000+
dataset:
  - CIFAR-10
  - MNIST
  - Continuous-time dynamical systems benchmarks
tags:
  - paper
  - deep-learning
  - odes
  - continuous-models
fields:
  - machine-learning
  - optimization
  - time-series
related:
  - "[[Residual Networks (ResNet, 2015)]]"
  - "[[Continuous Normalizing Flows]]"
predecessors:
  - "[[ResNet (2015)]]"
successors:
  - "[[Augmented Neural ODEs]]"
  - "[[Neural SDEs]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
Neural ODEs reframed deep neural networks as **continuous-depth models**, where hidden states evolve according to an **ordinary differential equation (ODE)** parameterized by a neural network. Instead of stacking discrete layers, the model integrates a learned ODE over time.

# Key Idea
> Replace discrete layer updates with a **continuous ODE solver**, enabling adaptive computation and memory-efficient training.

# Method
- Defines hidden state evolution:  
$$
  \frac{dz(t)}{dt} = f(z(t), t; \theta)
  $$  
- **Forward pass**: Use an ODE solver (e.g., Runge-Kutta) to compute outputs at desired times.  
- **Backward pass**: Use **adjoint sensitivity method** to compute gradients with constant memory.  
- Applications: classification, density modeling, continuous-time sequence modeling.  

# Results
- Achieved competitive performance on **image classification benchmarks** with fewer parameters.  
- Continuous-time models excelled in **irregularly-sampled time series**.  
- Enabled **continuous normalizing flows (CNFs)** for generative modeling.  

# Why it Mattered
- Introduced the idea of **continuous-depth networks**, generalizing ResNets.  
- Opened new directions in combining deep learning with dynamical systems.  
- Spawned variants: Neural SDEs, Augmented ODEs, ODE-RNNs.  

# Architectural Pattern
- ODE-defined hidden dynamics.  
- Adaptive solvers for efficiency vs accuracy trade-off.  
- Memory-efficient adjoint gradients.  

# Connections
- **Contemporaries**: Deep generative models (2018 surge).  
- **Influence**: CNFs, time-series models, physics-informed ML.  

# Implementation Notes
- Solver choice impacts speed and accuracy.  
- Adjoint method allows O(1) memory cost but can be numerically unstable.  
- Training slower than discrete nets for vision, but better for time series.  

# Critiques / Limitations
- ODE solvers can be slow compared to fixed-depth CNNs.  
- Sensitive to solver tolerance hyperparameters.  
- Struggles with chaotic dynamics (later addressed by Augmented Neural ODEs).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1806.07366)  
- [Official PyTorch implementation](https://github.com/rtqichen/torchdiffeq)  
- [Tutorial on Neural ODEs](https://diffeqflux.sciml.ai/)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Derivatives and ODEs.  
- **Numerical Methods**: Euler, Runge-Kutta integration.  
- **Optimization Basics**: Gradient descent with chain rule through ODEs.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Continuous-depth architectures.  
- **Machine Learning Theory**: Connection to dynamical systems.  
- **Research Methodology**: Benchmarking ODE-based vs discrete models.  
- **Advanced Optimization**: Adjoint sensitivity, CNFs.  

---

# My Notes
- Very relevant for **video frame interpolation and continuous motion models**.  
- Open question: Can diffusion models be seen as **Neural SDEs** generalizing Neural ODEs?  
- Possible extension: Apply ODE-based models for **temporal consistency in video editing**.  

---
