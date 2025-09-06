---
title: Gaussian Error Linear Units (GELU)
aliases:
  - GELU (2016)
authors:
  - Dan Hendrycks
  - Kevin Gimpel
year: 2016
venue: arXiv 2016 (ICLR Workshop 2016)
tags:
  - activation-functions
  - deep-learning
  - neural-networks
  - transformers
  - nonlinearity
arxiv: https://arxiv.org/abs/1606.08415
related:
  - "[[ReLU Activation]]"
  - "[[ELU (2015)]]"
  - "[[Swish (2017)]]"
  - "[[Transformers (2017)]]"
---

# Summary
The **Gaussian Error Linear Unit (GELU)** is a smooth, differentiable activation function that blends properties of ReLU and sigmoid-based activations. Instead of gating inputs by a hard threshold (as in ReLU), GELU weights them by their value’s position on a Gaussian distribution. GELU has become the default activation in many Transformer-based architectures, including **BERT and GPT**.

# Key Idea (one-liner)
> Multiply inputs by the probability that a standard normal variable is less than the input — a smooth alternative to ReLU.

# Formula
\[
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
\]

- Where \(\Phi(x)\) is the CDF of a standard normal distribution.
- Approximation for efficiency:
\[
\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)
\]

# Results
- Outperforms ReLU and ELU on several benchmarks (MNIST, CIFAR, TIMIT).  
- Becomes especially effective in **large models and NLP tasks**.  
- Adopted as the default nonlinearity in Transformers (BERT, GPT, Vision Transformers).  

# Why it Mattered
- Smooth and differentiable → better gradient flow.  
- Combines benefits of ReLU (sparsity, robustness) and sigmoid (probabilistic weighting).  
- Mathematically grounded in probability (CDF of Gaussian).  
- Became a **core building block of modern LLMs**.  

# Architectural Pattern
- [[Activation Functions]] → improves nonlinearity choice.  
- [[Transformers (2017)]] → standard nonlinearity in feed-forward layers.  
- [[ReLU vs GELU]] → smooth vs hard gating.  

# Connections
- **Predecessors**: ReLU (2011), ELU (2015), Maxout (2013).  
- **Successors**: [[Swish (2017)]], [[Mish (2019)]] (other smooth nonlinearities).  
- **Influence**: Default in BERT, GPT, ViT, and most state-of-the-art deep models.  

# Implementation Notes
- Exact formula uses error function (erf), but tanh approximation is faster.  
- Supported in most ML frameworks (`torch.nn.GELU`, `tf.nn.gelu`).  
- Drop-in replacement for ReLU.  

# Critiques / Limitations
- Slightly more expensive than ReLU (needs erf or tanh).  
- The probabilistic interpretation is heuristic, not strictly necessary.  
- Gains over ReLU are dataset/model-dependent (largest in NLP).  

# Repro / Resources
- Paper: [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)  
- Implementations: PyTorch, TensorFlow, JAX built-ins.  
- Used in: BERT, GPT series, Vision Transformer (ViT).  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Elementwise nonlinear function applied to vectors.  
- **Probability & Statistics**
  - Uses Gaussian CDF (\(\Phi(x)\)).  
  - Probabilistic gating of activations.  
- **Calculus**
  - Differentiable everywhere (unlike ReLU at 0).  
  - Gradient involves Gaussian PDF.  
- **Signals & Systems**
  - Smooth gating vs hard clipping (ReLU).  

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Smooth gradient improves training stability.  
  - Less brittle than ReLU in very deep networks.  
- **Numerical Methods**
  - Approximations of erf for faster computation.  
- **Machine Learning Theory**
  - Expected value interpretation: output = input * probability(random Gaussian ≤ x).  
  - Smooth nonlinearity helps avoid “dead neurons.”  
- **Neural Network Design**
  - Transformer FFN layers: Linear → GELU → Linear.  
- **Research Methodology**
  - Empirical comparisons vs ReLU/ELU across tasks.  
  - Adoption in benchmarks demonstrated scalability.  
