---
title: "Swish: Self-Gated Activation Function (2017)"
aliases: 
  - Swish
  - SiLU
authors:
  - Prajit Ramachandran
  - Barret Zoph
  - Quoc V. Le
year: 2017
venue: "NeurIPS 2018 (arXiv 2017)"
doi: ""
arxiv: "https://arxiv.org/abs/1710.05941"
code: ""
citations: 0
dataset:
  - ImageNet
tags:
  - activation-functions
  - deep-learning
  - nonlinearity
  - neural-networks
  - transformers
fields:
  - vision
  - nlp
related:
  - "[[ReLU Activation (2011)]]"
  - "[[Maxout Networks (2013)]]"
  - "[[ELU (2015)]]"
  - "[[GELU (2016)]]"
  - "[[Mish (2019)]]"
  - "[[EfficientNet (2019)]]"
impact: ⭐⭐⭐⭐☆  
status: "read"
---

# Summary
Swish is a smooth, non-monotonic activation function introduced via Neural Architecture Search (NAS). It outperformed ReLU in large-scale image classification, especially in deep and wide networks. Swish became the default activation in EfficientNet and is competitive with GELU.

# Key Idea
> Multiply each input by its sigmoid gate → smooth, trainable self-gating mechanism.

# Method
- **Formula**:  
  \[
  \text{Swish}(x) = x \cdot \sigma(\beta x)
  \]  
  where \(\sigma\) is the sigmoid function and \(\beta\) can be fixed or learned.
- **Variants**:  
  - Swish-1: fixed \(\beta = 1\).  
  - Swish-β: learnable \(\beta\).
- **Properties**:  
  - Smooth, differentiable, non-monotonic.  
  - Produces small negative outputs for positive inputs.  
  - Unbounded above, bounded below.

# Results
- Consistently outperformed ReLU across ImageNet benchmarks (ResNet, Inception, MobileNet).  
- Best gains in deep and wide networks.  
- Similar performance to GELU.  
- Core to EfficientNet’s success.

# Why it Mattered
- Showed that **smooth nonlinearities** can outperform ReLU at scale.  
- First widely adopted activation discovered via NAS.  
- Part of the lineage: ReLU → ELU → GELU → Swish → Mish.

# Architectural Pattern
- **Activation Functions**: nonlinear transformation.  
- **Self-Gating Mechanism**: sigmoid modulates input directly.  
- **Automated Discovery**: NAS designed rather than hand-crafted.

# Connections
- **Predecessors**: ReLU (2011), Maxout (2013), ELU (2015).  
- **Contemporaries**: GELU (2016).  
- **Successors**: Mish (2019).  
- **Influence**: EfficientNet (2019), explored in Transformers as GELU alternative.

# Implementation Notes
- More compute than ReLU due to sigmoid.  
- Easy to implement:  
  ```python
  def swish(x):
      return x * torch.sigmoid(x)
```
- PyTorch: `torch.nn.SiLU`.
- TensorFlow: `tf.nn.swish`.

# Critiques / Limitations

- Gains modest compared to GELU/ReLU in some tasks.
- Higher compute overhead vs ReLU.
- GELU became dominant in NLP models.

# Repro / Resources

- Paper: [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)
- Libraries: PyTorch (`SiLU`), TensorFlow (`swish`)
- Implemented in: EfficientNet, NASNet

---

# Educational Connections

## Undergraduate-Level Concepts

- **Linear Algebra**: elementwise operations.
- **Probability & Statistics**: sigmoid as probability gate.
- **Calculus**: smooth gradients vs ReLU cutoff.
- **Signals & Systems**: soft vs hard thresholding.
- **Optimization Basics**: stable training through smoother gradients.

## Postgraduate-Level Concepts

- **Advanced Optimization**: non-monotonic activations shape better gradient landscapes.
- **Numerical Methods**: sigmoid approximations for efficiency.
- **Machine Learning Theory**: smooth nonlinearities expand universal approximation.
- **Neural Network Design**: particularly effective in deep CNNs.
- **Research Methodology**: activation discovered via NAS on ImageNet.
---

# My Notes
- Relevant for comparing smooth activations (Swish vs GELU vs Mish).
- EfficientNet adoption shows practical impact.
- Could re-examine Swish in diffusion or transformer architectures where GELU dominates.
- Open question: does learnable β add significant value over fixed β?