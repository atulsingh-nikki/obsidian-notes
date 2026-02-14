---
title: "Backpropagation (1986)"
aliases:
  - Learning representations by back-propagating errors
  - Rumelhart-Hinton-Williams 1986
authors:
  - David E. Rumelhart
  - Geoffrey E. Hinton
  - Ronald J. Williams
year: 1986
venue: "Nature"
doi: "10.1038/323533a0"
arxiv: ""
code: ""
citations: 80,000+
dataset:
  - Synthetic toy tasks
  - Internal representation learning examples
tags:
  - paper
  - deep-learning
  - optimization
  - representation-learning
fields:
  - deep-learning
  - optimization
  - learning-theory
related:
  - "[[Optimal Brain Damage (1989)]]"
  - "[[LeNet-5 (1998)]]"
predecessors:
  - "[[Perceptron (1958)]]"
  - "[[Chain Rule (Calculus)]]"
successors:
  - "[[Optimal Brain Damage (1989)]]"
  - "[[Gradient-Based Learning Applied to Document Recognition (1998)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
**Rumelhart, Hinton, and Williams (1986)** showed how multi-layer neural networks can be trained efficiently by propagating output errors backward through each layer using the chain rule. This solved the core training bottleneck for hidden layers and transformed neural networks from mostly shallow models into practical deep function approximators.

# Key Idea
> Compute gradients for every weight in a multilayer network by recursively applying the chain rule from output to input.

# Method
- Define a differentiable network with hidden layers and a scalar loss.
- Run a **forward pass** to compute activations and output error.
- Run a **backward pass** to compute local error signals (deltas) layer by layer.
- Update parameters with gradient descent:

  \[
  w \leftarrow w - \eta \frac{\partial E}{\partial w}
  \]

- Reuse intermediate derivatives, making gradient computation efficient compared with naive finite-difference alternatives.

# Results
- Demonstrated successful learning of non-linear mappings that single-layer perceptrons cannot represent.
- Showed hidden units can learn useful intermediate representations automatically.
- Provided practical examples where iterative gradient learning converged on meaningful internal features.

# Why it Mattered
- Established the algorithmic foundation for nearly all modern deep learning training.
- Reintroduced neural networks as a scalable alternative to hand-crafted feature pipelines.
- Enabled later breakthroughs in vision, speech, language, and reinforcement learning.

# Architectural Pattern
- Differentiable computation graph.
- Forward evaluation + reverse-mode differentiation.
- Iterative optimization with first-order updates.

# Connections
- **Contemporaries**: Renewed interest in connectionist models after limits of single-layer perceptrons.
- **Influence**: Core training engine behind CNNs, RNNs, Transformers, and also pruning work like **OBD (1989)**.

# Implementation Notes
- Works best with differentiable activations and properly scaled learning rates.
- Early systems were sensitive to initialization and saturation with sigmoid/tanh activations.
- Batch and stochastic variants of gradient descent both apply naturally.

# Critiques / Limitations
- Can suffer from vanishing/exploding gradients in deep or recurrent networks.
- Requires differentiability and end-to-end gradient flow.
- Convergence can be slow or unstable without normalization/optimizer advances.

# Repro / Resources
- [Nature paper landing page](https://www.nature.com/articles/323533a0)
- [DOI link](https://doi.org/10.1038/323533a0)
- [Historical overview of backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Multivariable chain rule and partial derivatives.
- **Linear Algebra**: Matrix-vector products in layered transformations.
- **Optimization Basics**: Gradient descent and learning rates.
- **Probability & Statistics**: Error metrics and empirical risk.

## Postgraduate-Level Concepts
- **Advanced Optimization**: Conditioning, curvature, and gradient pathologies.
- **Numerical Methods**: Stability of recursive derivative computations.
- **Machine Learning Theory**: Representation learning and function approximation.
- **Neural Network Design**: Depth, activation choices, and trainability.

---

# My Notes
- Backprop is the bridge between expressive multilayer models and practical optimization.
- OBD relies on a trained backprop network, then adds second-order pruning logic on top.
- Most later deep learning work changes architecture/optimizer/data scale, not the core reverse-mode gradient idea.
