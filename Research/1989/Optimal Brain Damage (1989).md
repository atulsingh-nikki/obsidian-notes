---
title: "Optimal Brain Damage (1989)"
aliases:
  - OBD
  - Optimal Brain Damage: Pruning Neural Networks
authors:
  - Yann LeCun
  - John S. Denker
  - Sara A. Solla
year: 1989
venue: "NeurIPS 2 (NIPS 1989)"
doi: "10.5555/2969830.2969879"
arxiv: ""
code: ""
citations: 8,000+
dataset:
  - MNIST-like handwritten digit recognition tasks
tags:
  - paper
  - deep-learning
  - pruning
  - model-compression
fields:
  - deep-learning
  - optimization
  - efficient-ml
related:
  - "[[Optimal Brain Surgeon (1992)]]"
  - "[[LeNet-5 (1998)]]"
predecessors:
  - "[[Backpropagation (1986)]]"
  - "[[Second-Order Optimization Methods]]"
successors:
  - "[[Optimal Brain Surgeon (1992)]]"
  - "[[The Lottery Ticket Hypothesis (2019)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
**Optimal Brain Damage (OBD)** introduced one of the first principled methods for **pruning neural networks**. Instead of removing weights using only magnitude, OBD estimates how much each parameter contributes to the loss by using a **second-order (Hessian-based) saliency approximation**, then removes weights predicted to hurt performance the least.

# Key Idea
> Prune parameters with the smallest estimated increase in objective value, using a diagonal Hessian approximation for tractable second-order saliency.

# Method
- Start from a trained network near a local minimum.
- Approximate the loss change from deleting weight \(w_i\) with a second-order Taylor expansion:

  \[
  \Delta E_i \approx \frac{1}{2} h_{ii} w_i^2
  \]

  where \(h_{ii}\) is the \(i\)-th diagonal entry of the Hessian.
- Define each weight's **saliency** as \(S_i = \frac{1}{2} h_{ii} w_i^2\).
- Iteratively remove low-saliency weights, then fine-tune the network.

# Results
- Showed that large fractions of parameters could be removed while preserving accuracy.
- Demonstrated that second-order saliency-based pruning outperformed naive magnitude-only pruning in their experiments.
- Reduced model size and computational cost without major degradation in recognition quality.

# Why it Mattered
- One of the earliest strong arguments that neural networks are often **over-parameterized**.
- Introduced a formal pruning criterion rooted in optimization geometry.
- Helped establish the foundation for later compression, sparsification, and efficient inference research.

# Architectural Pattern
- Train dense model -> compute parameter importance -> prune -> retrain/fine-tune.
- Second-order sensitivity analysis as a reusable model compression pattern.

# Connections
- **Contemporaries**: Early backprop-era efforts to improve generalization and reduce overfitting.
- **Influence**: Directly inspired **Optimal Brain Surgeon**, modern pruning pipelines, and renewed sparse training work (including lottery-ticket-style thinking).

# Implementation Notes
- Full Hessian is expensive; OBD uses only diagonal terms for practicality.
- Best applied after the model is reasonably converged.
- Pruning is typically done in rounds with short retraining phases between rounds.

# Critiques / Limitations
- Diagonal Hessian approximation ignores cross-parameter interactions.
- Estimating Hessian information can still be noisy or expensive for large modern models.
- Structured sparsity (channel/filter pruning) is often preferred for hardware efficiency today.

# Repro / Resources
- [NeurIPS proceedings entry](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)
- [Paper PDF](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Taylor expansion and second derivatives.
- **Linear Algebra**: Hessian matrix and diagonal approximations.
- **Optimization Basics**: Local minima and sensitivity of objective functions.
- **Data Structures**: Sparse vs dense parameter representations.

## Postgraduate-Level Concepts
- **Advanced Optimization**: Second-order approximations in non-convex problems.
- **Numerical Methods**: Practical Hessian estimation trade-offs.
- **Machine Learning Theory**: Capacity control and compression-generalization connections.
- **Neural Network Design**: Compression pipelines for deployment constraints.

---

# My Notes
- OBD is an early, elegant example of using optimization curvature for model compression.
- The diagonal Hessian assumption is the key practical compromise.
- Conceptually similar to modern saliency/importance pruning, but decades earlier.
