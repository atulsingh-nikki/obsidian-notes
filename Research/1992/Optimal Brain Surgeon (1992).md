---
title: "Optimal Brain Surgeon (1992)"
aliases:
  - OBS
  - Optimal Brain Surgeon and General Network Pruning
authors:
  - Babak Hassibi
  - David G. Stork
  - Gregory J. Wolff
year: 1992
venue: "NeurIPS 5 (NIPS 1992)"
doi: ""
arxiv: ""
code: ""
citations: 3,000+
dataset:
  - MNIST-like handwritten digit classification benchmarks
tags:
  - paper
  - deep-learning
  - pruning
  - model-compression
  - second-order-methods
fields:
  - deep-learning
  - optimization
  - efficient-ml
related:
  - "[[Optimal Brain Damage (1989)]]"
  - "[[LeNet-5 (1998)]]"
predecessors:
  - "[[Optimal Brain Damage (1989)]]"
  - "[[Second-Order Optimization Methods]]"
successors:
  - "[[The Lottery Ticket Hypothesis (2019)]]"
  - "[[Movement Pruning (2020)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
**Optimal Brain Surgeon (OBS)** extends Optimal Brain Damage by using the **inverse Hessian** instead of only diagonal Hessian terms, allowing pruning decisions that account for interactions between parameters. The method removes one parameter at a time while applying a compensating update to remaining weights, minimizing the predicted increase in loss.

# Key Idea
> When deleting a weight, update the remaining parameters using second-order curvature information so the network stays as close as possible to the original optimum.

# Method
- Assume the network has been trained near a local minimum where gradients are small.
- Use a constrained second-order optimization problem: set one weight to zero while minimizing the loss increase under a quadratic approximation.
- With Hessian \(H\), inverse Hessian \(H^{-1}\), and weight \(w_q\) selected for removal, the saliency is:

  \[
  \Delta E_q \approx \frac{w_q^2}{2\,[H^{-1}]_{qq}}
  \]

- Choose the weight with smallest predicted \(\Delta E_q\), prune it, and apply the compensating update to all remaining weights:

  \[
  \delta \mathbf{w} = -\frac{w_q}{[H^{-1}]_{qq}}\,H^{-1}\mathbf{e}_q
  \]

- Repeat pruning iteratively with periodic retraining or recomputation.

# Results
- Demonstrated better pruning quality than diagonal-only methods in small to medium neural networks.
- Preserved task accuracy at higher sparsity levels compared with simpler heuristics.
- Showed that curvature-aware compensation can significantly delay performance collapse during iterative pruning.

# Why it Mattered
- Provided a mathematically cleaner pruning rule than OBD by modeling cross-parameter coupling.
- Helped formalize pruning as a constrained second-order optimization problem.
- Became a foundational reference for curvature-aware pruning and later second-order compression work.

# Architectural Pattern
- Train dense model -> estimate curvature (or inverse curvature) -> select low-saliency weight -> compensate remaining weights -> iterate.
- Pruning is treated as an optimization step, not only a ranking heuristic.

# Connections
- **Builds on**: OBD's second-order saliency idea, but removes the diagonal approximation.
- **Influence**: Modern Hessian/Fisher-based pruning, low-rank curvature approximations, and iterative sparse fine-tuning pipelines.

# Implementation Notes
- Direct Hessian inversion is expensive (\(O(n^3)\)); practical use needs approximations for large models.
- Works best when parameters are close to a local optimum.
- Often combined with blockwise, diagonal-plus-low-rank, or Kronecker-factored approximations in modern settings.

# Critiques / Limitations
- Computationally prohibitive for today's very large neural networks without approximation.
- Sensitivity to curvature estimation quality and numerical conditioning.
- Unstructured sparsity may not translate to real hardware speedups unless deployment kernels are sparse-aware.

# Repro / Resources
- [NeurIPS proceedings entry](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)
- [Paper PDF](https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Second-order Taylor approximations of objective functions.
- **Linear Algebra**: Hessian, inverse Hessian, basis vectors, and quadratic forms.
- **Optimization Basics**: Constrained minimization and local curvature.
- **Algorithms**: Iterative greedy pruning with corrective updates.

## Postgraduate-Level Concepts
- **Numerical Optimization**: KKT-style constrained quadratic minimization.
- **Scientific Computing**: Stable inverse/approximate inverse curvature estimation.
- **Machine Learning Systems**: Compression-quality vs compute trade-offs.
- **Sparse Deep Learning**: Connections between pruning dynamics and generalization.

---

# My Notes
- OBS is conceptually elegant because pruning is accompanied by an explicit compensation step.
- The method highlights why curvature coupling matters: deleting one weight perturbs many others.
- Practical modern value lies in approximation strategies inspired by OBS rather than exact inverse-Hessian computation.
