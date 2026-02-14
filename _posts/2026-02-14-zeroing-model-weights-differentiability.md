---
layout: post
title: "What Actually Happens When You Set Model Weights to Zero (and Why Gradients Still Work)"
description: "A practical walkthrough of zeroing weights in deep learning code, autograd behavior, and why differentiability is usually preserved."
tags: [machine-learning, deep-learning, autograd, pytorch]
---

A common fear in deep learning is: **"If I set some weights to zero, won’t I break differentiability?"**

In most practical code paths, the answer is **no**. Zero values are still perfectly valid values for differentiable functions. What matters is **how** you zero things and whether your operation stays inside the autograd graph.

This post walks through the mechanics in code and the edge cases where things can go wrong.

## Table of Contents
- [Zero Is a Value, Not a Gradient Killer](#zero-is-a-value-not-a-gradient-killer)
- [Case 1: Initializing Parameters to Zero](#case-1-initializing-parameters-to-zero)
- [Case 2: Multiplying Weights by a Mask](#case-2-multiplying-weights-by-a-mask)
- [Case 3: Hard Thresholding with `where`](#case-3-hard-thresholding-with-where)
- [Why This Does Not Break Differentiability](#why-this-does-not-break-differentiability)
- [When You *Can* Break Training](#when-you-can-break-training)
- [A Safe Pattern for Structured Pruning](#a-safe-pattern-for-structured-pruning)
- [Key Takeaways](#key-takeaways)

## Zero Is a Value, Not a Gradient Killer

Suppose your layer is:

\[
y = Wx + b
\]

If one entry of \(W\) is zero, forward computation still works. Backprop still computes:

\[
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} x^T
\]

Nothing singular happens just because an entry in \(W\) equals 0. Zero is just another number.

## Case 1: Initializing Parameters to Zero

In PyTorch:

```python
import torch
import torch.nn as nn

linear = nn.Linear(4, 3)
with torch.no_grad():
    linear.weight.zero_()
    linear.bias.zero_()

x = torch.randn(2, 4)
out = linear(x).sum()
out.backward()

print(linear.weight.grad is None)  # False
```

Gradients are still computed. The main issue is not differentiability—it is **optimization symmetry**. For some models (especially multilayer networks), all-zero init can make units learn the same thing.

## Case 2: Multiplying Weights by a Mask

A very common pruning/reparameterization pattern is:

```python
masked_weight = weight * mask
output = x @ masked_weight.T
```

If `mask` is constant (no gradient), then:

\[
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial (W \odot M)} \odot M
\]

So masked entries get zero gradient, unmasked entries get normal gradient. This is exactly what you want for fixed sparsity.

## Case 3: Hard Thresholding with `where`

You might build a hard gate like:

```python
gated = torch.where(score > 0, weight, torch.zeros_like(weight))
```

This does **not** make the whole model non-differentiable. Gradients flow through the selected branch values. But the discrete condition `score > 0` is non-smooth as a function of `score`, so optimizing `score` directly can be unstable or zero almost everywhere.

That is why many methods use:
- straight-through estimators,
- soft gates (sigmoid/hard-concrete), or
- continuous relaxations during training.

## Why This Does Not Break Differentiability

Autograd tracks tensor operations and applies chain rule where derivatives exist (or subgradients in common piecewise cases). Operations like:
- multiplication by zero,
- addition with zero,
- selecting zero in one branch,

are still valid graph operations.

In short: **having zeros in tensors is not the same thing as having a non-differentiable program**.

## When You *Can* Break Training

You can run into trouble if you:

1. **Detach accidentally**
   ```python
   masked = (weight * mask).detach()  # kills gradient path to weight
   ```

2. **Write in-place on leaf parameters during forward** in ways autograd cannot reconcile.

3. **Use non-differentiable control flow for learnable decisions** (e.g., argmax-based routing) without a surrogate estimator.

4. **Zero everything and never unmask** so no trainable path remains.

So the failure mode is usually **graph disconnect** or **discrete optimization difficulty**, not the number zero itself.

## A Safe Pattern for Structured Pruning

A practical, stable pattern:

1. Keep a dense learnable parameter `weight`.
2. Apply a mask in forward: `effective_weight = weight * mask`.
3. Update only unmasked entries (automatic if mask is constant).
4. Optionally fine-tune after fixing sparsity.

Example:

```python
class MaskedLinear(nn.Module):
    def __init__(self, in_f, out_f, mask):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.register_buffer("mask", mask)  # not trainable

    def forward(self, x):
        w = self.weight * self.mask
        return x @ w.T + self.bias
```

This keeps autograd intact and enforces exact zeros where needed.

## Key Takeaways

- Setting weights to zero does **not** inherently break differentiability.
- Zeroed values still participate in normal autograd-tracked tensor math.
- Fixed masks naturally zero gradients on masked parameters.
- Real problems come from detached graphs or discrete gating, not from zero values.

If your training collapses after zeroing parameters, debug the **computation graph and gating logic** first—don’t blame zero itself.
