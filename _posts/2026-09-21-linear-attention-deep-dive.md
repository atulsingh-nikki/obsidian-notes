---
layout: post
title: "Linear Attention: Escaping the Quadratic Bottleneck"
description: "How linear and kernelized attention reorganize or approximate attention for long sequences, and what they trade away."
tags: [deep-learning, attention, linear-attention, efficient-transformers]
---

Standard softmax attention forms $QK^\mathsf{T}$, an $N \times N$ matrix. Linear attention uses an alternative kernel or computation order, often schematically written as:

$$
\operatorname{Attention}(Q,K,V) \approx \phi(Q)\left(\phi(K)^\mathsf{T}V\right).
$$

The computational trick is to aggregate keys and values before applying a query. Instead of storing one score for every query-key pair, the model maintains a summary such as $\phi(K)^\mathsf{T}V$. With suitable feature maps and normalization, memory grows toward a function of $N$ times the feature dimension rather than a full $N\times N$ matrix.

### A concrete example

In a streaming sensor model, new measurements arrive continuously and cannot wait for a full sequence-wide matrix. A linear-attention state can be updated as observations arrive, giving the current query access to accumulated history.

## Where it is used

- Long-context sequence models.
- Streaming audio and sensor processing.
- Long video or high-resolution feature processing.
- Memory-efficient Transformer variants.

## What problem did it solve?

Quadratic attention becomes impossible when sequence length is very large. Reordering the aggregation or using a feature-map approximation can reduce dependence on $N^2$ memory and compute.

## Benefits

- Longer contexts within a fixed memory budget.
- Better streaming behavior in some designs.
- Can avoid materializing the full attention matrix.

## Demerits

The approximation may not preserve the sharp pairwise selection of softmax attention. Some variants have weaker performance on tasks needing precise retrieval, and kernel choice, normalization, and numerical stability become important.

## What it made possible

Linear attention opened a path toward Transformer-like models for sequences too long for dense attention, including online and memory-constrained applications.

## Continue

Efficiency is only one axis. [Spatial attention]({{ site.baseurl }}/2026/09/21/spatial-attention-deep-dive.html) focuses on which locations matter in visual features.