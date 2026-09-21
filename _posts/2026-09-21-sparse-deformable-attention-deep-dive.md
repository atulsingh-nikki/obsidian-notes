---
layout: post
title: "Sparse and Deformable Attention: Learn Where to Look"
description: "How selected and learned sampling patterns make attention efficient for detection, multi-scale vision, and large inputs."
tags: [deep-learning, attention, sparse-attention, deformable-attention, object-detection]
---

Sparse attention removes most query-key pairs. Deformable attention goes further by learning a small set of sampling locations around reference points.

For a query at reference location $p$, deformable attention predicts offsets $\Delta p_k$ and weights $a_k$, then samples features at $p+\Delta p_k$:

$$
y(p)=\sum_{k=1}^{K}a_k\,x\left(p+\Delta p_k\right).
$$

The model learns both **where** to look and **how much** to use each sampled feature, unlike a fixed window whose geometry is chosen before seeing the image.

### A concrete example

An object query for a small car can sample the car's corners and interior across a feature pyramid rather than inspect every image location. If the car grows or moves, learned offsets can move with it.

## Where it is used

- Deformable DETR object detection.
- Multi-scale feature pyramids.
- Long sequences where only selected context is relevant.
- Video memory and tracking systems.

## What problem did it solve?

Dense attention wastes computation comparing a query with locations that cannot explain it. Fixed windows are cheaper but may be poorly aligned with object size or shape. Deformable attention learns where evidence is likely to be.

## Benefits

- Efficient access to large feature maps.
- Adaptive to object scale and geometry.
- Natural fit for detection queries and multi-scale features.

## Demerits

The model can miss important evidence if its sampling locations are wrong. The implementation is more complex, irregular memory access can hurt hardware efficiency, and the learned sparsity may be harder to inspect.

## What it made possible

Sparse and deformable attention made end-to-end detection practical without requiring every object query to inspect every pixel or feature location.

## Recommended reading

- [Hierarchical Attention Transformers](https://medium.com/@ceo_44783/16x16x16x16-hierarchical-attention-transformers-how-to-train-an-llm-with-a-65-536-token-context-3f348ed38370) - useful context for reducing attention's effective search space.
- [Deformable DETR](https://arxiv.org/abs/2010.04159) - the original research paper for learned sparse sampling around reference points.

## Recommended video

- [Introduction to Transformers with Andrej Karpathy](https://www.youtube.com/watch?v=XfpMkf4rD6E) - the Stanford CS25 lecture gives useful context for why learned routing is valuable before studying deformable sampling.

## Continue

Sparse attention chooses fewer pairs. [Linear attention]({{ site.baseurl }}/2026/09/21/linear-attention-deep-dive.html) changes the computation itself to avoid explicitly forming all pairwise scores.