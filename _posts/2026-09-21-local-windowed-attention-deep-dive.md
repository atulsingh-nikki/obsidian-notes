---
layout: post
title: "Local and Windowed Attention: Trading Reach for Efficiency"
description: "How windowed attention reduces Transformer cost for high-resolution vision while preserving communication across neighborhoods."
tags: [deep-learning, attention, local-attention, computer-vision]
---

Local attention restricts each query to nearby keys. If each window contains $w$ tokens and there are $N$ tokens overall, the cost is closer to $O(Nw)$ than $O(N^2)$.

The mask or tensor reshaping defines which tokens share a window. Within each window, attention remains content-dependent and multi-headed; only the candidate set is restricted. Shifted windows, pooling, or occasional global tokens provide routes across boundaries.

### A concrete example

For a high-resolution image, a model can process $7\times7$ patch windows instead of comparing every patch with every other patch. A window can understand an edge or small object locally, while shifted windows let that object communicate across the original partition.

## Where it is used

- Swin Transformer and hierarchical vision backbones.
- High-resolution image restoration and generation.
- Long documents and video models with local temporal context.

## What problem did it solve?

Dense global attention becomes impractical as image resolution grows. Local windows preserve content-dependent attention while limiting the number of pairwise comparisons.

## Benefits

- Much lower memory and compute for large inputs.
- Natural alignment with image locality.
- Fixed windows can create predictable deployment cost.

## Demerits

A token cannot see outside its window in one layer. Fixed boundaries can split objects, and the model may need shifted windows, pooling, or occasional global blocks to communicate across regions.

## What it made possible

Windowed attention made hierarchical, high-resolution Transformers practical for detection, segmentation, restoration, and other dense vision tasks.

## Continue

Windows impose a fixed neighborhood. [Sparse and deformable attention]({{ site.baseurl }}/2026/09/21/sparse-deformable-attention-deep-dive.html) lets the model choose a smaller set of locations more selectively.