---
layout: post
title: "Self-Attention: Let Every Token Read the Same Sequence"
description: "Where self-attention is used, what it enables, and why content-dependent global context is worth its quadratic cost."
tags: [deep-learning, attention, self-attention, transformers]
---

Self-attention lets every token ask for information from other tokens in the same sequence. Each token produces a query, key, and value, then updates itself by mixing values according to query-key compatibility.

For an input sequence $X$, the three roles are learned projections of the same sequence:

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V.
$$

The score $q_i^\mathsf{T}k_j$ measures how useful token $j$ may be to token $i$. Softmax turns those scores into weights, and the weighted values become the new representation for token $i$. The receptive field is chosen from content, not just distance.

### A concrete example

In the sentence “The animal did not cross the road because it was tired,” the representation of “it” may need information from “animal,” “tired,” and the surrounding syntax. A fixed local filter must pass that evidence through several positions; self-attention creates a direct path in one layer. In a ViT, a patch containing a wheel can consult distant patches that reveal the whole vehicle.

## Where it is used

- Transformer language encoders and decoders.
- Vision Transformers, where image patches exchange context.
- Audio, video, document, and point-cloud models.

## What problem did it solve?

Before self-attention, long-range interaction usually required recurrence or many layers of local convolution. A token at one end of a sequence could not directly consult a distant token. Self-attention created a direct path between any two positions in one layer.

## Benefits

- Global receptive field in one operation.
- Content-dependent relationships rather than a fixed neighborhood.
- Highly parallel training because tokens can be processed together.

## Demerits

Dense attention builds an $N \times N$ score matrix, costing approximately $O(N^2)$. It also has weak built-in locality and translation bias, so it often needs positional information and substantial data.

Unrestricted access is not automatically useful: a model can attend to shortcuts, dilute a small signal among many tokens, or learn redundant heads. At high resolution, the attention matrix and its temporary gradients can dominate accelerator memory.

## What it made possible

Self-attention made one architecture adaptable across language, images, video, and multimodal inputs. It turned context selection into a learned operation instead of a fixed architectural neighborhood.

## Continue

The next question is what happens when the query sequence and the context sequence are different. That is the role of [cross-attention]({{ site.baseurl }}/2026/09/21/cross-attention-deep-dive.html).