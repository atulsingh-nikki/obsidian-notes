---
layout: post
title: "Global Attention: Every Position Can See Every Position"
description: "The benefits and costs of dense global attention, and when its complete receptive field is worth quadratic scaling."
tags: [deep-learning, attention, global-attention, transformers]
---

Global attention permits every query to compare with every key. Its attention matrix is dense and has shape $N \times N$ for $N$ tokens.

For each output token, there is no preselected neighborhood. A patch at the top-left can compare directly with a patch at the bottom-right, and the comparison can change with image content. Global attention is maximum connectivity, not a claim that every pair is equally important.

### A concrete example

A small object may be recognized only by combining distant evidence: a handle on one side of an image and a cup body on the other. A global layer connects those patches immediately; local designs need several layers, pooling, or a global token to bring them together.

## Where it is used

- Short and medium-length language sequences.
- Standard Vision Transformers at moderate resolution.
- Global context blocks in image and video models.

## What problem did it solve?

Local operators can miss relationships between distant positions. Global attention makes the full input available immediately, so a patch can relate to another patch on the opposite side of an image.

## Benefits

- Maximum receptive field in one layer.
- No need to guess a fixed neighborhood.
- Useful for global composition, long-range dependencies, and object-part relations.

## Demerits

Compute and memory grow as $O(N^2)$. High-resolution images and long videos quickly make the score matrix the bottleneck. Dense access can also spend capacity on irrelevant pairs.

## What it made possible

Global attention provided a simple, general mechanism for scene-wide reasoning and helped make the same Transformer block usable across language and vision.

## Recommended reading

- [Large Language Models II: Attention, Transformers and LLMs](https://medium.com/@mitultiwari/large-language-models-ii-attention-transformers-and-llms-6107cf37232e) - broader reading on attention and Transformer context.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - useful reference implementation for dense scaled dot-product attention.

## Recommended video

- [Attention in transformers, step-by-step - Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) - especially useful for seeing how a token can connect to distant context.

## Continue

When $N$ is too large, [local and windowed attention]({{ site.baseurl }}/2026/09/21/local-windowed-attention-deep-dive.html) trades immediate global access for scalable neighborhoods.