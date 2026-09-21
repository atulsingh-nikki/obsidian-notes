---
layout: post
title: "Soft and Hard Attention: Weight Everything or Select a Few"
description: "The difference between differentiable soft attention and discrete hard attention, including their uses, benefits, and optimization trade-offs."
tags: [deep-learning, attention, soft-attention, hard-attention]
---

Soft attention computes a differentiable weighted mixture of features. Hard attention selects a discrete location, region, or token and processes only that choice.

Soft attention computes a mixture such as:

$$
y=\sum_i a_i v_i, \qquad a_i\geq 0, \qquad \sum_i a_i=1.
$$

Hard attention instead samples or chooses an index $z$ and uses $y=v_z$. The first is easy to differentiate but may process many candidates; the second can save computation but makes the choice discontinuous.

### A concrete example

An image-captioning model can softly combine several image regions while generating a word. An active-vision agent might make a hard glimpse decision and crop to one region, avoiding full-resolution processing of the entire image.

## Where they are used

- Soft attention: Transformers, captioning, diffusion, multimodal fusion, and feature weighting.
- Hard attention: visual glimpses, region selection, routing, token pruning, and active perception.

## What problem did they solve?

Uniform processing wastes capacity, but a fully discrete choice is difficult to train. Soft attention solves the optimization problem with smooth weights. Hard attention solves the computation problem by selecting a small amount of evidence.

## Benefits of soft attention

- Trains with ordinary backpropagation.
- Can combine evidence from several locations.
- Stable and easy to compose with other layers.

## Benefits of hard attention

- Can reduce computation and memory.
- Produces explicit decisions about where to look.
- Useful when only a few regions should be processed.

## Demerits

Soft attention may spread computation broadly and produce diffuse, hard-to-interpret maps. Hard attention introduces discrete sampling, which usually needs reinforcement learning, straight-through estimators, or a differentiable approximation.

## What they made possible

Soft attention made attention broadly trainable inside end-to-end neural networks. Hard attention opened the door to learned glimpses and adaptive computation, where the model chooses what not to process.

## Final perspective

The attention family is not a list of competing inventions. It is a set of choices about source streams, connectivity, time direction, spatial scope, feature dimensions, and whether selection is smooth or discrete. The right choice depends on what information the task needs and what computation the deployment budget allows.

## Recommended reading

- [How Words Learn to Pay Attention: Transformers Part 1](https://medium.com/towards-artificial-intelligence/how-words-learn-to-pay-attention-transformers-part-1-08c34dd76721) - a readable introduction to soft attention as learned weighting.
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) - the classic soft-versus-hard visual attention reference for image captioning.

## Recommended video

- [Attention in transformers, step-by-step - Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) - a clear visual comparison point for differentiable weighted routing.

At this point we have described the main ways attention can route information. The research question is now sharper: how can we tell whether a particular routing pattern is meaningful, necessary, or actually useful? [How Do We Know Attention Helps?]({{ site.baseurl }}/2026/09/21/measuring-whether-attention-helps.html) answers that question with visualization, alignment, ablation, attribution, and causal tests.