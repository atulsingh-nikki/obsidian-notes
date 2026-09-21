---
layout: post
title: "Multi-Head Attention: Several Routing Spaces in Parallel"
description: "Why Transformers use multiple attention heads, where multi-head attention helps, and why heads are not automatically human-readable concepts."
tags: [deep-learning, attention, multi-head-attention, transformers]
---

Multi-head attention runs several attention operations with separate query, key, and value projections:

$$
h_r=\operatorname{Attention}(QW_Q^{(r)},KW_K^{(r)},VW_V^{(r)}).
$$

The head outputs are concatenated and projected back into the model dimension.

If the model dimension is $d$ and there are $h$ heads, each head commonly works in a smaller dimension $d_h=d/h$. The heads see different learned projections of the same tokens, and their outputs are recombined.

### A concrete example

In language, one head may connect a pronoun to its antecedent while another tracks negation. In an image, one head may favor nearby texture while another connects object parts across the frame. These are useful intuitions, not guarantees that each head has one human-readable meaning.

## Where it is used

It is the standard attention block in language Transformers, Vision Transformers, multimodal encoders, and diffusion architectures.

## What problem did it solve?

A single attention space must compromise between different relationships. Multi-head attention lets the model compute several kinds of compatibility at once: local, global, syntactic, spatial, or cross-modal.

## Benefits

- Multiple relational subspaces.
- Parallel computation.
- Different heads can use different masks or focus patterns.
- More expressive than one projection with the same total interface.

## Demerits

More heads do not guarantee more useful information. Heads can be redundant, interpretation is unreliable, and the projections add parameters and memory traffic. Small head dimensions can also make each head weak.

## What it made possible

Multi-head attention allowed one layer to represent several relationships without hand-designing separate modules for each one. It became a flexible general-purpose block for sequence and spatial reasoning.

## Continue

Multi-head attention says how many routing spaces exist. [Global attention]({{ site.baseurl }}/2026/09/21/global-attention-deep-dive.html) says how broadly each space can connect.