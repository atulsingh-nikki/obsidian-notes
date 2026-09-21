---
layout: post
title: "Cross-Attention: Let One Representation Read Another"
description: "How cross-attention connects modalities and feature streams, with its benefits, costs, and applications."
tags: [deep-learning, attention, cross-attention, multimodal-learning]
---

Cross-attention lets one sequence generate queries while another supplies keys and values:

$$
Q=XW_Q, \qquad K=YW_K, \qquad V=YW_V.
$$

The query stream is updated; the context stream is consulted.

This directionality is the defining idea. If image features form $Q$ and text forms $K,V$, each image location asks which words are relevant. If text forms $Q$ and image features form $K,V$, each word asks which visual evidence should update it. Swapping the streams changes the computation.

### A concrete example

In text-to-image diffusion, a noisy latent patch may query the text representation for “red bicycle near a tree.” One latent location can emphasize “bicycle,” another “tree,” and another “red,” with the weights changing at every denoising step.

## Where it is used

- Text conditioning in diffusion models.
- Vision-language models and visual question answering.
- DETR object queries reading image features.
- Encoder-decoder translation and summarization.

## What problem did it solve?

Concatenating modalities does not clearly say which stream should read or be updated. Cross-attention gives the architecture a directional interface: a text token, object query, or image latent can selectively retrieve information from another representation.

## Benefits

- Clean modality and feature-stream fusion.
- Query length and context length can differ.
- Context features can be reused or kept frozen.
- Supports conditioning without destroying each stream's representation.

## Demerits

The two streams must be projected into compatible spaces. Cross-attention adds memory and compute, and poor alignment can make the query attend to irrelevant context.

Repeated cross-attention can become expensive for long contexts. The query stream may overfit to superficial language or visual correlations, and a compressed context may not contain the fine-grained evidence the query needs.

## What it made possible

Cross-attention enabled text-controlled image generation, promptable detection and segmentation, and multimodal systems where one representation can ask another for task-specific evidence.

## Recommended reading

- [How Words Learn to Pay Attention: Transformers Part 1](https://medium.com/towards-artificial-intelligence/how-words-learn-to-pay-attention-transformers-part-1-08c34dd76721) - a readable introduction to the Transformer attention family.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - implementation-oriented reading for encoder-decoder attention and masking.

## Recommended video

- [Cross Attention in Transformers](https://www.youtube.com/watch?v=smOnJtCevoU) - CampusX's focused walkthrough of cross-attention inputs, processing, and outputs.

## Continue

Cross-attention changes the source of information. [Causal attention]({{ site.baseurl }}/2026/09/21/causal-attention-deep-dive.html) changes the time direction of information flow.