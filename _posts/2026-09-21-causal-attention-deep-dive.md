---
layout: post
title: "Causal Attention: Preventing the Model from Seeing the Future"
description: "Why causal masks matter for autoregressive generation, streaming systems, and valid next-token prediction."
tags: [deep-learning, attention, causal-modeling, transformers]
---

Causal attention masks future positions. At position $i$, the model may read positions $j \leq i$, but not $j>i$:

$$
M_{ij}=\begin{cases}0,&j\leq i,\\-\infty,&j>i.\end{cases}
$$

The mask is applied before softmax. A score of $-\infty$ becomes zero probability, so future tokens cannot contribute to the output. During training, all positions can still be computed in parallel because the full lower-triangular mask is applied to the shifted sequence.

### A concrete example

To predict the next word in “The cat sat on the,” the model may read that prefix but not the answer “mat.” The causal mask makes training obey the same information boundary as generation.

## Where it is used

- Autoregressive language models.
- Streaming speech, video, and sensor models.
- Any system that must make decisions online.

## What problem did it solve?

Training a generator with unrestricted self-attention would let it use the answer it is supposed to predict. The causal mask preserves the direction of generation and prevents information leakage.

## Benefits

- Training objective matches left-to-right inference.
- Supports generation one token or frame at a time.
- Makes the available information at each step explicit.

KV caching avoids recomputing old states during generation, but the cache grows with sequence length. An early wrong token also becomes context for later predictions, so causal generation has an error-propagation cost.

## Demerits

The model cannot use future context, even when future context would improve an offline decision. Sequential generation is also slower than parallel encoder inference, and long histories still create memory pressure.

## What it made possible

Causal attention made scalable autoregressive Transformers possible: the model can train on parallel shifted sequences while generating valid outputs one step at a time.

## Recommended reading

- [Evolving Self-Attention: Positional Encoding, Multi-Head, and Masked Attention](https://medium.com/@luvverma2011/evolving-self-attention-positional-encoding-multi-head-and-masked-attention-transformers-f818e5567f86) - includes the masked-attention perspective behind causal decoding.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - especially useful for the decoder's masked self-attention and autoregressive flow.

## Recommended video

- [Understanding causal attention or masked self attention](https://www.youtube.com/watch?v=CJSYo2Mw8R0) - a dedicated explanation of why future tokens are masked.

## Continue

Causal masking controls *when* information may flow. [Multi-head attention]({{ site.baseurl }}/2026/09/21/multi-head-attention-deep-dive.html) controls how many learned routing spaces operate in parallel.