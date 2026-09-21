---
layout: post
title: "Different Types of Attention in Deep Learning"
description: "A practical guide to self-attention, cross-attention, causal attention, local attention, sparse attention, linear attention, and spatial-temporal attention."
tags: [deep-learning, attention, transformers, computer-vision, multimodal-learning]
---

Attention is not one mechanism with one fixed meaning. It is a family of operations that lets a model assign different weights to different inputs before combining them.

The useful question is not only “does this model use attention?” It is:

> **Which tokens are allowed to interact, what information may they use, and how are the weights computed?**

This distinction matters because a Vision Transformer, a text decoder, a diffusion U-Net, and a video model may all use attention while solving very different information-flow problems.

The previous post established the vocabulary: queries express a need, keys determine relevance, and values carry the information that is retrieved. We can now keep that mechanism fixed and study the design choices around it. [Queries, Keys, and Values: The Intuition Behind Attention]({{ site.baseurl }}/2026/09/21/queries-keys-values-intuition.html) is the place to start if those roles are unfamiliar; this post builds on them by classifying attention according to who supplies $Q$, $K$, and $V$, and according to the connectivity mask that controls their interactions.

## The common mathematical core

Given queries $Q$, keys $K$, and values $V$, scaled dot-product attention is:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}} + M\right)V.
$$

Each query compares itself with the keys, converts the scores into weights, and uses those weights to combine the values. The mask $M$ determines which interactions are allowed. Many attention variants differ mainly in how they construct $Q$, $K$, $V$, or $M$.

For a token $i$, the output is a weighted mixture:

$$
z_i = \sum_j a_{ij}v_j,
\qquad
a_{ij} = \operatorname{softmax}_j\left(\frac{q_i^\mathsf{T}k_j}{\sqrt{d_k}} + M_{ij}\right).
$$

The matrix $A=[a_{ij}]$ is often called the attention map. Its shape and sparsity tell us which parts of the input can influence each output.

## 1. Self-attention

In self-attention, queries, keys, and values come from the same sequence:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V.
$$

Every token can use information from other tokens in the same input. In language, a word can use surrounding words. In an image, a patch can use distant patches. In video, a token can use other spatial or temporal positions.

[Vision Transformer]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html) uses self-attention to let image patches exchange information after patch embedding. [Attention Is All You Need]({{ site.baseurl }}/Research/2017/Attention%20Is%20All%20You%20Need%20(2017).html) introduced the Transformer architecture around this operation.

### Strengths

- Direct long-range interaction.
- Content-dependent relationships rather than fixed neighborhoods.
- A shared mechanism for language, images, audio, and video tokens.

### Cost

For $N$ tokens, dense self-attention forms an $N \times N$ score matrix. Its compute and memory are approximately $O(N^2)$, which becomes expensive for high-resolution images and long videos.

## 2. Cross-attention

Cross-attention uses one sequence to generate queries and another sequence to provide keys and values:

$$
Q = XW_Q, \qquad K = YW_K, \qquad V = YW_V.
$$

The query stream asks questions of the context stream. Examples include:

- text queries attending to image features;
- diffusion latents attending to text embeddings;
- object queries attending to image features in DETR;
- decoder tokens attending to encoder outputs.

[ViLBERT]({{ site.baseurl }}/Research/2019/ViLBERT%20Pretraining%20Task-Agnostic%20Visiolinguistic%20Representations%20(2019).html) uses cross-attention to align visual and linguistic streams. Stable Diffusion uses cross-attention to inject text conditioning into image-generation features.

Cross-attention is the main mechanism for **conditioning one representation on another**. It is different from simply concatenating the two sequences: the query stream retains its role while selectively reading from the context stream.

## 3. Causal or masked self-attention

Causal attention prevents a token from seeing future tokens. For an autoregressive sequence, the mask is:

$$
M_{ij} =
\begin{cases}
0, & j \leq i,\\
-\infty, & j > i.
\end{cases}
$$

The resulting lower-triangular pattern ensures that the prediction at position $i$ depends only on positions up to $i$. It is central to language generation and can also support streaming audio, video, and sensor models.

## 4. Multi-head attention

Multi-head attention runs several attention operations in parallel with different learned projections:

$$
\operatorname{MHA}(Q,K,V)
= \operatorname{Concat}(h_1,\ldots,h_H)W_O,
$$

where each head is:

$$
h_r = \operatorname{Attention}(QW_Q^{(r)},KW_K^{(r)},VW_V^{(r)}).
$$

Different heads can learn different relationships: local texture, long-range structure, object parts, syntax, or modality alignment. Multi-head attention is a **parallel representation pattern**, not a separate connectivity pattern. A multi-head layer can be self-attention, cross-attention, causal attention, or local attention depending on its inputs and mask.

## 5. Global attention

Global attention allows every query to interact with every key. It gives the widest receptive field in one layer, but its $O(N^2)$ cost limits the sequence length.

For a short sequence, global attention is often the simplest choice. For an image with thousands of patches or a long video, it can become the dominant memory cost.

## 6. Local or windowed attention

Local attention restricts each query to a neighborhood. In vision, the neighborhood is often a fixed spatial window. If the window contains $w$ tokens and there are $N$ total tokens, the cost is approximately:

$$
O(Nw)
$$

when $w$ is fixed, instead of $O(N^2)$.

[Swin Transformer]({{ site.baseurl }}/Research/2021/Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows%20(2021).html) uses window attention and shifts the windows between layers. The shift lets information cross window boundaries over depth while retaining efficient local computation.

Local attention is a good fit when nearby tokens contain strong signal, such as image texture and edges. It can struggle when a task needs immediate global coordination.

## 7. Sparse and deformable attention

Sparse attention allows only selected query-key pairs rather than all pairs. The selection may be fixed, learned, or content-dependent.

Deformable attention samples a small set of locations around learned reference points. [Deformable DETR]({{ site.baseurl }}/Research/2021/Deformable%20DETR%20Deformable%20Transformers%20for%20End-to-End%20Object%20Detection%20(2021).md) uses this idea to focus object queries on relevant multi-scale image features instead of attending densely to every location.

These patterns are useful when relevant evidence is concentrated, objects occupy different scales, or latency matters more than exhaustive global interaction. The trade-off is that a bad sampling pattern can hide important evidence.

## 8. Linear or kernelized attention

Linear attention tries to avoid explicitly materializing the full $N \times N$ attention matrix. Some variants use feature maps $\phi$ to rearrange the computation:

$$
\operatorname{Attention}(Q,K,V)
\approx \phi(Q)\left(\phi(K)^\mathsf{T}V\right).
$$

The aggregation can then be computed in an order that scales more nearly linearly with $N$. This can support longer sequences, but it changes the attention kernel and may lose some exact pairwise selectivity.

## 9. Spatial attention

Spatial attention assigns weights over locations in an image or feature map. It may emphasize object boundaries, salient parts, text regions, or foreground regions.

Spatial attention is often confused with self-attention. They describe different axes:

- **Self-attention:** where do $Q$, $K$, and $V$ come from?
- **Spatial attention:** over which image locations do the weights operate?

Self-attention can be spatial, but spatial attention can also be implemented with simpler gates or convolutions.

## 10. Channel attention

Channel attention weights feature channels rather than token-to-token spatial relationships. A typical block computes a summary of the spatial features and uses it to produce per-channel gates:

$$
\tilde{X}_{:,:,c} = g_c(X)X_{:,:,c}.
$$

It answers a different question from spatial attention:

- **Spatial attention:** where should the model look?
- **Channel attention:** which feature types should the model emphasize?

The two can be combined in convolutional and hybrid vision networks.

## 11. Temporal attention

Temporal attention operates across time. A video model may attend from a token in one frame to tokens in other frames, allowing it to model motion, persistence, and long-range events.

Common designs include:

- full space-time attention over spatial and temporal tokens;
- factorized spatial attention followed by temporal attention;
- temporal windows over nearby frames;
- causal temporal attention for streaming;
- memory attention over selected past features.

The design determines whether the model can use future frames, how much history it retains, and how its cost grows with clip length.

## 12. Soft and hard attention

Soft attention computes a differentiable weighted mixture of features. The model can distribute probability mass across several locations and train with ordinary backpropagation.

Hard attention selects discrete locations or regions. The selection may require sampling and gradient estimators such as REINFORCE, or differentiable relaxations.

[Show, Attend and Tell]({{ site.baseurl }}/Research/2015/Show,%20Attend%20and%20Tell%20Neural%20Image%20Caption%20Generation%20with%20Visual%20Attention%20(2015).html) is a useful reference for the distinction.

| Type | Selection | Training | Typical trade-off |
|---|---|---|---|
| Soft attention | Weighted combination | Direct backpropagation | Smooth but may spend computation broadly |
| Hard attention | Discrete samples or choices | Sampling or gradient estimator | Efficient selection but harder optimization |

## These categories can be combined

Attention labels are not mutually exclusive. A single layer can be described in several ways:

- **causal multi-head self-attention:** multiple heads, same sequence, no future tokens;
- **local spatial self-attention:** same image feature map, restricted to windows;
- **cross-attention with multi-head projections:** text queries reading image features;
- **deformable temporal attention:** video queries selecting a small set of frames and locations;
- **channel-spatial attention:** separate gates over feature channels and spatial positions.

The clearest description asks:

1. What produces the queries?
2. Where do the keys and values come from?
3. Which pairs are allowed to interact?
4. Are the weights dense, sparse, local, or sampled?
5. Is the output computed with full softmax attention or an approximation?

## A practical comparison

| Attention type | Main interaction | Typical use | Main cost or risk |
|---|---|---|---|
| Self-attention | Tokens within one sequence | ViT, language encoders | Quadratic dense cost |
| Cross-attention | One stream reads another | VLMs, diffusion conditioning | Fusion cost and alignment |
| Causal attention | Past tokens only | Autoregressive generation | Cannot use future context |
| Global attention | All positions | Short sequences, global reasoning | $O(N^2)$ scaling |
| Local/windowed attention | Nearby positions | High-resolution vision | Limited immediate receptive field |
| Sparse/deformable attention | Selected positions | Detection, long inputs | Missed evidence if sampling fails |
| Linear attention | Approximate kernel interaction | Long sequences | May change modeling capacity |
| Spatial attention | Image locations | Saliency and dense tasks | Can over-focus on shortcuts |
| Channel attention | Feature channels | CNN and hybrid blocks | Does not model token relations alone |
| Temporal attention | Across frames or time | Video and streaming | Memory and temporal bias |
| Soft attention | Differentiable mixture | Captioning and feature weighting | Broad computation |
| Hard attention | Discrete selection | Glimpse or region selection | Difficult optimization |

## How to choose an attention pattern

Choose based on the information flow required by the task:

- Use **global self-attention** when the sequence is short and relationships may be arbitrary.
- Use **local or windowed attention** when spatial neighborhoods dominate and inputs are large.
- Use **cross-attention** when one representation needs to condition on another modality or feature stream.
- Use **causal attention** when the model must operate left to right or online.
- Use **sparse or deformable attention** when relevant evidence is concentrated and the input is expensive.
- Use **temporal attention** when persistence, motion, or event order matters.
- Use **channel attention** when the main problem is selecting useful feature types within a CNN or hybrid block.
- Use **linear attention** only after checking that its approximation preserves the task's important interactions.

The best design is often hybrid. A video system may use local spatial attention, sparse temporal memory, and cross-attention from a text prompt. An image editor may use a frozen image encoder, cross-attention for conditioning, and local attention in a high-resolution decoder.

## Final perspective

Attention is best understood as a programmable information-routing mechanism. The names matter less than the connectivity pattern and the computational budget behind them.

When reading a new architecture, draw the attention graph:

- nodes are tokens, patches, regions, channels, or frames;
- directed edges show which inputs can influence each output;
- masks and sampling define the allowed edges;
- the score function defines how strongly the edges are weighted.

Once that graph is visible, “self-attention,” “cross-attention,” “windowed attention,” and “deformable attention” become precise variations on the same idea rather than a list of unrelated terms.

## Recommended reading

- [Evolving Self-Attention: Positional Encoding, Multi-Head, and Masked Attention](https://medium.com/@luvverma2011/evolving-self-attention-positional-encoding-multi-head-and-masked-attention-transformers-f818e5567f86) - a Medium series-style overview of several attention variants.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - a visual companion for comparing self-attention, masking, and encoder-decoder attention.

## Recommended video

- [Introduction to Transformers with Andrej Karpathy](https://www.youtube.com/watch?v=XfpMkf4rD6E) - Stanford CS25's broad treatment of Transformer components and attention.

## Continue reading

- [Attention Is All You Need]({{ site.baseurl }}/Research/2017/Attention%20Is%20All%20You%20Need%20(2017).html)
- [Vision Transformer]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html)
- [Swin Transformer]({{ site.baseurl }}/Research/2021/Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows%20(2021).html)
- [Show, Attend and Tell]({{ site.baseurl }}/Research/2015/Show,%20Attend%20and%20Tell%20Neural%20Image%20Caption%20Generation%20with%20Visual%20Attention%20(2015).html)