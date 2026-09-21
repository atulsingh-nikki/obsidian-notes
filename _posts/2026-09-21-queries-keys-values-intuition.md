---
layout: post
title: "Queries, Keys, and Values: The Intuition Behind Attention"
description: "An intuitive explanation of what queries, keys, and values represent, why attention uses three projections, and how the idea maps to language, images, and cross-attention."
tags: [deep-learning, attention, transformers, computer-vision, intuition]
---

The letters $Q$, $K$, and $V$ can make attention look more mysterious than it is. They are not three different kinds of data stored in the input. They are three different **roles** that the same information can play:

- **Query:** What am I looking for?
- **Key:** What kind of information do I contain?
- **Value:** What information should I provide if selected?

Attention uses queries and keys to decide **where to look**, then uses the values to decide **what to bring back**.

Before we compare different kinds of attention, we need this common vocabulary. Once the roles of $Q$, $K$, and $V$ are clear, the next question is how the sources and destinations of that routing change across self-attention, cross-attention, causal attention, local attention, and other patterns.

## The search-engine analogy

Imagine searching a document collection.

- Your search request is the **query**.
- Each document's index label is its **key**.
- The document content returned to you is its **value**.

The search system compares your query with every key. A document whose key matches the query receives a high score. The final answer is a weighted combination of the values from the most relevant documents.

Attention does the same thing inside a neural network:

1. Create a query describing the current information need.
2. Compare it with the keys available in the context.
3. Turn the comparison scores into weights.
4. Mix the corresponding values using those weights.

The important separation is this:

> Keys decide **which information is relevant**; values contain **the information that gets transmitted**.

## A tiny example

Suppose an image contains three patches:

| Patch | Key describes | Value contains |
|---|---|---|
| 1 | blue sky texture | visual features of the sky |
| 2 | curved wheel-like structure | visual features of a wheel |
| 3 | dark road texture | visual features of the road |

Now consider a query from a patch that appears to belong to a car. Its query may match the key for the wheel more strongly than the keys for sky or road:

$$
\text{scores} = [0.1, 2.4, 0.3].
$$

After softmax, the weights might be approximately:

$$
\text{weights} = [0.07, 0.86, 0.07].
$$

The output for the querying patch becomes mostly the wheel value, with small contributions from sky and road. The patch has not copied another patch directly. It has updated its representation using information selected from the whole context.

## Why not use the input directly?

The model learns three projections of each input representation:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V.
$$

The matrices $W_Q$, $W_K$, and $W_V$ allow the same token to be viewed in three task-specific ways.

Consider a word such as “bank.”

- Its **query** might ask: what nearby words help resolve my meaning?
- Its **key** might advertise: I represent a financial institution or a river edge.
- Its **value** contains the features that should be passed to another token if the match is useful.

If query, key, and value were forced to use exactly the same representation, the model would have less freedom to separate matching from information transfer. The three projections let attention learn a retrieval space and a communication space at the same time.

## The equation in plain language

Scaled dot-product attention is:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V.
$$

Read it from right to left conceptually:

1. $QK^\mathsf{T}$ compares every query with every key.
2. Dividing by $\sqrt{d_k}$ keeps the scores numerically well behaved as the key dimension grows.
3. Softmax turns scores into positive weights that sum to one.
4. Multiplying by $V$ collects the information selected by those weights.

For one query $q_i$:

$$
z_i = \sum_j a_{ij}v_j,
\qquad
a_{ij} = \operatorname{softmax}_j\left(\frac{q_i^\mathsf{T}k_j}{\sqrt{d_k}}\right).
$$

In words: output $z_i$ is a weighted mixture of all values, where the weights are determined by how well query $i$ matches each key.

## What does “matching” mean?

The dot product is not necessarily literal semantic similarity in the human sense. It is similarity in a learned feature space. During training, the model adjusts the projections so that useful dependencies receive high compatibility scores.

In language, a query may learn to match:

- a pronoun with its antecedent;
- a verb with its subject;
- a word with nearby negation;
- a token with a long-range topic cue.

In images, a query may learn to match:

- a patch with other parts of the same object;
- a texture with a repeated texture elsewhere;
- a small detail with a global object context;
- a region with a text prompt in cross-attention.

The model does not receive a rule saying “look for the wheel.” It learns useful query-key relationships from the training objective.

## Self-attention: everyone is both a searcher and a document

In self-attention, all queries, keys, and values come from the same sequence:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V.
$$

Every token can ask a question of every other token, including itself. A word, image patch, or video token plays all three roles simultaneously:

- it asks for context through its query;
- it advertises its content through its key;
- it offers transferable features through its value.

This is why self-attention is more than a fixed filter. The neighborhood is not chosen only by physical distance. It is chosen dynamically from the content and the learned compatibility function.

## Cross-attention: one stream asks another stream

In cross-attention, queries and keys/values come from different sources:

$$
Q = XW_Q, \qquad K = YW_K, \qquad V = YW_V.
$$

For text-conditioned image generation, image or latent features may form the queries while text embeddings provide keys and values. The latent asks, in effect:

> Which parts of the text are relevant to updating this location?

For a vision-language model, text tokens may query image features. For object detection, object queries can query an image feature map. The direction matters: it tells us which representation is being updated and which representation is being consulted.

## Keys are not labels and values are not raw pixels

The names can create misleading intuitions.

- A **key** is not a human-readable label or database identifier. It is a learned vector used for matching.
- A **value** is not necessarily the original input. It is a learned projection containing features useful for downstream computation.
- A **query** is not a question written in natural language. It is a learned vector representing the current information need.

The search analogy explains the roles, but the objects being searched are continuous learned vectors.

## Why the value projection matters

Suppose two keys match a query equally well. The output should not merely say “these two things matched.” It should bring back useful content from both sources.

That content may emphasize different dimensions from the ones used for matching. For example, a key can encode “this patch belongs to a wheel-like region,” while its value carries shape, texture, and position features that help the next layer reason about the object.

This separation lets the model use one subspace for routing and another for communication.

## Attention is a learned information-routing system

An attention layer can be viewed as a soft routing system:

- queries are requests;
- keys are routing addresses;
- attention weights are routing probabilities;
- values are the payloads;
- the output is the delivered message.

Unlike a hard lookup table, the router can send a fractional amount of information from many sources. Unlike a fixed convolution, it can change its effective receptive field based on content.

## Where multi-head attention fits

Multi-head attention creates several independent sets of projections:

$$
h_r = \operatorname{Attention}(QW_Q^{(r)},KW_K^{(r)},VW_V^{(r)}).
$$

One head might learn a short-range relation while another tracks a long-range dependency. In an image, heads may specialize in boundaries, object parts, or global context. The model is not required to assign one clean meaning to every head, but separate heads give it multiple routing spaces.

The phrase “multi-head” therefore describes how many $Q/K/V$ projection sets run in parallel. It does not replace the distinction between self-attention and cross-attention.

## A practical debugging checklist

When reading an attention implementation, ask:

1. What tensor creates $Q$?
2. What tensor creates $K$ and $V$?
3. Do the query and context sequences have the same length?
4. What does a row of the score matrix represent?
5. What mask prevents forbidden interactions?
6. What does the output replace or update?

These questions usually reveal the architecture faster than the layer's name.

## One-sentence summary

> A query says what a token needs, keys say what each token can be matched by, and values say what each token contributes once selected.

That is the conceptual core behind the equation. Attention first computes relevance, then uses that relevance to route information.

### From roles to patterns

We now have the basic mechanism: queries express a need, keys determine relevance, and values carry the retrieved information. The next step is to ask what changes when we change the participants and the allowed connections. [Different Types of Attention in Deep Learning]({{ site.baseurl }}/2026/09/21/different-types-of-attention.html) answers that question by keeping $Q/K/V$ fixed while changing the information-flow rules.

## Continue the Attention series

- [Different Types of Attention in Deep Learning]({{ site.baseurl }}/2026/09/21/different-types-of-attention.html)
- [Vision Transformers: From Image Patches to General-Purpose Encoders]({{ site.baseurl }}/2026/09/19/vision-transformers.html)
- [What Does a Vision Transformer Actually Output?]({{ site.baseurl }}/2026/09/19/vit-patch-embeddings-cls-token-explained.html)