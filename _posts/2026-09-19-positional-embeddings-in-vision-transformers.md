---
layout: post
title: "Positional Embeddings in ViT: Why Add Them, and How the Shapes Actually Match"
date: 2026-09-19
tags: [vision-transformers, vit, attention, computer-vision, representation-learning]
description: "Why Vision Transformers need positional embeddings, what happens if you skip them, and the exact mechanics of how they're combined with patch embeddings — addition, not concatenation or projection — plus what happens when patch counts don't match."
---

### TL;DR

Self-attention has no built-in sense of order — permute the input tokens and the output just gets permuted the same way. A Vision Transformer's patch embeddings, on their own, carry no information about *where* in the image a patch came from. Positional embeddings fix that. The mechanism is simpler than it sounds: the position embedding table is built with the exact same shape as the patch embeddings, so combining them is just element-wise **addition** — no concatenation, no learned projection at that step. The only place shapes genuinely stop matching is when the image resolution changes between pretraining and fine-tuning, and ViT handles that with a **resize**, not a redesign of the combination step.

Full architecture background: [What Does a Vision Transformer Actually Output?]({{ site.baseurl }}{% post_url 2026-09-19-vit-patch-embeddings-cls-token-explained %}).

## Table of Contents

  - [TL;DR](#tldr)
  - [Why Add Position Information At All](#why-add-position-information-at-all)
  - [What Happens If You Don't](#what-happens-if-you-dont)
  - [The Mechanics: Addition, Not Concatenation](#the-mechanics-addition-not-concatenation)
  - [So Where Do Concatenation and Projection Actually Show Up?](#so-where-do-concatenation-and-projection-actually-show-up)
  - [When the Sizes Really Don't Match: Changing Resolution](#when-the-sizes-really-dont-match-changing-resolution)
  - [FAQs](#faqs)
  - [References](#references)

---

### Why Add Position Information At All
A Transformer's self-attention layer computes, for every token, a weighted sum over every other token's value vector, where the weights come from query-key similarity. Nothing about that computation depends on token order — shuffle the input sequence and you get the same set of outputs, just shuffled back the same way. That's fine for a bag of words, but an image is emphatically not a bag of patches: a sky patch at the top means something different from the same-looking patch at the bottom.

Patch embeddings alone don't encode this. Each one is just a linear projection of that patch's pixels — it has no idea it's patch #1 versus patch #150. Positional embeddings are how ViT reintroduces "where," by adding a learned vector to each patch embedding that depends only on that patch's position in the grid.

---

### What Happens If You Don't
The ViT paper ran exactly this ablation (Appendix D.4, ViT-B/16, ImageNet 5-shot linear evaluation):

| Scheme | Accuracy |
|---|---|
| No positional embedding | 61.4% |
| 1-D learned (the default used everywhere else in the paper) | 64.2% |
| 2-D learned | 64.0% |
| Relative | 64.0% |

Two things stand out. First, dropping position embeddings entirely costs about 3 points — a real but not catastrophic drop; the model still trains and still gets fairly far, likely by leaning on whatever weak positional cues leak in elsewhere. Second, and more surprising: once you add *any* positional scheme, which one you pick barely matters. The paper's explanation is that ViT operates on a coarse patch grid (14×14 for a 224px image, 16px patches) rather than raw pixels — at that resolution, the exact encoding scheme matters less because the model can learn whatever 2D structure it needs from data either way. They confirm this by showing that plain 1-D learned embeddings, with no built-in 2D bias at all, end up organizing themselves into a 2D grid on their own: nearby patches converge to similar embeddings, and row/column structure emerges purely from training.

---

### The Mechanics: Addition, Not Concatenation
This is the actual equation from the paper:

$$
z_0 = [x_{\text{class}};\ x_p^1E;\ x_p^2E;\ \cdots;\ x_p^N E] + E_{pos}
$$

where $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ projects flattened patches into the model's $D$-dimensional working space, and $E_{pos} \in \mathbb{R}^{(N+1) \times D}$ is the position embedding table (one row per patch, plus one for the `[CLS]` token).

The combination is **plain element-wise addition** — the same $D$-dimensional patch/`[CLS]` embeddings plus the same-shaped position embeddings, added directly. There's no concatenation step here and no projection layer at the point of combination.

The reason this works without any shape reconciliation is that $E_{pos}$ is **defined** with shape $(N+1) \times D$ from the start — matching the patch embeddings by construction, not by any adaptive step at combination time.

---

### So Where Do Concatenation and Projection Actually Show Up?
Both appear in this pipeline — just not at the point where patch and position embeddings meet.

- **Projection** happens earlier, turning each patch's raw flattened pixels ($P^2 \cdot C$ dimensions, e.g. 768 for a 16×16 RGB patch) into the model's working dimension $D$ (e.g. 768 for ViT-Base, but not necessarily the same number). That's the learned matrix $E$ in the equation above — a genuine linear projection, resolving the one real dimensionality mismatch in the pipeline (raw pixel count vs. model width).
- **Concatenation** shows up inside one specific *variant* the paper ablated: the **2-D learned** position embedding. Instead of one $D$-dimensional table, this scheme learns two half-width tables — an $X$-embedding and a $Y$-embedding, each of size $D/2$ — and for each patch, concatenates its $X$ and $Y$ vectors to build the final $D$-dimensional position embedding. Only after that concatenation does the result get added to the patch embedding, the same way as always.

So: projection solves "raw pixels → model width," concatenation (when used) solves "build one position vector out of two half-size axis vectors," and addition is always the final step that merges position information into the patch/`[CLS]` embeddings.

---

### When the Sizes Really Don't Match: Changing Resolution
There's exactly one place in this whole pipeline where the shapes genuinely stop lining up: fine-tuning or running inference at a different image resolution than pretraining. Patch size stays fixed, so a larger image just means more patches — a longer sequence, a bigger $N$. The pretrained position embedding table, however, has a fixed number of rows, built for the old $N_{\text{old}}+1$. It no longer matches the new $N_{\text{new}}+1$.

ViT's fix is neither addition, concatenation, nor projection — it's a **resize**. The pretrained 1-D sequence of position embeddings is reshaped back into the 2-D grid it originally corresponds to, that grid is resized with 2D (bicubic) interpolation to the new number of patches, and the result is flattened back into a sequence. The paper is explicit that this — along with the initial patch-cutting step — is the *only* place any 2D spatial structure about the image is manually built into the architecture; everywhere else, the model has to learn spatial relationships entirely from data.

---

### FAQs
**Could you use concatenation instead of addition for the main patch + position combination?**
Architecturally yes, but it isn't what ViT does, and it would come at a cost: concatenating a position vector alongside a $D$-dim patch vector either shrinks how much of $D$ is left for patch content (if you keep the total width fixed) or grows the model's working dimension (if you don't). Addition keeps the patch embedding's full capacity intact and lets the position signal live in the same space, which is simpler and is what the ablations were run against.

**Does the position embedding know about 2D image structure by default?**
Not inherently — the default scheme is a **1-D** learned table, treating patches as a flat sequence in raster order with no built-in row/column awareness. Any 2D structure it ends up representing (Figure 7 in the paper) is learned from training data, not designed in.

**Why not just always use 2-D or relative position embeddings, since they sound more principled?**
Because empirically they don't help — all three schemes land within a point of each other (64.0–64.2%) once you have *some* positional signal. The gap that matters is having positional information at all (61.4% → ~64%), not which specific flavor you use.

**Does the resolution-change interpolation trick only matter for very large images?**
It matters any time inference or fine-tuning resolution differs from pretraining resolution, which is actually common practice — the paper itself fine-tunes at higher resolution than it pretrains, since that's known to improve accuracy.

---

### References
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929, 2020 (Section 3.1, Appendix D.4) — [Research/2020/Vision Transformer (ViT) An Image is Worth 16×16 Words.md]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html)
- Companion post: [What Does a Vision Transformer Actually Output? Patches, Positions, and the [CLS] Token]({{ site.baseurl }}{% post_url 2026-09-19-vit-patch-embeddings-cls-token-explained %})
