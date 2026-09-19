---
layout: post
title: "What Does a Vision Transformer Actually Output? Patches, Positions, and the [CLS] Token"
date: 2026-09-19
tags: [vision-transformers, vit, attention, computer-vision, representation-learning]
description: "A plain-language look at how a Vision Transformer turns an image into tokens, why it borrows BERT's [CLS] token, and what you actually get back at the output."
---

### TL;DR

## Table of Contents

  - [TL;DR](#tldr)
  - [Turning an Image Into a Sequence](#turning-an-image-into-a-sequence)
  - [The Extra Token: Why ViT Borrows BERT's `[CLS]`](#the-extra-token-why-vit-borrows-berts-cls)
  - [What Comes Out the Other End](#what-comes-out-the-other-end)
  - [`[CLS]` vs. Patch Tokens, Side by Side](#cls-vs-patch-tokens-side-by-side)
  - [FAQs](#faqs)
  - [References](#references)

A ConvNet ends with a single prediction. A Vision Transformer (ViT) doesn't — it ends with a **sequence of vectors**, and it's up to whatever sits on top to decide which ones to use. That sequence has two kinds of entries: one `[CLS]` vector meant to summarize the whole image, and a grid of patch vectors, each still tied to one spot in the image. Understanding that split explains both plain ViT classification and later methods like DINO, which reuse the same output in very different ways.

Full architecture details: [the ViT research note]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html). This post just walks through the part it assumes you already know.

---

### Turning an Image Into a Sequence
A Transformer expects a sequence of vectors — that's what made it work for text, where the sequence is just word embeddings. ViT's trick is deciding what the "words" of an image should be: fixed-size, non-overlapping **patches**.

Split a 224×224 image into 16×16 patches and you get a 14×14 grid — 196 patches. Flatten each one and pass it through a single learned linear layer, and you get 196 embedding vectors — the same shape as 196 word embeddings feeding a text Transformer.

Two things are added before this sequence goes anywhere:
- **Position embeddings.** Self-attention has no built-in sense of order, so a learned position vector is added to each patch embedding — otherwise the model couldn't tell a patch in the top-left from one in the bottom-right.
- **The `[CLS]` token**, described next.

It's a deliberately simple way to "look" at an image — no convolutions, no multi-scale pyramid, just chop and project. Everything more sophisticated is built by the self-attention layers afterward.

---

### The Extra Token: Why ViT Borrows BERT's `[CLS]`
Before the patch sequence enters the Transformer, ViT adds one more vector: the `[CLS]` (classification) token, borrowed from BERT, where it plays the same role for text — giving you one sentence-level vector out of a sequence of word tokens.

A couple of things worth being precise about:
- It's **not** computed from the patches and it's **not** a pooling operation. It's its own learned vector, with its own learned position embedding, just like any other parameter.
- Inside the Transformer it has no special treatment — it's one more token that can attend to every patch, and be attended to by every patch, exactly like any other token pair. Its role as a "global summary" only emerges because training pushes it that way: in supervised ViT, it's the one token connected to the classification loss, so gradient descent shapes its attention into something useful for that task.

So a 224×224 image with 16×16 patches gives you 196 patch tokens + 1 `[CLS]` token = **197 tokens** total. That's the count you'll see quoted for ViT-S/16 and ViT-B/16 in papers like [DINO]({{ site.baseurl }}/Research/2021/DINO%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers.html).

---

### What Comes Out the Other End
The encoder is just a stack of standard Transformer blocks — attention, then MLP, over and over. It doesn't pool or compress anything: feed in 197 vectors, get out 197 vectors, each one refined by several rounds of attention.

What you do with those 197 output vectors depends entirely on the task:

- **Supervised ViT classification** keeps only the final `[CLS]` vector, feeds it to a small MLP to predict the class, and throws away all 196 patch vectors.
- **DINO's distillation loss** also uses just the `[CLS]` vector — from both the student and the teacher — passed through a projection head to build the distribution used in its cross-entropy loss. Same token, same "global summary" role, different objective on top.
- **DINO's emergent segmentation** doesn't use the `[CLS]` output vector at all. It uses the `[CLS]` token's **attention weights** over the patches in the last layer. Since those weights sum to 1, thresholding them (keep the patches making up 60% of the attention mass) gives you a mask directly — no decoder, no segmentation head, just reading off what the summarizing token was looking at.
- **Dense tasks** — video segmentation, retrieval — use the **patch vectors**, not `[CLS]`, because they need a spatial grid of features rather than one global one. DINO's video-segmentation evaluation, for instance, matches patch tokens directly between frames.

---

### `[CLS]` vs. Patch Tokens, Side by Side

| | `[CLS]` token | Patch tokens |
|---|---|---|
| Count | 1 per image | 196 (for 224px images, 16px patches) |
| Tied to a location? | No — global | Yes — each keeps its patch's spot |
| Where it comes from | A learned vector, not derived from any patch | Linear projection of that patch's pixels |
| Typical use | Classification, DINO's distillation loss | Dense prediction, retrieval, video tracking |
| DINO's segmentation trick | Reads its attention weights | Are the thing being attended to |

---

### FAQs
**Does the `[CLS]` token see the image directly?**
No — only through attention. It never touches pixels; it only mixes in whatever the patch tokens represent at each layer, weighted by learned attention.

**Why not just average the patch tokens instead of using a `[CLS]` token?**
You can — some ViT variants do. The difference is that `[CLS]`'s weighting is *learned*, so it can favor some patches over others, instead of averaging everything equally.

**Does the `[CLS]` token's position embedding mean anything, since it has no real location?**
Not spatially — it still gets a learned position vector, but that's mostly just another parameter that tells the model "this one is different from the patches," not a real coordinate.

---

### References
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929, 2020 — [Research/2020/Vision Transformer (ViT) An Image is Worth 16×16 Words.md]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html)
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv:1810.04805, 2018 — origin of the `[CLS]` token idea.
- Caron et al. "Emerging Properties in Self-Supervised Vision Transformers" (DINO). arXiv:2104.14294, 2021 — [Research/2021/DINO Emerging Properties in Self-Supervised Vision Transformers.md]({{ site.baseurl }}/Research/2021/DINO%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers.html)
- Companion post: [DINO — Self-Distillation With No Labels for Vision Transformers]({{ site.baseurl }}{% post_url 2026-09-19-dino-self-distillation-no-labels-vision-transformers %})
