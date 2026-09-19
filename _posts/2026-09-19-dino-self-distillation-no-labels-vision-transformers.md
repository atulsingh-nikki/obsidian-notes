---
layout: post
title: "DINO: Self-Distillation With No Labels for Vision Transformers"
date: 2026-09-19
tags: [self-supervised-learning, vision-transformers, representation-learning, computer-vision]
description: "How DINO trains a Vision Transformer with no labels and no negatives, and why that produces object segmentation for free in the attention maps."
---

### TL;DR

## Table of Contents

  - [TL;DR](#tldr)
  - [Motivation](#motivation)
  - [The Core Idea: Distillation With No Teacher](#the-core-idea-distillation-with-no-teacher)
  - [How Collapse Is Avoided: Centering + Sharpening](#how-collapse-is-avoided-centering--sharpening)
  - [Multi-Crop: Local-to-Global Correspondence](#multi-crop-local-to-global-correspondence)
  - [The Emergent Property: Segmentation for Free](#the-emergent-property-segmentation-for-free)
  - [Results (Highlights)](#results-highlights)
  - [What Actually Matters: The Ablations](#what-actually-matters-the-ablations)
  - [Why This Mattered](#why-this-mattered)
  - [FAQs](#faqs)
  - [References](#references)

DINO (**self-di**stillation with **no** labels) trains a Vision Transformer by having a "student" network match the output of a "teacher" network built purely from an exponential moving average of the student's own weights — no labels, no negatives, no contrastive loss, no clustering. The surprising payoff isn't just a strong classifier: the ViT's own attention maps spontaneously learn to outline objects, something that doesn't happen with supervised ViTs or with convnets. A small ViT trained this way reaches 78.3% top-1 on ImageNet with a plain k-NN classifier on frozen features, and a base ViT with small (8×8) patches hits 80.1% top-1 in linear evaluation.

Reference: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin — Facebook AI Research / Inria, ICCV 2021)

Full research note with method details, all benchmark tables, and ablations: [DINO — Emerging Properties in Self-Supervised Vision Transformers]({{ site.baseurl }}/Research/2021/DINO%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers.html).

---

### Motivation
- Vision Transformers (ViT) had matched convnets on accuracy but hadn't shown any *unique* benefit — they're more data-hungry and compute-hungry, with no special properties to show for it.
- In NLP, the big win of Transformers came from self-supervised pretraining (BERT's masking, GPT's language modeling), not architecture alone.
- The paper's question: does the *muted* success of ViTs in vision come from training them with plain supervised labels instead of a richer self-supervised signal?

---

### The Core Idea: Distillation With No Teacher
Standard knowledge distillation trains a student network $g_{\theta_s}$ to match a fixed, pretrained teacher $g_{\theta_t}$. DINO borrows the mechanics but removes the requirement of having a teacher at all:

- **Same architecture, two sets of weights.** Student and teacher are identical ViTs (backbone + 3-layer MLP projection head), differing only in their parameters.
- **The teacher is the student's own EMA.** There's no separately trained teacher — its weights are updated every step as $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$, with $\lambda$ following a cosine schedule from 0.996 to 1. This is exactly a momentum encoder, the same idea used in MoCo and BYOL.
- **Match distributions, not pixels or embeddings directly.** Each network outputs a $K$-dimensional feature, normalized into a probability distribution with a temperature softmax:

$$
P_s(x)^{(i)} = \frac{\exp(g_{\theta_s}(x)^{(i)}/\tau_s)}{\sum_{k=1}^{K}\exp(g_{\theta_s}(x)^{(k)}/\tau_s)}
$$

  The student is trained to minimize the cross-entropy $H(P_t(x), P_s(x)) = -P_t(x)\log P_s(x)$ against the teacher's distribution, with a stop-gradient on the teacher so only the student receives gradients.

The surprising empirical finding: the teacher, despite being nothing but an average of past students, **consistently outperforms the student throughout training** — a form of Polyak-Ruppert averaging (model ensembling via exponential decay) that keeps handing the student a slightly-better target to chase, so both keep improving together. For the full story on why this works (and why a raw copy of the student collapses instead), see [The Momentum Encoder: Why EMA Teachers Work in Self-Supervised Learning]({{ site.baseurl }}{% post_url 2026-09-19-momentum-encoder-why-ema-teachers-work %}).

---

### How Collapse Is Avoided: Centering + Sharpening
Every non-contrastive SSL method needs *some* trick to stop the network from collapsing to a trivial constant output. DINO's answer is unusually simple — no negative pairs, no clustering constraint, no extra predictor network:

- **Centering**: subtract a running mean $c$ from the teacher's logits before the softmax, updated as $c \leftarrow mc + (1-m)\frac{1}{B}\sum_i g_{\theta_t}(x_i)$. This stops any single dimension from dominating, but on its own it collapses the output toward a **uniform** distribution.
- **Sharpening**: use a low temperature $\tau_t$ in the teacher's softmax, which pushes the opposite way, toward a **peaked** distribution.

Applied together, these two failure modes cancel out and are sufficient to keep training stable — dropping either one collapses the model to near-0% accuracy in the ablations.

---

### Multi-Crop: Local-to-Global Correspondence
DINO generates several distorted "views" of each image: two **global** crops (224², covering more than half the image) and multiple **local** crops (96², covering less). All crops go through the student, but only the global crops go through the teacher — forcing the student to predict a global-context target from what might only be a small local patch. This "local-to-global" correspondence is one of the two components (along with the momentum encoder) that the ablations show is not optional.

---

### The Emergent Property: Segmentation for Free
This is the headline result. If you look at the self-attention of the `[CLS]` token in the last Transformer block — a token that is never attached to any label — different attention heads spontaneously attend to different objects or object parts in the image, with clean boundaries around them.

Quantifying this: thresholding the attention map to keep 60% of the attention mass and comparing against ground-truth masks on PASCAL VOC12 gives a Jaccard similarity of **45.9** for a DINO-trained ViT-S/8, versus **27.3** for the *same architecture* trained with plain supervised labels. Same model, same patch size, same data — the only difference is the training objective, and it's the difference between "no real segmentation signal" and "usable object masks." Supervised ViTs and convnets don't show this property nearly as clearly.

---

### Results (Highlights)
All numbers are top-1 accuracy on ImageNet-1k unless noted.

- **Same architecture, ViT-S/16 (21M params):** DINO reaches 77.0% (linear probe) / 74.5% (k-NN) — beating BYOL, MoCo-v2, and SwAV on the identical backbone by **+3.5%** linear and **+7.9%** k-NN. With a ResNet-50 instead, DINO is roughly on par with SwAV/BYOL (75.3% / 67.5%) — the outsized gain is specific to ViT.
- **k-NN nearly matches linear probing**, only on ViT: 74.5% vs. 77.0%. A plain weighted 20-nearest-neighbor classifier on frozen features — no training, no augmentation — gets within 2.5 points of a fully trained linear head. This doesn't happen with ResNet-50 or with other SSL methods on ViT.
- **Best full result:** ViT-B/8 (8×8 patches) reaches **80.1%** linear / **77.4%** k-NN, beating the prior convnet state of the art with 10× fewer parameters and 1.4× faster inference.
- **Small-compute budget:** ViT-S/16 trained on just two 8-GPU machines for 3 days reaches 76.1% linear — already ahead of comparably-sized convnet SSL systems trained with far more compute.
- **Zero-shot video segmentation (DAVIS-2017)**: nearest-neighbor propagation of frozen patch tokens between video frames (no training on video at all) reaches competitive $\mathcal{J\&F}_m$ scores, showing the spatial information survives all the way to the patch-token level.
- **Retrieval and transfer**: DINO features beat supervised ViT features on image retrieval (Oxford/Paris), copy detection, and fine-tuning transfer to other classification datasets.

---

### What Actually Matters: The Ablations
The paper's component ablation (ViT-S/16, 300 epochs) is the most useful part for deciding what to actually keep if you're implementing this:

| Change from default DINO | Effect |
|---|---|
| Remove momentum encoder | Collapses to ~0.1% accuracy — nothing else compensates |
| Remove multi-crop | −3.4% linear accuracy |
| Swap cross-entropy loss for MSE | Drops from 76.1% → 62.4% |
| Add a BYOL-style predictor | Negligible change (unlike BYOL, where it's required) |
| Smaller patch size (16→8→5) | Consistently better features, at a steep throughput cost |

The takeaway: the momentum encoder and multi-crop are load-bearing; the predictor and heavier normalization tricks used by other frameworks are not needed here.

---

### Why This Mattered
- Extended non-contrastive self-supervised learning (in the lineage of BYOL and SwAV) cleanly to Vision Transformers, with no architecture changes.
- Demonstrated that self-supervision — not just architecture — is what unlocks ViT's distinctive properties.
- Directly inspired follow-on work: iBOT combines DINO's self-distillation with masked image modeling; MAE explores masked autoencoding for ViTs; DINOv2 later scaled this recipe up with a curated large-scale dataset.

---

### FAQs
**Is DINO a contrastive method?**
No. There are no negative pairs and no contrastive loss. It's closer in spirit to BYOL (matching a momentum-teacher's output) but interpreted here explicitly as self-distillation, using a cross-entropy loss on sharpened, centered softmax outputs.

**Why does the teacher outperform the student if it's just an average of past students?**
Averaging weights over time (Polyak-Ruppert averaging) tends to produce a better single model than any individual snapshot along the way — the same reasoning behind EMA/weight-averaging tricks used elsewhere in training. Because the teacher is consistently a bit better, it gives the student a moving target that keeps pulling training forward.

**Does this work with convnets too?**
Yes — DINO applied to ResNet-50 matches state-of-the-art SSL methods for convnets. The distinctive emergent segmentation and near-linear k-NN performance, however, are specific to the ViT + DINO combination.

**What's the practical cost?**
Small patches (8×8) give the best features but are expensive: throughput drops from 1007 im/s (ViT-S/16) to 180 im/s (ViT-S/8) on a V100. A reasonable small-scale recipe (ViT-S/16, 300 epochs, two 8-GPU machines, ~3 days) already beats comparable convnet SSL baselines.

---

### References
- Paper: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (arXiv:2104.14294)
- Official code: [facebookresearch/dino](https://github.com/facebookresearch/dino)
- Full research note: [DINO — Emerging Properties in Self-Supervised Vision Transformers]({{ site.baseurl }}/Research/2021/DINO%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers.html)
