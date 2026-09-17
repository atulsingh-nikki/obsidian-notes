---
layout: post
title: "Representation Intervention: Proving a Direction Is Causal, Not Just Correlational"
date: 2026-09-17
description: "How to move from 'this direction predicts color' to 'this direction causes color' — direction arithmetic, activation patching, and the specificity tests needed to rule out entangled side effects."
tags: [representation-learning, interpretability, activation-patching, causality, vlm, research-methods]
math: true
reading_time: "9 min read"
---

## Representation Intervention: Proving a Direction Is Causal, Not Just Correlational

*9 min read*

At the end of the [last post]({{ site.baseurl }}{% post_url 2026-09-17-disentanglement-vs-decodability-factorized-testing %}) we left off with a puzzle: a linear probe predicts **object position** from a VLM's representation with 98% accuracy, yet intervening on that "position information" doesn't change the model's spatial answers at all. Three explanations are possible:

1. The probe found information that is *present but unused* — decodable, but not on the causal path to the behavior we're measuring.
2. The probe found the *right variable, wrong direction* — position is encoded, but not as a single additive vector; the intervention was too crude to move it.
3. The intervention itself was misapplied — wrong layer, wrong token, wrong magnitude, or a direction contaminated by correlated factors.

All three point to the same missing concept: **decoding a factor and controlling a factor are different experiments.** Probing asks "is $y$ recoverable from $h$?" Intervention asks "if I change $h$ in a specific way, does $y$ change as predicted, *and nothing else*?" This post is about that second experiment.

**Related Posts:**
- [What Is a Linear Probe? A Practical Guide to Probing Classifiers]({{ site.baseurl }}{% post_url 2026-09-17-what-is-a-linear-probe %})
- [Decodability Is Not Disentanglement: Testing for Factorized Representations]({{ site.baseurl }}{% post_url 2026-09-17-disentanglement-vs-decodability-factorized-testing %})

> **Intervention** means deliberately modifying an internal representation in a controlled way, then observing whether the model's behavior changes as predicted.

The challenge is that the representation $h$ is usually a high-dimensional vector or tensor, and there is no guaranteed "color slot" sitting inside it waiting to be nudged.

---

## 1. What is $h$?

For a vision encoder, we typically have

$$h \in \mathbb{R}^{N \times D}$$

where:

- $N$ = number of visual tokens or spatial patches;
- $D$ = embedding dimension.

Schematically:

```
h = [visual token 1, visual token 2, ..., visual token N]
```

Each token can carry information about several properties at once:

- color
- shape
- texture
- object identity
- position
- lighting
- background
- semantic category

So $h$ does **not** look like a tidy, hand-labeled vector:

```
h = [color, shape, texture, position]   # NOT how it actually works
```

Instead, the information is distributed across many dimensions and many tokens, and usually mixed together. Any intervention has to work with that reality — there is no isolated "color neuron" to flip.

---

## 2. The simplest intervention: shift a known direction

Suppose we've found a direction $v_{\text{color}}$ in representation space that correlates with color — for instance, by contrasting matched pairs of otherwise-similar images:

$$v_{\text{color}} \approx \mathbb{E}[h(\text{red object})] - \mathbb{E}[h(\text{blue object})]$$

We can then construct an edited representation:

$$h' = h + \alpha \, v_{\text{color}}$$

where $\alpha$ controls the strength of the intervention.

Conceptually:

```
Original representation:
h = representation of a red circle

Intervened representation:
h' = h + color-change direction

Expected behavior:
red circle → blue circle
```

But there's an important caveat:

> **Adding a direction does not prove that the direction represents only color.**

Shifting $h$ along $v_{\text{color}}$ might *also* alter:

- object identity
- brightness
- texture
- background
- the model's confidence
- other attributes correlated with color in the training data

So finding a direction that moves color is only step one. Step two — proving the intervention is *specific* — is the harder and more important part.

---

## 3. How do we find a color direction?

There are a few approaches, of increasing rigor.

### Approach A — Difference of paired examples

Use matched images that differ only in color:

```
red circle
blue circle
```

and compute

$$\Delta_{\text{color}} = h(\text{red circle}) - h(\text{blue circle})$$

Repeat across several shapes:

```
red circle   ↔ blue circle
red square   ↔ blue square
red triangle ↔ blue triangle
```

If the resulting difference vectors are similar —

$$\Delta_{\text{red-blue, circle}} \approx \Delta_{\text{red-blue, square}}$$

— that supports the idea that the representation contains a *reusable* color-related direction, not just an accident of one particular red-circle-vs-blue-circle pair. This is the same factor-swapping logic from the [disentanglement post]({{ site.baseurl }}{% post_url 2026-09-17-disentanglement-vs-decodability-factorized-testing %}), applied here to construct the intervention vector rather than just to test for factorization.

### Approach B — Train a linear probe

Train a probe $\hat{c} = Wh + b$ and use a row of $W$ as the candidate direction.

This is convenient, but recall from the [linear probe post]({{ site.baseurl }}{% post_url 2026-09-17-what-is-a-linear-probe %}):

> The probe's weight vector is a direction useful for *prediction*, not necessarily a pure causal color direction.

The probe is free to exploit whatever correlations minimize its loss — shape, identity, texture — so a direction derived purely from probe weights inherits all of probing's usual caveats before we've even gotten to the intervention.

### Approach C — Activation patching

This is especially useful for VLMs, and it sidesteps the need to hand-construct a direction at all.

Suppose we have two images:

- Image A: red circle
- Image B: blue circle

Run both through the model and record internal activations $h_A$ and $h_B$. Then, instead of adding a vector, we **replace** a selected activation from one run with the corresponding activation from the other:

$$h_{\text{patched}} = [h_A^{(1)}, \ldots, h_B^{(k)}, \ldots, h_A^{(L)}]$$

i.e., everything comes from the red-circle run except layer/token $k$, which is swapped in from the blue-circle run. The question becomes:

> If we insert the activation associated with "blue" into the red-image computation, does the model's answer change toward blue?

This is called **activation patching** (or **representation patching**). Unlike Approaches A and B, it doesn't require us to believe a single linear direction exists — we're transplanting a whole activation, whatever structure it has.

The difficult part is choosing:

- which layer;
- which token;
- which spatial region;
- which activation;
- whether the patched activation is even semantically aligned with the target position in the other run (patching token 5 of image A with token 5 of image B only makes sense if the two tokens correspond to the same spatial region or role).

---

## 4. What does "keeping everything else unchanged" really mean?

Strictly speaking, we usually can't guarantee that. Instead, we approximate it through controlled counterfactuals.

For example:

```
Original:                red circle
Intervention target:     color
Expected counterfactual: blue circle
Should stay unchanged:   shape, position, background, object identity
```

We then check whether the model's behavior changes *only* in color-related ways. A useful way to organize this is an intervention matrix:

| Intervention | Expected outcome |
|---|---|
| Red → blue, shape unchanged | Color answer changes |
| Red → blue, shape unchanged | Shape answer stays the same |
| Red → blue, object identity unchanged | Object identity answer stays the same |
| Red → blue, spatial position unchanged | Position answer stays the same |

If changing the supposed color direction *also* changes the predicted shape, one of two things is true: the direction is entangled with shape, or the intervention was applied at the wrong layer/token and dragged shape-relevant information along with it. Either way, the result doesn't support a clean causal story about color.

---

## 5. A more rigorous causal test

Suppose the model answers "the object is a red circle." We intervene on the (candidate) color direction and measure the intended effect:

$$\Delta_{\text{color}} = P(\text{blue} \mid h') - P(\text{blue} \mid h)$$

and, critically, the unintended side effects:

$$\Delta_{\text{shape}} = P(\text{circle} \mid h') - P(\text{circle} \mid h)$$

(and analogously $\Delta_{\text{identity}}$, $\Delta_{\text{position}}$ for whatever else should plausibly be left alone).

A desirable intervention shows

$$|\Delta_{\text{color}}| \text{ large}$$

while

$$|\Delta_{\text{shape}}|, \; |\Delta_{\text{identity}}|, \; |\Delta_{\text{position}}| \text{ small}$$

This doesn't establish perfect causal isolation — no single experiment does — but it provides evidence that the intervention is *selectively* affecting color-related behavior rather than perturbing the representation wholesale.

---

## 6. Where this fits in the bigger picture

Intervention is the third and hardest rung on the ladder we've been climbing across this series:

1. **Is the information present at all?** → linear probing ([post]({{ site.baseurl }}{% post_url 2026-09-17-what-is-a-linear-probe %}))
2. **Is it independently/factorially represented?** → factor-swapping and compositional generalization tests ([post]({{ site.baseurl }}{% post_url 2026-09-17-disentanglement-vs-decodability-factorized-testing %}))
3. **Is it causally used by the model?** → representation intervention / activation patching (this post)

Each rung can fail where the previous one succeeded. A probe can decode position at 98% accuracy (rung 1) while the direction found is entangled with three other factors (rung 2 would catch this), or while the "position information" is real but simply not on the causal path the model uses to answer spatial questions (only rung 3 — intervention — can catch that). That's exactly the puzzle we opened with: high probe accuracy plus a null intervention result is not a contradiction, it's a sign that decodability and causal use are genuinely different properties of a representation, and each needs its own experiment to establish.
