---
layout: post
title: "When There's No Single Direction: Subspaces, Distributed Codes, and a Ladder of Evidence"
date: 2026-09-17
description: "What to do when a color direction doesn't hold up — testing single-direction, subspace, and distributed hypotheses with cosine consistency, PCA, cross-context transfer, and subspace interventions, plus why absence claims need extra care."
tags: [representation-learning, interpretability, activation-patching, causality, vlm, research-methods]
math: true
reading_time: "10 min read"
---

## When There's No Single Direction: Subspaces, Distributed Codes, and a Ladder of Evidence

*10 min read*

The [previous post]({{ site.baseurl }}{% post_url 2026-09-17-representation-intervention-activation-patching %}) assumed we already had a candidate color direction $v_{\text{color}}$ and asked how to test whether shifting along it is *causally specific* to color. That leaves a prior question unanswered: **how do we know a single direction is the right kind of object to be looking for in the first place?**

There are really two separate problems bundled together:

1. How do we determine whether color is represented by a single direction, a subspace, or a distributed pattern?
2. How do we intervene if there is no single clean direction?

**Related Posts:**
- [What Is a Linear Probe? A Practical Guide to Probing Classifiers]({{ site.baseurl }}{% post_url 2026-09-17-what-is-a-linear-probe %})
- [Decodability Is Not Disentanglement: Testing for Factorized Representations]({{ site.baseurl }}{% post_url 2026-09-17-disentanglement-vs-decodability-factorized-testing %})
- [Representation Intervention: Proving a Direction Is Causal, Not Just Correlational]({{ site.baseurl }}{% post_url 2026-09-17-representation-intervention-activation-patching %})

Let's handle them in that order.

---

## 1. Three competing hypotheses about structure

We shouldn't start by assuming a "color direction" exists — that's a hypothesis to test, not a premise. Given $h \in \mathbb{R}^{D}$, there are at least three structures color could plausibly take.

**Hypothesis A — single-direction representation.** Color is approximately encoded along one direction, and changing color mainly changes a scalar coefficient:

$$h \approx h_{\text{other}} + c\, v_{\text{color}}$$

**Hypothesis B — low-dimensional subspace.** Color is represented using several directions at once:

$$h \approx h_{\text{other}} + U_{\text{color}} z$$

where $U_{\text{color}} \in \mathbb{R}^{D \times k}$ for some small $k$ (2, 5, 20 — whatever the data supports), and $z$ holds the color-related coordinates. There is no single color vector here, but there may be a **color subspace**.

**Hypothesis C — distributed, context-dependent representation.** The representation of color depends strongly on everything else in the scene:

$$h = f(\text{color}, \text{shape}, \text{object}, \text{background}, \ldots)$$

The effect of changing red to blue may differ for a circle, a car, a person, or a heavily textured object. In this regime there may be no globally consistent color direction at all — only a family of context-specific ones.

None of these is the default assumption. We need experiments that can distinguish them.

---

## 2. Test 1 — consistency of difference vectors

Take controlled pairs where only color changes:

```
red circle   → blue circle
red square   → blue square
red triangle → blue triangle
```

For each pair, compute

$$\Delta_i = h(x_i^{\text{red}}) - h(x_i^{\text{blue}})$$

If color is represented by a single direction, these vectors should point roughly the same way. We can measure that with cosine similarity:

$$\cos(\Delta_i, \Delta_j) = \frac{\Delta_i^\top \Delta_j}{\|\Delta_i\| \|\Delta_j\|}$$

- **High similarity** across many contexts → evidence for a shared direction (Hypothesis A).
- **Low similarity** → color effects vary with context.
- **Intermediate similarity** → possibly a shared subspace rather than one direction (Hypothesis B).

This test has a real limitation, though: low cosine similarity doesn't prove color is absent or fully distributed. The same color information could still be encoded across several dimensions whose particular combination just happens to shift with context — which is exactly Hypothesis B, not evidence against structure altogether.

---

## 3. Test 2 — PCA or low-rank structure

Collect many color-change vectors $\Delta_1, \Delta_2, \ldots, \Delta_n$ and stack them into a matrix $\Delta \in \mathbb{R}^{n \times D}$. Then run PCA (or an SVD) on it.

If the first principal component explains nearly all the variance,

$$\frac{\lambda_1}{\sum_j \lambda_j} \approx 1$$

that suggests a dominant one-dimensional direction — Hypothesis A looks right. If instead the first few components explain most of the variance,

$$\frac{\sum_{j=1}^{k} \lambda_j}{\sum_j \lambda_j} \approx 1$$

then color may live in a genuinely $k$-dimensional subspace — Hypothesis B. If many components are needed and the structure keeps changing across contexts, that leans toward Hypothesis C.

But a caveat matters here:

> PCA reveals the geometry of *measured changes*. It does not automatically reveal semantic meaning.

A dominant direction could just as easily reflect lighting, brightness, object identity, or a dataset artifact as it could "color itself." PCA tells you about dimensionality, not about what the dimensions mean.

---

## 4. Test 3 — cross-context prediction

This is a stronger test than PCA, because it checks *transfer*, not just geometry.

Estimate a color-change direction using only circles, $v_{\text{color}}^{\text{circle}}$, then apply it somewhere it was never fit:

```
Estimate direction:
red circle → blue circle

Apply to:
red square → predicted blue square?
red triangle → predicted blue triangle?
red car → predicted blue car?
```

If adding this direction to a red square makes the model behave like a blue square, the direction transfers — that's real evidence for a reusable color representation, not an artifact of one context. If it only works for circles, either the representation is genuinely context-dependent, or the estimated direction absorbed circle-specific artifacts along the way. This is a form of **cross-context intervention generalization**, and it's a much harder bar to clear than either of the first two tests.

---

## 5. Test 4 — subspace intervention

If a single direction turns out to be insufficient, the natural next move is to intervene on a *subspace* instead of a vector.

Suppose PCA (or another decomposition) gives candidate basis vectors $U_{\text{color}} = [u_1, u_2, \ldots, u_k]$. The projection onto that subspace is

$$P_{\text{color}} = U_{\text{color}} U_{\text{color}}^\top$$

and the color-related component of a representation is $h_{\text{color}} = P_{\text{color}} h$. A subspace intervention removes the estimated color component and inserts a target one:

$$h' = h - P_{\text{color}} h + P_{\text{color}} h_{\text{target}}$$

Conceptually:

```
Original representation:        red circle
Remove estimated color subspace: object with color info suppressed
Insert target color subspace:    blue circle
```

This is strictly more flexible than nudging one vector — but the subspace can still smuggle in non-color information, so the same specificity checks from the [intervention post]({{ site.baseurl }}{% post_url 2026-09-17-representation-intervention-activation-patching %}) still apply: does the color answer change, does shape stay stable, does object identity stay stable, does spatial information stay stable, and does the effect hold across contexts?

---

## 6. Test 5 — sparse or feature-based decomposition

It's also possible that color isn't a dense direction or a smooth subspace at all, but is distributed across many nonlinear features, e.g.:

```
Feature 1: reddish surface under bright light
Feature 2: red clothing
Feature 3: red circular object
Feature 4: warm-colored background
Feature 5: red object boundary
```

No single feature means "red" on its own — recognition emerges from a combination of features, each of which is itself context-sensitive. Methods like sparse autoencoders try to recover more interpretable latents, $h \approx Dz$, where $z$ is sparse (only a handful of features active per input). If several such features fire for red objects but each one also responds to context, that's evidence for a **feature-distributed** representation rather than one clean semantic axis.

Still, sparse features are hypotheses produced by a decomposition method, not guaranteed ground truth — they need the same transfer and specificity tests as everything else on this list.

---

## 7. What would count as evidence that no single direction exists?

We shouldn't conclude this just because one direction failed to work. A stronger conclusion needs a *pattern* of results:

1. Color-change vectors have low cross-context consistency.
2. A direction learned in one context doesn't transfer to others.
3. PCA needs several components to explain the color-change variance.
4. A one-dimensional intervention has weak or inconsistent behavioral effects.
5. A carefully validated subspace intervention outperforms any single-direction intervention.
6. The results replicate across datasets, layers, and object categories.

Only then can we say something like:

> The tested representation does not appear to support a globally consistent one-dimensional color direction. Color-related information is better modeled as a context-dependent subspace or distributed feature structure.

Notice the wording: **"does not appear to support"** is far more defensible than **"there is no color direction."** Absence claims are hard, because a direction may genuinely exist but stay hidden behind poor stimulus design, too small a sample, the wrong layer, a nonlinear coordinate transform, noisy representations, misaligned tokens, or simply the wrong intervention method. Failing to find something is not the same as showing it isn't there.

---

## 8. A hierarchy of claims

Putting the tests together gives a ladder, each rung licensing a stronger claim than the last:

| Evidence | Supported claim |
|---|---|
| Color can be decoded | Color information is accessible |
| Difference vectors are consistent across contexts | A shared color-related direction may exist |
| One direction transfers across contexts | Evidence for reusable color structure |
| Multiple directions are needed to explain the variance | Color may occupy a subspace |
| Subspace intervention changes color behavior selectively | The subspace is causally relevant to color behavior |
| Intervention transfers across contexts | The representation supports reusable color manipulation |
| Strong causal and cross-layer evidence, replicated | A genuine mechanistic account of color representation |

The deepest point underlying all four posts in this series is this:

> **We do not discover "the true color direction" directly. We construct candidate explanations of the representation and progressively test how well they predict and control behavior.**

That's the essence of mechanistic interpretability: not merely finding patterns, but testing whether those patterns actually explain the model's computation — decoding, then disentangling, then intervening, and, when the simplest intervention fails, widening the hypothesis from a direction to a subspace to a distributed code, and running the same discipline of tests all over again at the new level.
