---
layout: post
title: "Decodability Is Not Disentanglement: Testing for Factorized Representations"
date: 2026-09-17
description: "Why a linear probe can never prove disentanglement, how factor-swapping experiments expose entangled representations, and a rigorous test matrix for compositional generalization in vision-language models."
tags: [representation-learning, interpretability, disentanglement, probing, vlm, research-methods]
math: true
reading_time: "10 min read"
---

## Decodability Is Not Disentanglement: Testing for Factorized Representations

*10 min read*

Imagine we train a vision-language model on images of colored shapes, and a linear probe can decode both **color** and **shape** from its internal representation with high accuracy. Three natural questions follow:

- **Q1.** Does this prove the model has separate, disentangled representations for color and shape?
- **Q2.** If not, what experiment *would* distinguish "disentangled" from "entangled but linearly separable"?
- **Q3.** How would we test whether the model generalizes to new combinations of color and shape it has never seen together?

A good answer to Q1 is **no** — a linear probe alone cannot establish disentanglement. Q3 is best answered by testing generalization to unseen combinations. The missing piece for Q2 is an important research technique: **factorized testing**. Let's work through all three in detail.

**Related Posts:**
- [What Is a Linear Probe? A Practical Guide to Probing Classifiers]({{ site.baseurl }}{% post_url 2026-09-17-what-is-a-linear-probe %})

---

## 1. Why linear probing does not prove disentanglement

If you're not sure exactly what a linear probe is or why probes are kept linear on purpose, see [What Is a Linear Probe?]({{ site.baseurl }}{% post_url 2026-09-17-what-is-a-linear-probe %}) for the full setup. In short: it's a simple linear classifier trained on a frozen representation $h$ to predict a label $y$ via $\hat{y} = \text{softmax}(Wh + b)$, and it tells us only whether $y$ is **linearly decodable** from $h$.

The intuitive answer to Q1 is:


> "The model doesn't necessarily have separate, disentangled representations for color and shape just because a linear probe can decode both."

Exactly.

A linear probe only establishes:

> Color and shape information can be recovered from the representation.

It does **not** establish that the representation contains independent variables such as:

```
color = red
shape = circle
```

The model might instead encode combinations:

```
red-circle
blue-circle
red-square
blue-square
```

All four categories could be linearly separable even if color and shape are completely entangled.

### Key distinction

| Claim | Does a linear probe establish it? |
|---|---|
| Color information is present | Yes, approximately |
| Shape information is present | Yes, approximately |
| Color and shape are independently encoded | No |
| The model can recombine color and shape compositionally | No |
| The model uses these features causally | No |

This distinction is worth remembering:

> **Decodability is not disentanglement.**

---

## 2. The missing experiment: hold one factor constant, vary the other

To distinguish the two hypotheses, we should test whether the representation supports **factor swapping**.

Suppose we have:

- red circle
- blue circle
- red square
- blue square

Now construct controlled pairs:

```
Red circle → Blue circle
```

Only color changes.

Then:

```
Red circle → Red square
```

Only shape changes.

If color and shape are independently represented, changing color should produce a relatively consistent change in representation, regardless of shape.

Mathematically, let $h(c, s)$ be the representation of color $c$ and shape $s$.

A factorized representation might approximately behave like:

$$h(c, s) \approx h_{\text{color}}(c) + h_{\text{shape}}(s)$$

Then the color difference should be approximately stable:

$$h(\text{red}, s) - h(\text{blue}, s)$$

for different shapes $s$.

For example:

$$h(\text{red circle}) - h(\text{blue circle})$$

should resemble:

$$h(\text{red square}) - h(\text{blue square})$$

Likewise, the shape difference should be relatively stable across colors.

We don't need to assume the model literally uses addition. This is simply a testable signature of factorization.

---

## 3. Experiment design

We could measure:

### Color consistency

Compare:

$$\Delta_{\text{color,circle}} = h(\text{red circle}) - h(\text{blue circle})$$

with:

$$\Delta_{\text{color,square}} = h(\text{red square}) - h(\text{blue square})$$

If these differences are similar, that supports the idea that color is represented in a reusable way.

### Shape consistency

Compare:

$$\Delta_{\text{shape,red}} = h(\text{red circle}) - h(\text{red square})$$

with:

$$\Delta_{\text{shape,blue}} = h(\text{blue circle}) - h(\text{blue square})$$

Again, similar differences would support factorized structure.

This is much stronger than simply training four-way classifiers.

---

## 4. Answering Q3: two kinds of generalization, not one

A natural first instinct for Q3 is to test:

> Different combinations of known and unknown colors and shapes.

That's the right instinct. But let's organize it carefully.

Suppose training contains:

| Color | Shape |
|---|---|
| Red | Circle |
| Blue | Circle |
| Red | Square |
| Blue | Square |

Now test:

### Case A — Known factors, new combination

Training:

- red circle
- blue square

Testing:

- red square
- blue circle

The individual color and shape are known, but their combination is new.

This tests **compositional generalization**.

If the model succeeds, it suggests it can recombine known factors rather than merely memorize complete combinations.

### Case B — New factor, known other factor

Training:

- red circle
- blue circle
- red square
- blue square

Testing:

- green circle
- green square

This tests whether the model can generalize to a new color.

But this is a harder and somewhat different question. It tests whether the representation captures a meaningful color structure, not just whether it can recombine familiar categories.

### Case C — New factor combination

Testing:

- green triangle

This combines:

- an unseen color
- an unseen shape
- an unseen combination

Failure here would not necessarily disprove disentanglement. It may simply be outside the model's learned visual vocabulary or training distribution.

So we should avoid interpreting all failures as the same kind of failure.

---

## 5. A more rigorous test matrix

| Test | What it investigates |
|---|---|
| Known color + known shape, new combination | Compositionality |
| Known color + unseen shape | Shape generalization |
| Unseen color + known shape | Color generalization |
| Unseen color + unseen shape | Extrapolation |
| Same object, only color changed | Color sensitivity |
| Same object, only shape changed | Shape sensitivity |
| Color and shape independently swapped | Factorization |
| Representation intervention on color direction | Causal role of color representation |

This is the beginning of **representation science**: designing experiments that distinguish competing explanations.

---

## 6. One important correction to the hypothetical

Our initial example used only four combinations:

- red circle
- blue circle
- red square
- blue square

That is actually too small to establish much.

A stronger experiment would include many combinations:

- multiple colors
- multiple shapes
- different sizes
- rotations
- textures
- backgrounds
- object positions
- lighting conditions

Otherwise, a model might exploit shortcuts such as:

- color correlated with background
- shape correlated with position
- one object type always appearing larger
- one combination having a distinctive texture

So we should use **controlled synthetic data first**, where we know exactly which factor changed, and then validate on natural images.

This is a general research principle:

> **Use controlled data to test mechanisms; use natural data to test realism and transfer.**

---

## Next question

Suppose a probe can predict **object position** from a VLM's representation with 98% accuracy.

But when we intervene on the supposed "position information," the model's answer to spatial questions does not change.

**What are three possible explanations for this?**
