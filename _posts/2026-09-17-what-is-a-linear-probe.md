---
layout: post
title: "What Is a Linear Probe? A Practical Guide to Probing Classifiers"
date: 2026-09-17
description: "A deep dive into linear probes: the formal setup, why linearity is the point rather than a limitation, how to read probe accuracy correctly, and the common pitfalls (control tasks, probe capacity, and confounded correlations) that make probing results easy to misinterpret."
tags: [representation-learning, interpretability, probing, linear-probe, research-methods]
math: true
reading_time: "7 min read"
---

## What Is a Linear Probe? A Practical Guide to Probing Classifiers

*7 min read*

If you read papers on representation learning or interpretability, you'll constantly run into a claim like: *"a linear probe can decode X from the model's representation with 95% accuracy."* This post explains exactly what that sentence means, why researchers deliberately choose something as simple as a **linear** probe, and where the technique quietly breaks down.

**Related Posts:**
- [Decodability Is Not Disentanglement: Testing for Factorized Representations]({{ site.baseurl }}{% post_url 2026-09-17-disentanglement-vs-decodability-factorized-testing %})
- [Representation Intervention: Proving a Direction Is Causal, Not Just Correlational]({{ site.baseurl }}{% post_url 2026-09-17-representation-intervention-activation-patching %})
- [When There's No Single Direction: Subspaces, Distributed Codes, and a Ladder of Evidence]({{ site.baseurl }}{% post_url 2026-09-17-single-direction-vs-subspace-representations %})

---

## 1. The basic setup

A **linear probe** is a small, deliberately simple classifier trained on top of a *frozen* representation to test whether a particular piece of information is present in that representation.

Concretely:

1. Take a trained model (a vision encoder, a language model, a VLM — anything that produces internal activations).
2. Pick a layer and extract its representation $h \in \mathbb{R}^d$ for each input.
3. Freeze the model. Do not fine-tune it.
4. Train a linear map on top of $h$ to predict a label $y$ you care about (color, part-of-speech, sentiment, object position, etc.):

$$\hat{y} = \text{softmax}(Wh + b), \qquad W \in \mathbb{R}^{k \times d},\ b \in \mathbb{R}^{k}$$

for a $k$-class classification problem, or simply $\hat{y} = w^\top h + b$ for a regression target.

5. Measure the probe's accuracy (or $R^2$, or F1) on held-out data.

That's it. The "model" being tested is the frozen network; the probe is just an instrument for reading it out.

---

## 2. Why *linear*? Isn't that a weak classifier?

This is the most commonly misunderstood part of the technique. The weakness of the probe is the entire point.

If we allowed the probe to be an arbitrarily deep, nonlinear network, then a probe with enough capacity could learn to predict almost *any* label from almost *any* representation — including from raw pixels, or even from random noise, given enough data and training time. A powerful probe doesn't tell you what the frozen model encoded; it tells you what the probe itself is capable of learning.

By restricting the probe to a linear function, we remove most of the probe's own representational power. If a *linear* function can decode $y$, the information must already be present in a **linearly accessible** form inside $h$ — the frozen model did the representational work, not the probe.

So the probe's simplicity is a control on the experiment, not a compromise:

| Probe capacity | What high accuracy would tell you |
|---|---|
| Linear (1 layer, no nonlinearity) | The information is linearly accessible in $h$ |
| MLP (deep, nonlinear) | The information is present *somewhere* in $h$, in some form — but so would almost anything given enough capacity |
| Nearest-neighbor / lookup table | Tells you almost nothing about $h$'s structure — memorization is likely |

---

## 3. What a probe result does and does not establish

This is the crux most people get wrong.

**A high-accuracy linear probe establishes:**

> The label $y$ can be recovered from $h$ using a simple linear function.

**A high-accuracy linear probe does *not* establish:**

- That the model *uses* this information for its downstream behavior (accessibility ≠ causal use).
- That the information is stored in a clean, dedicated subspace separate from other factors (accessibility ≠ disentanglement — see the [companion post]({{ site.baseurl }}{% post_url 2026-09-17-disentanglement-vs-decodability-factorized-testing %}) for a full treatment).
- That the representation would generalize to new combinations of factors it hasn't seen.
- That the correlation the probe found isn't a shortcut or confound in the data (e.g., color correlated with background).

In short:

> **Probing measures information + accessibility. It does not measure use, causality, or structure.**

---

## 4. Control tasks: the sanity check most probing papers skip

A subtle failure mode: even a linear probe has *some* capacity, and with a high-dimensional $h$ and a small number of training examples, it can overfit and appear to "decode" a label that isn't really encoded in any meaningful sense.

Hewitt & Liang (2019) proposed **control tasks**: pair every real probing task with a task of matched difficulty but *random*, meaningless labels (e.g., assign each word a random integer label rather than its true part-of-speech). If the probe fits the control task nearly as well as the real task, the probe's capacity — not the representation's content — is doing the work, and the real result should be discounted.

A well-designed probing experiment reports:

- Real task accuracy
- Control task accuracy
- **Selectivity** = real task accuracy − control task accuracy

High selectivity is what actually supports the claim "this information is meaningfully encoded," not raw accuracy alone.

---

## 5. Practical checklist for running or reading a probing experiment

- **Freeze the base model.** If you fine-tune it while training the probe, you're no longer testing the original representation.
- **Use a held-out test set.** Probe accuracy on training data tells you nothing; probes can memorize.
- **Keep the probe linear (or otherwise minimal).** The whole point is that the probe should be weak enough that success is attributable to the representation.
- **Run a control task.** Compare against random labels of matched difficulty to estimate selectivity.
- **Watch for confounds in the data.** If color correlates with background or position in your dataset, a probe "detecting color" might really be detecting the confound.
- **Don't conflate decodability with disentanglement, causality, or generalization.** Each of those requires a different, additional experiment.

---

## 6. Where probing fits in the bigger picture

Linear probing is a foundational tool in interpretability, but it answers a narrow question: *is this information linearly present?* It's the first rung on a ladder of increasingly demanding questions:

1. **Is the information present at all?** → linear (or nonlinear) probing
2. **Is it independently/factorially represented?** → factor-swapping and compositional generalization tests
3. **Is it causally used by the model?** → representation interventions / activation patching

Probing is cheap, fast, and a good first filter — but treat a strong probe result as the *beginning* of an investigation, not the conclusion of one.
