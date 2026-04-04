---
layout: post
title: "Beta Distribution: A Short Primer"
date: 2026-04-03
description: "Definition, shape parameters, mean and variance, what conjugacy means (and why it matters), Beta–Binomial posterior updates, and common use cases for the Beta distribution on the unit interval."
tags: [beta-distribution, probability, statistics, bayesian-inference, binomial, machine-learning]
math: true
reading_time: "11 min read"
---

## Beta Distribution: A Short Primer

*11 min read*

The Beta family is the standard way to model uncertainty for **quantities in $(0,1)$**—probabilities, proportions, and normalized scores. This post gathers the core definitions and updates in one place. For a lighter read on *why* a two-parameter Beta feels like a **compressed** summary of belief—and how that mindset connects to modern compression narratives—see the companion note [Compressed Uncertainty: Beta Intuition (no formula sheet)]({{ site.baseurl }}{% post_url 2026-04-04-beta-distribution-intuition-compressed-parameterization %}).

**Related Posts:**
- [Bayesian Inference: A Short Primer]({{ site.baseurl }}{% post_url 2026-04-05-bayesian-inference-short-primer %}) - Prior, likelihood, posterior, and when inference is exact vs approximate
- [Expected Value: Mathematical Foundations]({{ site.baseurl }}{% post_url 2026-01-01-expected-value-expectation-mathematical-foundations %})
- [Random vs Stochastic: Foundations]({{ site.baseurl }}{% post_url 2025-03-05-random-vs-stochastic-foundations %})

---

## Density and support

A random variable $X$ has a **Beta $(\alpha,\beta)$** distribution with parameters $\alpha > 0$, $\beta > 0$ if its density on $(0,1)$ is

$$
f_X(x) = \frac{1}{B(\alpha,\beta)}\, x^{\alpha - 1}(1 - x)^{\beta - 1}, \quad x \in (0,1),
$$

where $B(\alpha,\beta)$ is the Beta function that normalizes the integral to $1$.

---

## Interpreting $\alpha$ and $\beta$

The exponents attach **affinity** toward $1$ and toward $0$:

- **$\alpha = \beta = 1$** gives the **uniform** distribution on $(0,1)$.
- **$\alpha > \beta$** shifts mass toward **larger** $x$; **$\alpha < \beta$** toward **smaller** $x$.
- **Large $\alpha + \beta$** with fixed ratio $\alpha / (\alpha + \beta)$ makes the density **concentrated**—strong evidence about the proportion.
- **Small $\alpha + \beta$** yields a **diffuse** prior or belief.

In Bayesian language with Binomial data, it is common to read $\alpha - 1$ and $\beta - 1$ as **pseudocounts** (effective prior successes and failures).

---

## Mean and variance

$$
\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}, \qquad
\operatorname{Var}(X) = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}.
$$

Fixing the mean $\alpha / (\alpha + \beta)$ and increasing $\alpha + \beta$ **reduces variance**—same location, tighter uncertainty.

---

## What is conjugacy?

In Bayesian inference you specify a **prior** $p(\theta)$ over parameters, a **likelihood** $p(\text{data} \mid \theta)$ for how data arise given $\theta$, and obtain the **posterior** using Bayes’ rule:

$$
p(\theta \mid \text{data}) \;\propto\; p(\text{data} \mid \theta)\, p(\theta).
$$

Often $p(\theta \mid \text{data})$ is not a density you can write down in closed form—you integrate, sample (MCMC), or approximate. **Conjugacy** is a fortunate special case:

> A **conjugate prior** for a given likelihood is a family of priors such that, after observing data from that likelihood, the **posterior** still belongs to the **same family** as the prior—only the **parameters** change.

Example pattern: **prior** $\mathrm{Beta}(\alpha,\beta)$, observe Binomial counts, **posterior** $\mathrm{Beta}(\alpha',\beta')$. Same distribution *type*, updated $(\alpha,\beta) \to (\alpha',\beta')$.

The open book [*An Introduction to Bayesian Thinking*](https://statswithr.github.io/book/bayesian-inference.html) states the conjugacy idea in the same words used above and connects it to **avoiding integrals** when the prior–likelihood pair matches: see [§2.1.3 Conjugacy](https://statswithr.github.io/book/bayesian-inference.html) and [§2.2 Three conjugate families](https://statswithr.github.io/book/bayesian-inference.html#three-conjugate-families) (Beta–Binomial is their first family).

**Why conjugacy matters (and why we show it here):**

1. **Closed-form updates** — No search or sampling is required to normalize the posterior for this model; the math has already been done once, forever.
2. **Interpretable bookkeeping** — New evidence becomes **simple operations on parameters** (here: add successes and failures to pseudo-counts). That is exactly what makes the Beta feel like a small *state vector* paired with Binomial data.
3. **Pedagogy** — The Beta is not pulled out of thin air: it is **the** natural prior for a Binomial rate because of this conjugacy. Showing the update makes that partnership obvious.

Conjugacy is **not** a universal requirement for good modeling; many real models use non-conjugate priors and approximate inference. But for teaching the Beta—and for any workflow that still looks like “counts of successes”—the Beta–Binomial conjugate pair is the canonical reference.

---

## Beta prior, Binomial likelihood (conjugate update)

If

$$
p \sim \mathrm{Beta}(\alpha,\beta), \qquad (k \mid p) \sim \mathrm{Binomial}(n,p),
$$

then the posterior is

$$
(p \mid k) \sim \mathrm{Beta}(\alpha + k,\, \beta + n - k).
$$

So **all** Binomial evidence is absorbed by **adding** counts to $\alpha$ and $\beta$. That closed-form update is why the Beta is the workhorse for Bernoulli/Binomial inference—and is the concrete instance of conjugacy described above.

### Example

Prior $p \sim \mathrm{Beta}(2,2)$. Data: $k = 7$ successes in $n = 10$ trials. Then

$$
(p \mid \text{data}) \sim \mathrm{Beta}(9, 5), \qquad \mathbb{E}[p \mid \text{data}] = \frac{9}{14} \approx 0.64.
$$

---

## Where the Beta shows up

- **A/B testing**, click or conversion rates, reliability per trial
- **Thompson sampling** and bandits with Bernoulli rewards
- **Priors or variational factors** on weights or gates constrained to $(0,1)$
- **Bridge to the simplex**: the **Dirichlet** is the multicategory analog of the Beta

---

## Caveat

The Beta is ideal when the generative story is **Binomial-like** (counts of successes). For other data (heavy tails, covariates without a clear link to a single $p$, time-changing rates), richer models (hierarchical, Gaussian processes, etc.) replace or extend this baseline.
