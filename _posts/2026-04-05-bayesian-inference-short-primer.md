---
layout: post
title: "Bayesian Inference: A Short Primer"
date: 2026-04-05
description: "Ladder of examples from a two-theory coin to deep nets: explicit θ, data D, likelihood p(D|θ), and how Bayesian inference updates belief; then history and when inference is exact vs approximate."
tags: [bayesian-inference, statistics, probability, priors, posteriors, mcmc, machine-learning]
math: true
reading_time: "18 min read"
---

## Bayesian Inference: A Short Primer

*18 min read*

This post is a **standalone** map of Bayesian inference. The heart is a **ladder of examples** ([jump there](#examples-from-small-to-complex-)): each step names **$\theta$**, **$D$**, the **likelihood** $p(D \mid \theta)$, and **how Bayes is used** to get a posterior. For more Beta–Binomial detail see the [Beta primer]({{ site.baseurl }}{% post_url 2026-04-03-beta-distribution-primer %}); for recursive state estimation see [Bayesian foundations of Kalman filtering]({{ site.baseurl }}{% post_url 2024-09-22-bayesian-foundations-kalman %}).

The free online book [*An Introduction to Bayesian Thinking*](https://statswithr.github.io/book/) (companion to Bayesian statistics with R) uses closely matching language. On [Chapter 2 — Bayesian Inference](https://statswithr.github.io/book/bayesian-inference.html), they write:

> Bayes’ rule is a machine to turn one’s **prior beliefs into posterior beliefs**.

On [§2.1.3 Conjugacy](https://statswithr.github.io/book/bayesian-inference.html) they define conjugacy this way:

> Conjugacy occurs when the **posterior distribution** is in the **same family** of probability density functions as the prior belief, but with **new parameter values**, which have been updated to reflect what we have learned from the data.

They organize three workhorse conjugate pairs in [§2.2 Three conjugate families](https://statswithr.github.io/book/bayesian-inference.html#three-conjugate-families): **Beta–Binomial**, **Gamma–Poisson**, and **Normal–Normal** (known variance). **Examples 1–4** below mirror that ladder (with an ML-flavored fifth step). When new batches of data arrive in sequence, the same source stresses: **yesterday’s posterior is today’s prior**—each update stays self-consistent.

**Related Posts:**
- [Beta Distribution: A Short Primer]({{ site.baseurl }}{% post_url 2026-04-03-beta-distribution-primer %}) - Conjugate Beta–Binomial updates
- [Bayesian Foundations of Kalman Filtering]({{ site.baseurl }}{% post_url 2024-09-22-bayesian-foundations-kalman %}) - Sequential Bayes under Gaussian structure
- [The Normalization Constant Problem]({{ site.baseurl }}{% post_url 2025-12-24-normalization-constant-problem %}) - Why $p(\text{data})$ is often hard
- [How Variational Autoencoders Avoid Computing the Partition Function]({{ site.baseurl }}{% post_url 2026-01-01-how-vaes-avoid-computing-partition-function %}) - Variational inference in ML
- [Understanding KL Divergence]({{ site.baseurl }}{% post_url 2025-03-20-understanding-kl-divergence %}) - Common objective in approximate inference

---

## What question does Bayesian inference answer?

You have a **model** with unknown parameters $\theta$ (could be a single probability, a vector of weights, or a latent trajectory). You observe **data** $D$. Bayesian inference returns **a full distribution over $\theta$ after seeing $D$**, not only a single “best” value.

That distribution is the **posterior** $p(\theta \mid D)$. It encodes **remaining uncertainty** about $\theta$ once data are accounted for.

---

## Examples: from small to complex ($\theta$, $D$, likelihood, Bayes)
{: #examples-from-small-to-complex-}

Every example below uses the **same pattern**:

| Symbol | Meaning |
|--------|--------|
| $\theta$ | **Unknown** in the problem—the thing you want to learn or act under uncertainty about |
| $D$ | **Observed data** (fixed once measured) |
| $p(D \mid \theta)$ | **Likelihood**: “how probable is this $D$ if $\theta$ were true?”—a **score** over $\theta$, not a distribution over $\theta$ |
| $p(\theta)$ | **Prior**: belief **before** $D$ |
| **Bayesian inference** | Form **posterior** $p(\theta \mid D) \propto p(D \mid \theta)\, p(\theta)$, normalize; use that distribution for decisions, predictions, or further learning |

**Belief** in formulas is just the prior or posterior: **weights** (discrete) or a **density** (continuous) over $\theta$.

### Can $D$ be discrete or continuous?

**Yes—both.** Bayes’ rule does not care *a priori* whether data are discrete or continuous; you only need a **well-defined likelihood** $p(D \mid \theta)$ (or a density $f(D \mid \theta)$ when $D$ is continuous).

| Kind of data | Typical $D$ | Comment |
|--------------|-------------|---------|
| **Discrete** | Counts (heads $k$ out of $n$), class labels, word tokens, event times binned into categories | Likelihoods are often Binomial, categorical/softmax, Poisson, etc. **Examples 1–2** use discrete summaries $(k,n)$. |
| **Continuous** | Real measurements: temperatures, lengths, sensor voltages, stacks of pixel values modeled as $\mathbb{R}^d$ | Likelihoods are often **Gaussians** or other densities on $\mathbb{R}^d$. **Example 3** uses $D = (x_1,\ldots,x_n)$ with a Gaussian sampling model. |
| **Mixed** | A label **and** a feature vector per example, or partly censored data | You write one joint model $p(D \mid \theta)$ that includes both parts. |

**Independence of roles:** whether $\theta$ is discrete or continuous (your **unknown**) is a **separate** choice from whether $D$ is discrete or continuous (your **observations**). You can have continuous $\theta$ with discrete $D$ (Beta–Binomial), continuous $\theta$ with continuous $D$ (Gaussian mean), discrete $\theta$ with continuous $D$ (finite list of hypotheses about a physical constant measured with noise), and so on.

---

### Example 1 — Smallest case: only two possible values of $\theta$ (coin)

- **$\theta$:** which bias the coin has. Only two possibilities: $\theta \in \{0.5,\, 0.9\}$ (fair vs biased).
- **$D$:** you flip 5 times and record **5 heads** (order does not matter here; counts are enough).
- **Likelihood:** Binomial with $n=5$, $k=5$. For each candidate $\theta$,

$$
p(D \mid \theta) = \theta^{5}(1-\theta)^{0} = \theta^5.
$$

So $p(D \mid 0.5) = (1/2)^5 = 1/32$, and $p(D \mid 0.9) = (0.9)^5 \approx 0.590$.
- **Prior:** e.g. $p(\theta = 0.5) = 0.6$, $p(\theta = 0.9) = 0.4$.
- **Bayesian inference:** multiply **likelihood $\times$ prior**, then divide by their sum so the result is probabilities:

  - Unnormalized: $0.6 \times (1/32) = 0.01875$ for $\theta = 0.5$; $0.4 \times 0.590 \approx 0.236$ for $\theta = 0.9$.
  - Sum $\approx 0.255$. **Posterior:** $p(\theta = 0.5 \mid D) \approx 0.07$, $p(\theta = 0.9 \mid D) \approx 0.93$.

**Read that line slowly:** under $\theta = 0.9$, getting five heads is fairly ordinary; under $\theta = 0.5$ it is rare. So the **likelihood** is much larger for $\theta = 0.9$ than for $\theta = 0.5$. That is why the data **pull** belief toward the biased theory.

**What about “penalizing if the prior were tiny”?** Posterior mass is **prior $\times$ likelihood** (then renormalized). So the biased theory only gets a large posterior share if **both** (i) it fits the data well **and** (ii) you did not assign it essentially **zero** prior weight. Example: keep the same likelihoods but swap to an extreme prior $p(\theta = 0.5) = 0.99$, $p(\theta = 0.9) = 0.01$. Then the unnormalized weight for $\theta = 0.9$ is about $0.01 \times 0.590 \approx 0.0059$, while $\theta = 0.5$ gets $0.99 \times 0.03125 \approx 0.031$. After normalizing, **most** mass still lands on “fair”—because you started almost convinced the coin was fair. Strong data can overcome a modestly low prior (**as in our main $0.6/0.4$ case**), but a **very** small prior keeps acting like a **downweight** until evidence is overwhelming.

*Takeaway:* $\theta$ is **discrete and tiny**; $D$ is **counts**; the likelihood is **explicit**; Bayes **updates a 2-way belief**.

---

### Example 2 — Still one coin, but $\theta$ is any bias in $(0,1)$ (Beta–Binomial)

- **$\theta$:** unknown probability of heads, **one real number** in $(0,1)$.

  **Why continuous here but discrete in Example 1?** That is a **modeling choice**, not a law of nature. In Example 1 we **deliberately** said: “only two stories exist—fair or this one specific bias.” In Example 2 we say: “the coin has **some** bias, and I am not restricting it to a short list—it could be $0.37$, $\sqrt{2}/10$, etc.” A real coin’s physical bias is **not** magically confined to two numbers; allowing **any** value in $(0,1)$ is the usual textbook setup and matches the **Beta–Binomial conjugate pair**.

  **Could $\theta$ still be discrete?** Yes. You could list $1{,}000$ plausible values on a grid and put a prior **over that finite set**—then belief would be $1{,}000$ probabilities again (like Example 1, but bigger). Many applied models do that (**discretization**). Example 2 is the **continuous** version: one unknown proportion, prior with a **density** on the interval, posterior computed exactly as another Beta.

- **$D$:** $k$ heads in $n$ flips (again summarized by $(k,n)$).
- **Likelihood:** same Binomial story,

$$
p(D \mid \theta) = \binom{n}{k}\, \theta^{k}(1-\theta)^{n-k}.
$$

The combinatorial factor is **constant in $\theta$** for fixed $D$, so for updating belief only the shape $\theta^{k}(1-\theta)^{n-k}$ matters.
- **Prior:** a **probability density** on $(0,1)$, e.g. $\theta \sim \mathrm{Beta}(\alpha,\beta)$—see the [Beta primer]({{ site.baseurl }}{% post_url 2026-04-03-beta-distribution-primer %}).

  **Why “density” and not just two probabilities like Example 1?** There $\theta$ could only be $0.5$ or $0.9$, so belief was two numbers that sum to $1$. Here $\theta$ could be **any** number in a whole interval—**infinitely many** distinct values, not just two.

  **Why you cannot assign positive probability to every single exact $\theta$:** suppose, toward a contradiction, that **every** possible bias you care about got at least a small slice, say $P(\theta = \text{that exact number}) \ge \varepsilon$ for some fixed $\varepsilon > 0$. Pick $100$ **different** candidate values of $\theta$: their probabilities alone would add to at least $100\varepsilon$. Pick $1{,}000$, or $10^9$ different values—the sum keeps growing, but **total probability can never exceed $1$**. So “every point gets a fixed positive lump” is impossible. In fact an interval of $\mathbb{R}$ contains **so many** points that the only consistent option is: probability **concentrates on sets with positive length** (intervals), not on isolated exact numbers; a **density** $p(\theta)$ tells you how much probability **per unit length** lands in each region, so $\mathbb{P}(a < \theta < b) = \int_a^b p(\theta)\, d\theta$ **(area under the curve)**. The Beta formula in the primer **is** that density.

- **Bayesian inference:** **posterior is another Beta**, $\theta \mid D \sim \mathrm{Beta}(\alpha + k,\, \beta + n - k)$. You report that **posterior density** (or summaries: mean, credible intervals, etc.).

*Takeaway:* same $D$ and same likelihood **family** as Example 1, but $\theta$ is **continuous**; belief is a **curve**, not two numbers.

---

### Example 3 — Unknown mean of a Gaussian (conjugate: prior and posterior both Normal)

- **$\theta$:** unknown mean $\mu \in \mathbb{R}$ of a measurement process (variance **known** for simplicity, say $\sigma^2$ fixed).
- **$D$:** independent draws $x_1,\ldots,x_n$, modeled as $x_i \mid \mu \sim \mathcal{N}(\mu, \sigma^2)$.
- **Likelihood:**

$$
p(D \mid \mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \,\exp\!\Big(-\frac{(x_i - \mu)^2}{2\sigma^2}\Big).
$$

As a function of $\mu$, this is driven by how close $\mu$ is to the data; define the sample mean $\bar{x} = \frac{1}{n}\sum_i x_i$—the likelihood is **peaked** near $\bar{x}$.
- **Prior:** Gaussian belief on the mean, $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$.
- **Bayesian inference:** the posterior $p(\mu \mid D)$ is **another Gaussian** (conjugacy again). You get an **updated** mean and variance combining **prior center** and **data** $\bar{x}$ with weights that depend on precisions $1/\tau_0^2$ and $n/\sigma^2$.

*Takeaway:* $\theta$ is **one real number** but **unbounded**; likelihood is a **product of Gaussians**; Bayes still gives a **closed-form** posterior.

---

### Example 4 — Hidden **state** over time (tracking): $\theta$ is “where you are”

- **$\theta$ (often written $x_t$):** true position, velocity, or full state at time $t$—**not directly observed**.
- **$D$:** noisy sensor readings $y_t$ (and possibly a sequence $y_1,\ldots,y_T$).
- **Likelihood / generative story:** e.g. $y_t \mid \theta_t \sim \mathcal{N}(h(\theta_t), R_t)$—“if the true state were $\theta_t$, the measurement would be Gaussian around $h(\theta_t)$.” Dynamics $\theta_{t+1} \mid \theta_t$ complete the model.
- **Bayesian inference:** maintain **belief about the current state** $p(\theta_t \mid y_{1:t})$ and **predict forward**; each new $y_{t+1}$ **updates** that belief. Under linear-Gaussian structure this is the **Kalman filter**—closed-form Gaussian recursion.

Full detail: [Bayesian foundations of Kalman filtering]({{ site.baseurl }}{% post_url 2024-09-22-bayesian-foundations-kalman %}).

*Takeaway:* $\theta$ is a **vector evolving over time**; $D$ is a **sequence**; inference is **recursive** Bayes, not a single static update.

---

### Example 5 — Huge $\theta$: neural network weights (posterior not tractable)

- **$\theta$:** millions of weight and bias parameters of a model.
- **$D$:** large training set (images, text, labels).
- **Likelihood / model:** e.g. categorical labels with a softmax probability for each example; the full likelihood is the **product** of those probabilities under independence. **Schematically**, writing a training loss as $\mathrm{Loss}(\theta; D)$,

$$
p(D \mid \theta) \propto \exp\big(-\mathrm{Loss}(\theta; D)\big)
$$

when the loss is **negative log-likelihood** (ignoring constants that do not depend on $\theta$).
- **Prior:** e.g. independent Gaussians on weights (“weight decay”) or more structured priors.
- **Bayesian inference:** the **posterior** $p(\theta \mid D)$ lives in millions of dimensions—**no closed form**. In practice you use **MCMC** (samples) or **variational inference** (optimize $q(\theta) \approx p(\theta \mid D)$)—see [normalization difficulties]({{ site.baseurl }}{% post_url 2025-12-24-normalization-constant-problem %}), [VAE / variational view]({{ site.baseurl }}{% post_url 2026-01-01-how-vaes-avoid-computing-partition-function %}), [KL]({{ site.baseurl }}{% post_url 2025-03-20-understanding-kl-divergence %}).

*Takeaway:* **same Bayes formula**; only the **dimension** and **computational strategy** change.

---

### Bayes vs “just maximizing the likelihood”

In every example you *could* report $\hat{\theta} = \arg\max_\theta p(D \mid \theta)$ (**maximum likelihood**). **Bayesian inference** instead keeps **uncertainty**: $p(\theta \mid D)$ reflects both **how sharply** the likelihood peaks and **what you believed before** $D$. That matters for safety-critical or data-poor regimes—and for propagating uncertainty to predictions.

---

## Where Bayesian inference came from (brief intuition)

Historically, the core move is **inverse probability**: the forward problem is “given a chance mechanism, what data might we see?” The **inverse** problem is “given data we did see, what should we believe about the mechanism?” That is exactly the posture of **learning a parameter** from observations.

- **18th century (Bayes, then Laplace).** The famous essay published after Bayes’ death posed something essentially like: observe Binomial counts, infer an unknown success probability. The mathematics was **turning a likelihood into a distribution over the parameter**—the same spirit as the [Beta–Binomial conjugate update]({{ site.baseurl }}{% post_url 2026-04-03-beta-distribution-primer %}) you still teach first today. Laplace and others pushed the same logic across astronomy, demography, and measurement—always: **prior belief + data → updated belief**.

- **20th century (statistics broadens).** Much of mainstream statistics emphasized **repeatable-sample** reasoning (confidence intervals, $p$-values, design of experiments). Bayesian ideas kept a home wherever **explicit uncertainty about unknowns** was useful: decision theory, a strand of econometrics, sparsely in engineering—until computation caught up.

- **Engineering state estimation.** When dynamics and noise are **linear-Gaussian**, the Kalman filter is not an ad hoc recipe—it is a **closed-form recursion for Gaussian posteriors** over hidden states. That is Bayesian inference on a time axis; see [Bayesian foundations of Kalman filtering]({{ site.baseurl }}{% post_url 2024-09-22-bayesian-foundations-kalman %}).

- **Late 20th century → now.** **MCMC** turned high-dimensional integrals into feasible sampling. **Variational inference** traded exactness for scalability—crucial for large models (the same normalization issue in [the normalization-constant note]({{ site.baseurl }}{% post_url 2025-12-24-normalization-constant-problem %}) and [VAE-style bounds]({{ site.baseurl }}{% post_url 2026-01-01-how-vaes-avoid-computing-partition-function %})).

The through-line is unchanged from Bayes’ essay to modern deep learning: **posterior $\propto$ likelihood $\times$ prior**; only the **models** and **computers** changed.

---

## Bayes’ rule for parameters

The posterior is defined by **Bayes’ theorem**:

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)}.
$$

- **Prior** $p(\theta)$: what you believe about $\theta$ *before* $D$ (or a formal placeholder for regularization).
- **Likelihood** $p(D \mid \theta)$: how probable the observations are under $\theta$.
- **Marginal likelihood** (evidence) $p(D) = \int p(D \mid \theta)\, p(\theta)\, d\theta$: the normalizing constant that makes $p(\theta \mid D)$ integrate to $1$.

Equivalently, up to constants in $\theta$:

$$
p(\theta \mid D) \propto p(D \mid \theta)\, p(\theta),
$$

often written as **posterior $\propto$ likelihood $\times$ prior**. Computing $p(D)$ is the bottleneck for many models; see the normalization-constant note linked above.

---

## Posterior predictions (one line)

If you care about **future** observations $D_{\mathrm{new}}$ rather than $\theta$ itself, the **posterior predictive** averages the conditional model over the posterior:

$$
p(D_{\mathrm{new}} \mid D) = \int p(D_{\mathrm{new}} \mid \theta)\, p(\theta \mid D)\, d\theta.
$$

That integral is another place where **exact** answers are rare—but the **definition** is unified: **uncertainty about $\theta$ propagates** into predictions.

---

## Frequentist contrast (without taking sides)

A common frequentist habit is to treat $\theta$ as **fixed** and unknown, with randomness only in **data**. Estimators (MLE, etc.) summarize data by a recipe $\hat{\theta}(D)$; uncertainty is assessed by **sampling distributions** over *hypothetical repeated* datasets.

Bayesian inference instead treats **uncertainty about $\theta$** explicitly via $p(\theta \mid D)$. Both traditions can coexist: you can use **Bayesian models** with **frequentist evaluation** of procedures, and vice versa.

---

## When is inference “easy”?

**Closed-form or low-dimensional posteriors** appear when structure is kind—classic case: **conjugate priors**. [*An Introduction to Bayesian Thinking*](https://statswithr.github.io/book/bayesian-inference.html#three-conjugate-families) emphasizes that when prior and likelihood form a **conjugate pair**, “we could apply the continuous version of Bayes’ rule **without having to do any integration**” for the posterior—exactly the shortcut behind Beta–Binomial and Normal–Normal classroom models. The [Beta–Binomial]({{ site.baseurl }}{% post_url 2026-04-03-beta-distribution-primer %}) pair matches their first conjugate family.

**Linear-Gaussian state-space models** are another tractable island: the Kalman filter propagates **Gaussian posteriors** in closed form under linear dynamics and Gaussian noise (see the [Kalman Bayes post]({{ site.baseurl }}{% post_url 2024-09-22-bayesian-foundations-kalman %})).

---

## When do you need MCMC, VI, or other approximations?

For rich models (deep nets, hierarchical GLMs, complex graphical models), the posterior is **high-dimensional** and $p(D)$ has **no closed form**. Typical tools:

- **Markov chain Monte Carlo (MCMC)** — approximate $p(\theta \mid D)$ with correlated samples; asymptotically exact, can be slow.
- **Variational inference (VI)** — fit a simpler $q(\theta)$ to minimize divergence from the true posterior; scalable, biased but useful (VAEs build on this idea—link above).

The pattern is always the same **conceptually**; only the **algorithm** changes.

---

## Closing

Bayesian inference is **not** a single formula—it is a **workflow**: encode beliefs in a prior, score a model with a likelihood, obtain a posterior, and (often) struggle honorably with normalization or high dimensions. When structure lines up, that workflow collapses to **algebra** you can do by hand; when it does not, you reach for **samples** or **variational** bounds—same posterior definition, heavier computation.

For intuition about **low-dimensional** “compressed” belief states on $(0,1)$, see [Compressed Uncertainty: Beta Intuition Without the Formula Sheet]({{ site.baseurl }}{% post_url 2026-04-04-beta-distribution-intuition-compressed-parameterization %}).

### Textbook-style reference

[*An Introduction to Bayesian Thinking: A Companion to the Statistics with R Course*](https://statswithr.github.io/book/) — open-access book (bookdown), companion to the Bayesian Statistics material in the [Statistics with R specialization](https://www.coursera.org/specializations/statistics) on Coursera; [source on GitHub](https://github.com/StatsWithR/book). Chapter 2 — [Bayesian Inference](https://statswithr.github.io/book/bayesian-inference.html), especially [§2.2 Three conjugate families](https://statswithr.github.io/book/bayesian-inference.html#three-conjugate-families). **Block quotations above** are taken from that chapter.
