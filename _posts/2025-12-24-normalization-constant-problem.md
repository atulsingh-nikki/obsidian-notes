---
layout: post
title: "The Normalization Constant Problem: Why Computing Z Is So Hard"
description: "An intuitive explanation of why the partition function Z makes probabilistic modeling intractable in high dimensions—and how modern generative models cleverly sidestep this fundamental obstacle."
tags: [probability, machine-learning, generative-models, statistical-mechanics]
---

*This post explores one of the fundamental obstacles in generative modeling, discussed in [Why Discriminative Learning Dominated First]({{ site.baseurl }}{% link _posts/2025-12-25-why-discriminative-learning-came-first.md %}). For the broader context, see [Machine Learning Paradigms: Learning Distributions vs Approximating Functions]({{ site.baseurl }}{% link _posts/2025-12-26-machine-learning-paradigms-distributions-vs-functions.md %}).*

## The Setup: What We're Trying to Do

**Goal**: Model a probability distribution over data $p(x)$ where $x$ could be:
- An image: $x \in \mathbb{R}^{256 \times 256 \times 3} \approx \mathbb{R}^{200{,}000}$
- A sentence: $x = (w_1, w_2, \dots, w_n)$ where each $w_i$ is a word
- A molecule: $x$ represents atomic positions and bonds

We want to assign probabilities: **Which data points are likely? Which are not?**

## How We Build Generative Models (The Big Picture)

Here's what happens when we train a generative model:

**1. We have a dataset**: 1 million cat photos, for example.

**2. We build a neural network**: This network looks at any image and outputs a **score** - a single number representing "how much the model likes this image."

```
Neural Network: Image → Score
```

**3. We convert scores to unnormalized probabilities**: 

$$\tilde{p}_\theta(x) = \exp(\text{score})$$

This gives us positive numbers that capture the model's belief:
- High score → Large $\tilde{p}(x)$ → "This looks like training data"
- Low score → Small $\tilde{p}(x)$ → "This doesn't look like training data"

**Example outputs**:
- Cat photo: $\tilde{p}(x) = 1000$
- Dog photo: $\tilde{p}(x) = 10$  
- Random noise: $\tilde{p}(x) = 0.001$

**4. The problem**: These numbers **don't sum to 1**! 

When we sum $\tilde{p}(x)$ over all possible images, we might get 1 trillion, or 0.00001, or anything. We don't know because we can't compute the sum.

**5. What we need**: To convert to valid probabilities, we divide by the total:

$$p(x) = \frac{\tilde{p}(x)}{Z} \quad \text{where } Z = \text{sum of } \tilde{p}(x) \text{ over all possible images}$$

**This $Z$ is the normalization constant - and computing it is impossible.**

So in summary:
- **$\tilde{p}(x)$ = What our model actually outputs** (easy to compute, but not a valid probability)
- **$Z$ = What we need to make it valid** (impossible to compute)
- **$p(x) = \tilde{p}(x)/Z$ = Valid probability** (what we want, but can't get)

## The Mathematical Requirement

To define a valid probability distribution, we need:

$$p(x) = \frac{\tilde{p}(x)}{Z}$$

where:
- $\tilde{p}(x)$ is an **unnormalized** probability (easy to compute)
- $Z$ is the **normalization constant** (ensures $\int p(x) dx = 1$)

**The partition function** $Z$ is defined as:

$$Z = \int \tilde{p}(x) \, dx$$

or for discrete $x$:

$$Z = \sum_x \tilde{p}(x)$$

**The problem**: Computing $Z$ requires **summing or integrating over all possible data points**. In high dimensions, this becomes computationally impossible.

## Understanding $\tilde{p}(x)$ in Context

**What is $\tilde{p}(x)$ when modeling data?**

When building a generative model (say, to generate images), here's what happens:

1. **You have data**: A dataset of images (cats, dogs, faces, etc.)
2. **You build a neural network**: The network takes an image and outputs a **score** indicating how "realistic" it is
3. **You create $\tilde{p}(x)$**: Convert the score to a positive number: $\tilde{p}(x) = \exp(\text{network_score})$

**This $\tilde{p}(x)$ is the output of your model** - it's what the network "thinks" about each image:
- High score for realistic images (e.g., $\tilde{p}(\text{cat photo}) = 1000$)
- Low score for unrealistic images (e.g., $\tilde{p}(\text{noise}) = 0.01$)

**Why "easy to compute"?** For any specific image $x$, computing $\tilde{p}(x)$ is just:
- One forward pass through your neural network
- One exponential operation
- Takes milliseconds on a GPU

**Why "unnormalized"?** These scores don't sum to 1 across all possible images. Without knowing $Z$, you can't convert them to valid probabilities. The model can say "this is more realistic than that," but not "this has a 90% probability of being a cat."

**The tragedy**: Your model naturally outputs $\tilde{p}(x)$ for any image you show it, but to train it properly (maximize likelihood) or evaluate probabilities, you need $Z$ - which requires evaluating $\tilde{p}(x)$ for every possible image!

## An Intuitive Example: The Fair Coin

**Simple case**: A coin flip. Let's say we have an unnormalized model:

$$\tilde{p}(\text{heads}) = 3, \quad \tilde{p}(\text{tails}) = 3$$

To get probabilities:

$$Z = \tilde{p}(\text{heads}) + \tilde{p}(\text{tails}) = 3 + 3 = 6$$

$$p(\text{heads}) = \frac{3}{6} = 0.5, \quad p(\text{tails}) = \frac{3}{6} = 0.5$$

**Easy!** We only need to sum over 2 outcomes.

## When Things Get Hard: The Image Example

**Complex case**: A $256 \times 256$ grayscale image. Each pixel can take 256 values (0-255).

**Number of possible images**:

$$256^{256 \times 256} = 256^{65{,}536}$$

**Converting to base 10**: To understand the scale, we convert this to powers of 10:

$$256^{65{,}536} = 10^{x}$$

Taking $\log_{10}$ of both sides:

$$x = 65{,}536 \cdot \log_{10}(256) = 65{,}536 \cdot \log_{10}(2^8) = 65{,}536 \cdot 8 \cdot \log_{10}(2)$$

Since $\log_{10}(2) \approx 0.30103$:

$$x = 65{,}536 \times 8 \times 0.30103 \approx 157{,}826$$

Therefore: $256^{65{,}536} \approx 10^{157{,}826}$

To put this in perspective:
- Number of atoms in the observable universe: $\approx 10^{80}$
- Number of possible images: $\approx 10^{157{,}826}$

**Time perspective**: Suppose processing one image takes just 1 millisecond:
- Total time needed: $10^{157{,}826}$ milliseconds $= 10^{157{,}823}$ seconds
- Converting to years: $\frac{10^{157{,}823}}{3.16 \times 10^7} \approx 3.16 \times 10^{157{,}815}$ years
- Age of the universe: $\approx 1.38 \times 10^{10}$ years

**That's $10^{157{,}805}$ times the age of the universe!** Even processing a trillion images per millisecond wouldn't make a dent.

**To compute $Z$, we need to sum over $10^{157{,}826}$ images.** Even if we could evaluate $\tilde{p}(x)$ a billion times per second, the universe would end many times over before we finish.

**This is not a matter of "get better computers"—this is fundamentally impossible.**

## Why Do We Need Z?

### 1. To Evaluate Likelihoods

Given an image $x$, we want to compute $p(x)$—how likely is this image under our model?

Without $Z$, we only have $\tilde{p}(x)$, which doesn't mean anything probabilistically:
- Is $\tilde{p}(x) = 100$ high or low?
- We can't tell without comparing to $Z = \sum_{x'} \tilde{p}(x')$

### 2. To Train Models

**Why maximum likelihood?** The idea is simple: we want our model $p_\theta(x)$ to assign **high probability to the data we've actually seen**. If we have 1 million cat photos, we adjust the parameters $\theta$ so that these specific photos get high probability under $p_\theta(x)$. This makes the model "believe" that similar data is likely to occur.

**Maximum likelihood training** tries to maximize:

$$\max_\theta \prod_{i=1}^N p_\theta(x_i) = \max_\theta \sum_{i=1}^N \log p_\theta(x_i)$$

**Why is taking log still a max problem?** Because $\log$ is a monotonically increasing function: if $a > b$, then $\log a > \log b$. So maximizing the likelihood $\prod p_\theta(x_i)$ is equivalent to maximizing the log-likelihood $\sum \log p_\theta(x_i)$—they have the same optimal $\theta^*$. We use log because it turns products into sums (easier math) and prevents numerical underflow.

In words: find parameters $\theta$ that make the observed data $\{x_1, \dots, x_N\}$ as likely as possible.

**But here's the problem**: Recall from earlier that our probability is:

$$p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}$$

where $Z_\theta = \sum_x \tilde{p}_\theta(x)$ is the partition function. Taking the log:

$$\log p_\theta(x) = \log \tilde{p}_\theta(x) - \log Z_\theta$$

**The gradient with respect to parameters $\theta$**:

**Why compute the gradient?** To actually maximize the log-likelihood, we need to know how to update $\theta$. The gradient $\frac{\partial \log p_\theta(x)}{\partial \theta}$ tells us which direction to adjust the parameters to increase the likelihood.

$$\frac{\partial \log p_\theta(x)}{\partial \theta} = \frac{\partial \log \tilde{p}_\theta(x)}{\partial \theta} - \frac{\partial \log Z_\theta}{\partial \theta}$$

The second term is:

**Step 1**: Apply the chain rule to $\log Z_\theta$:

$$\frac{\partial \log Z_\theta}{\partial \theta} = \frac{1}{Z_\theta} \frac{\partial Z_\theta}{\partial \theta}$$

**Step 2**: Recall that $Z_\theta = \sum_x \tilde{p}_\theta(x)$ (or $\int \tilde{p}_\theta(x) dx$ in continuous case). Differentiate:

$$\frac{\partial Z_\theta}{\partial \theta} = \frac{\partial}{\partial \theta} \sum_x \tilde{p}_\theta(x) = \sum_x \frac{\partial \tilde{p}_\theta(x)}{\partial \theta}$$

**Step 3**: Combine the steps:

$$\frac{\partial \log Z_\theta}{\partial \theta} = \frac{1}{Z_\theta} \sum_x \frac{\partial \tilde{p}_\theta(x)}{\partial \theta}$$

**This sum is over all possible $x$!** For images, that's $10^{157,826}$ terms—intractable again.

### 3. To Sample

Even if we ignore training, generating samples from $p(x)$ typically requires knowing $Z$ or being able to evaluate $p(x)$ efficiently. Without $Z$, most classical sampling methods fail.


## Why Discriminative Models Don't Have This Problem

**Discriminative models** learn $p(Y \mid X)$, not $p(X)$. For classification with $K$ classes:

$$p(Y=k \mid X=x) = \frac{e^{f_\theta^{(k)}(x)}}{\sum_{j=1}^K e^{f_\theta^{(j)}(x)}}$$

**The "partition function" here** is:

$$Z(x) = \sum_{j=1}^K e^{f_\theta^{(j)}(x)}$$

**Key difference**: This sum is over **classes** (typically 10-1000), not over all possible inputs $x$ (typically $10^{157{,}826}$).

**Result**: Computing $Z(x)$ is trivial—just sum over a handful of classes. This is why discriminative learning was tractable decades before generative modeling.

## How Modern Generative Models Sidestep Z

The history of generative modeling is largely a story of clever ways to **avoid computing $Z$**.

### 1. Variational Autoencoders (VAEs, 2014)

**Key idea**: Don't compute $p(x)$ directly. Instead, optimize a **lower bound**:

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$$

This **evidence lower bound (ELBO)** is tractable to compute and differentiate. We never need $Z$!

**Trade-off**: We're optimizing a bound, not the true likelihood. VAEs tend to produce blurrier samples.

### 2. Generative Adversarial Networks (GANs, 2014)

**Key idea**: Don't model $p(x)$ at all. Just learn a mapping $G(z) \to x$ from noise to data.

**Training**: Use a discriminator $D(x)$ to distinguish real from fake. The generator $G$ is trained adversarially to fool $D$.

**Result**: No partition function ever appears! We never evaluate $p(x)$.

**Trade-off**: Notoriously unstable training, mode collapse, no likelihood evaluation.

### 3. Normalizing Flows (2015)

**Key idea**: Build $p(x)$ from invertible transformations of a simple base distribution $p(z)$:

$$x = f(z), \quad p(x) = p(z) \left\lvert \det \frac{\partial f^{-1}}{\partial x} \right\rvert$$

**The magic**: The change-of-variables formula gives us $p(x)$ **directly**, without integration!

**Trade-off**: Requires special architectures (invertible layers), limits expressiveness.

### 4. Diffusion Models (2020)

**Key idea**: Model a gradual noising process $x_0 \to x_1 \to \dots \to x_T$ where $x_T \sim \mathcal{N}(0, I)$.

Learn to reverse this process: $x_T \to \dots \to x_1 \to x_0$.

**Training objective**: Denoising score matching, which is tractable:

$$\mathbb{E}_{t, x_0, \varepsilon} \left[\left\lVert \varepsilon - \varepsilon_\theta(x_t, t)\right\rVert^2\right]$$

**Result**: No $Z$ appears! We just learn to denoise.

**Trade-off**: Inference is slow (requires many denoising steps), but sample quality is state-of-the-art.

### 5. Autoregressive Models (e.g., GPT)

**Key idea**: Factor $p(x)$ using the chain rule:

$$p(x_1, x_2, \dots, x_n) = p(x_1) p(x_2 \mid x_1) p(x_3 \mid x_1, x_2) \cdots p(x_n \mid x_1, \dots, x_{n-1})$$

Each conditional $p(x_i \mid x_{1:i-1})$ is a **small classification problem** (e.g., over vocabulary).

**The partition functions** $Z_i = \sum_{x_i} \tilde{p}(x_i \mid x_{1:i-1})$ are over vocabulary size (50K-100K words), not data space ($\infty$).

**Result**: Tractable! This is why GPT and other language models work.

**Trade-off**: Must generate sequentially (slow), and ordering matters (works well for text, less well for images).

## The Deeper Lesson

The normalization constant problem is not just a technical nuisance—it reveals a profound asymmetry in machine learning:

**Discriminative learning** (predict $Y$ given $X$):
- Partition function over labels: $Z = \sum_{y} \exp(f(x, y))$
- Typically 10-1000 terms → **tractable**

**Generative learning** (model $X$ itself):
- Partition function over data: $Z = \int \exp(f(x)) dx$
- Typically $10^{100{,}000}$ terms → **intractable**

**This is why discriminative learning dominated for decades.** It wasn't theoretically superior; it was just computationally feasible with available methods.

**Modern generative modeling succeeds** by cleverly avoiding the partition function altogether—through adversarial training, variational bounds, invertible transformations, or clever factorizations.

## A Statistical Physics Perspective

**Fun fact**: The partition function $Z$ comes from statistical mechanics, where it's central to thermodynamics:

$$Z = \sum_{\text{states}} e^{-E(\text{state})/kT}$$

In physics, computing $Z$ is equally hard (NP-complete in general), but for some special systems (e.g., Ising model on planar graphs), exact solutions exist.

**Machine learning borrowed the problem from physics**, and like physicists, ML researchers have developed approximate methods (mean-field approximations, MCMC, variational methods) when exact computation is impossible.

## Key Takeaways

1. **Normalization constant $Z$** is required to turn unnormalized scores into probabilities.

2. **Computing $Z$ requires summing/integrating over all possible data points**, which is intractable for high-dimensional data ($10^{100{,}000}$ possibilities).

3. **Discriminative models avoid this** by having partition functions over small label sets (10-1000 classes).

4. **Generative models were stuck** until algorithmic breakthroughs (GANs, VAEs, flows, diffusion) found ways to sidestep $Z$.

5. **The lesson**: Sometimes the path forward isn't solving the hard problem, but **reframing the problem** to avoid it entirely.

## Further Reading

**Classic Papers**:
- Hinton (2002), "Training Products of Experts by Minimizing Contrastive Divergence" - early attempt to approximate $Z$
- Kingma & Welling (2014), "Auto-Encoding Variational Bayes" - ELBO avoids $Z$
- Goodfellow et al. (2014), "Generative Adversarial Nets" - no likelihood, no $Z$

**Textbooks**:
- Murphy, *Machine Learning: A Probabilistic Perspective* (Chapter 20: undirected graphical models)
- Bishop, *Pattern Recognition and Machine Learning* (Chapter 8: graphical models)

**Related Blog Posts**:
- [Why Discriminative Learning Dominated First]({{ site.baseurl }}{% link _posts/2025-12-25-why-discriminative-learning-came-first.md %}) - historical context
- [Machine Learning Paradigms: Learning Distributions vs Approximating Functions]({{ site.baseurl }}{% link _posts/2025-12-26-machine-learning-paradigms-distributions-vs-functions.md %}) - the fundamental divide
- [The Curse of Dimensionality in Machine Learning]({{ site.baseurl }}{% link _posts/2025-12-23-curse-of-dimensionality.md %}) - why high dimensions are hard

---

**The bottom line**: The partition function $Z$ is the invisible villain in generative modeling. For decades, it made distribution learning intractable. Modern AI succeeded not by computing $Z$, but by building models that **never need it**. This is a masterclass in problem reframing: when you can't solve a problem, change the problem.
