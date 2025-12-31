---
layout: post
title: "How Variational Autoencoders Avoid Computing the Partition Function"
description: "A deep dive into how VAEs sidestep the intractable partition function Z through the Evidence Lower Bound (ELBO), making generative modeling tractable."
tags: [deep-learning, generative-models, vae, probability, machine-learning]
---

*This article examines one of machine learning's most elegant workarounds for the partition function problem. For background, see [The Normalization Constant Problem: Why Computing Z Is So Hard]({{ site.baseurl }}{% link _posts/2025-12-24-normalization-constant-problem.md %}) and [Why Discriminative Learning Dominated First]({{ site.baseurl }}{% link _posts/2025-12-25-why-discriminative-learning-came-first.md %}).*

## Understanding the Challenge

As we explored in our discussion of partition functions, modeling probability distributions $p(x)$ over high-dimensional data encounters a fundamental obstacle—the normalization constant:

$$p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}$$

Here, $Z_\theta = \sum_x \tilde{p}_\theta(x)$ demands summing across an impossibly vast configuration space (reaching $10^{157,826}$ possibilities for 256×256 images).

**What this blocks us from doing**:
1. Computing $p_\theta(x)$ exactly (requires knowing $Z_\theta$)
2. Implementing maximum likelihood training (gradient computation needs $\log Z_\theta$)
3. Generating samples efficiently (typical sampling algorithms need probability evaluations)

**The VAE solution**: Rather than attempting to compute $p(x)$ directly, **optimize a tractable lower bound instead!**

## Latent Variables: The Core Innovation

Variational Autoencoders introduce **latent variables** $z$—compact hidden codes that capture data's essential structure.

**How generation works**:
1. Draw a latent code: $z \sim p(z)$ (usually $\mathcal{N}(0, I)$)
2. Generate data from this code: $x \sim p_\theta(x \mid z)$

**Making it concrete**: Consider $z$ as a "compressed instruction set" or "blueprint" for data creation:
- **First**: Select random instructions from a simple distribution (imagine rolling Gaussian-weighted dice to pick $z$)
- **Second**: Execute these instructions through a decoder network that transforms $z$ into observable data $x$

Take face generation as an example:
- $z$ might encode: [smile_intensity=0.8, wears_glasses=0.2, apparent_age=25, hair_shade=brown, ...]
- The decoder interprets these specifications and renders a corresponding face
- Different random $z$ samples → diverse face outputs

**The fundamental shift**: Instead of directly modeling the overwhelming space of all possible images (requiring that intractable $Z$), we model the *generative process* that creates images from simple instructions.

**Connecting to the full data probability $p_\theta(x)$:**

Our model defines a joint distribution over observations and latent codes:

$$p_\theta(x, z) = p_\theta(x \mid z) p(z)$$

This expresses: "the probability of observing both $x$ and $z$ together equals the prior probability of $z$ multiplied by the conditional probability of $x$ given $z$."

**Probability foundations reminder**:
- **Joint distribution** $p(x, z)$: Probability of both $x$ AND $z$ occurring simultaneously. Example: "What's the chance someone is tall (x) AND has blue eyes (z)?"
- **Marginal distribution** $p(x)$: Probability of $x$ alone, independent of $z$. Example: "What's the chance someone is tall, regardless of eye color?"
- **Marginalization process**: Extract the marginal from the joint by summing over all values of the other variable: $p(x) = \sum_z p(x, z)$ (or $\int p(x, z) dz$ for continuous variables)

**Obtaining $p_\theta(x)$ alone**: We **marginalize out** (integrate over) every possible latent code:

$$p_\theta(x) = \int p_\theta(x, z) dz = \int p_\theta(x \mid z) p(z) dz$$

**The underlying intuition**: To determine the total probability of generating image $x$, we must account for every possible instruction set $z$ that could have produced it:
- Each instruction set $z$ has some prior probability $p(z)$ of being selected
- Given specific instructions $z$, there's a conditional probability $p_\theta(x \mid z)$ of producing image $x$
- Total probability: sum contributions from all possible instruction sets $\int p_\theta(x \mid z) p(z) dz$

**Clarifying normalization: Are these proper probability distributions?** Absolutely! Unlike unnormalized densities $\tilde{p}(x)$ in energy-based models:
- $p(z) = \mathcal{N}(0, I)$ is a fully normalized Gaussian
- $p_\theta(x \mid z)$ is also properly normalized (typically Gaussian or Bernoulli)
- **The challenge isn't normalization—it's integration!** Despite each component being normalized, integrating across all $z$ remains intractable.

**Dimensionality observation: Isn't $z$ much smaller than $x$?** Excellent point! Typical latent dimensions are $z \in \mathbb{R}^{100}$ while images live in $x \in \mathbb{R}^{200,000}$. Why is integration still problematic?

**Three fundamental reasons**:

1. **Computation scales with dataset size**: Training requires computing $p_\theta(x^{(i)}) = \int p_\theta(x^{(i)} \mid z) p(z) dz$ for millions of training images. Even if each integral takes one second, the total computational burden becomes prohibitive.

2. **Gradient requirements**: Training demands $\frac{\partial}{\partial \theta} \int p_\theta(x \mid z) p(z) dz$. Differentiating through integrals is numerically expensive.

3. **Unknown relevance regions**: For any specific image $x$, the vast majority of $z$ values yield negligible $p_\theta(x \mid z)$. We're integrating predominantly over irrelevant regions! Here's the critical detail:

   **The relevance problem**: Consider a particular cat image $x_{\text{cat}}$. We want:
   
   $$p_\theta(x_{\text{cat}}) = \int p_\theta(x_{\text{cat}} \mid z) p(z) dz$$
   
   - The prior $p(z) = \mathcal{N}(0, I)$ samples $z$ uniformly from a standard Gaussian
   - Most random $z$ values encode entirely different images (dogs, vehicles, noise)!
   - For those $z$ values: $p_\theta(x_{\text{cat}} \mid z) \approx 0$ (the decoder indicates: "these instructions would never produce a cat")
   
   **Concrete example** with $z \in \mathbb{R}^{100}$:
   - $z_1 = [0.1, -0.3, 0.8, \dots]$ → decoder generates a cat image resembling $x_{\text{cat}}$ → $p_\theta(x_{\text{cat}} \mid z_1)$ is substantial
   - $z_2 = [5.2, -3.1, 2.7, \dots]$ → decoder generates a dog image → $p_\theta(x_{\text{cat}} \mid z_2) \approx 0$
   - $z_3 = [-2.1, 4.5, -1.8, \dots]$ → decoder generates noise → $p_\theta(x_{\text{cat}} \mid z_3) \approx 0$
   
   **The computational waste**: Naive Monte Carlo estimation by sampling $z \sim p(z)$ means:
   - 99.99% of samples contribute essentially nothing (yielding near-zero $p_\theta(x_{\text{cat}} \mid z)$)
   - Millions of samples would be required to accidentally sample the tiny relevant region
   - This exemplifies the **curse of high-dimensional integration**!
   
   **What we actually need**: The posterior $p_\theta(z \mid x_{\text{cat}})$ answers: "which instruction sets $z$ likely generated this specific cat image?"
   
   $$p_\theta(z \mid x_{\text{cat}}) = \frac{p_\theta(x_{\text{cat}} \mid z) p(z)}{p_\theta(x_{\text{cat}})}$$
   
   **The circular dependency**: Computing the posterior requires $p_\theta(x_{\text{cat}})$ as a denominator—precisely the quantity we're trying to compute!

**VAE's breakthrough**: Introduce an **approximate posterior** $q_\phi(z \mid x)$ (the encoder network):
- The encoder examines $x_{\text{cat}}$ and reports: "relevant $z$ values cluster around $\mu = [0.1, -0.3, 0.8, \dots]$"
- Now we can sample $z$ from this concentrated region where $p_\theta(x_{\text{cat}} \mid z)$ is actually meaningful
- This transforms Monte Carlo estimation from intractable to practical—we sample where it matters!

## The Variational Framework: Deriving ELBO

Rather than directly computing $p_\theta(x)$, VAEs construct a **tractable lower bound** through mathematical ingenuity.

**Foundation: Prior and Posterior in VAEs:**

Let's establish two essential probability concepts before proceeding:

**Prior $p(z)$**: Our initial beliefs about latent codes *without* observing data. VAEs typically use $p(z) = \mathcal{N}(0, I)$ (standard Gaussian). Think: "If I select random instructions without knowing my desired output, what distribution should I sample from?" We choose something simple and easy to sample.

**Posterior $p_\theta(z \mid x)$**: Our updated beliefs about latent codes *after* observing specific data $x$. It answers: "Having seen this particular cat image, which instruction sets $z$ probably generated it?" From Bayes' theorem:

$$p_\theta(z \mid x) = \frac{p_\theta(x \mid z) p(z)}{p_\theta(x)} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$

**Why the posterior is problematic**: Computing $p_\theta(z \mid x)$ requires $p_\theta(x)$ as a denominator—the very integral we're trying to avoid! This creates circular reasoning:
- We need the posterior to efficiently identify relevant $z$ values
- Computing the posterior requires knowing $p_\theta(x)$
- Computing $p_\theta(x)$ requires integrating over all $z$

**VAE's workaround**: Create an **approximate posterior** $q_\phi(z \mid x)$ (the encoder) that we can compute directly, then use it to construct a tractable training objective.

### Constructing the Approximate Posterior

Here's our strategy: since the true posterior $p_\theta(z \mid x) = \frac{p_\theta(x \mid z)p(z)}{p_\theta(x)}$ is uncomputable (due to that troublesome $p_\theta(x)$ denominator), let's build an approximation!

We'll design a **recognition network** (encoder) $q_\phi(z \mid x)$—a neural network that examines data $x$ and directly produces parameters describing which $z$ values are relevant. For instance, if $q_\phi$ is Gaussian, the encoder outputs location $\mu_\phi(x)$ and scale $\sigma_\phi^2(x)$.

**Essential property**: $q_\phi(z \mid x)$ must be a valid probability distribution over $z$. Let me clarify what this entails:

**Why must any probability distribution integrate to 1?**

Consider the fundamental requirement: if you have a distribution over possible outcomes, the probabilities of *all conceivable outcomes* must total 100% (or 1.0).

**VAE encoder example with Gaussian parameters**:

Suppose our encoder processes a cat image and outputs:
- Location: $\mu_\phi(x_{\text{cat}}) = [0.5, -0.2, 0.8, ...]$ 
- Scale: $\sigma_\phi^2(x_{\text{cat}}) = [0.1, 0.15, 0.2, ...]$

These parameters specify a Gaussian distribution $q_\phi(z \mid x_{\text{cat}}) = \mathcal{N}(\mu_\phi(x_{\text{cat}}), \sigma_\phi^2(x_{\text{cat}}))$.

**The normalization requirement**: Like *every* probability distribution, this Gaussian must integrate to unity:

$$\int q_\phi(z \mid x_{\text{cat}}) dz = \int \mathcal{N}(z; \mu_\phi(x_{\text{cat}}), \sigma_\phi^2(x_{\text{cat}})) dz = 1$$

**Interpretation**: "The probability that latent code $z$ takes *some value* (anywhere in latent space) equals 100%." 

We're not specifying which particular $z$ value is most probable (that's $\mu$'s role), we're stating that total probability across all possible $z$ values must equal unity. This holds for *every* Gaussian, regardless of its specific $\mu$ and $\sigma$ values!

**Fundamental reasoning**: *Some* latent code must exist—we cannot have a distribution where probabilities exceed 100% (impossible) or fall short of 100% (implying "perhaps nothing exists").

**For VAEs**: $q_\phi(z \mid x)$ represents "given image $x$, what's the probability distribution over latent codes $z$?" Since *some* latent code must be responsible, probabilities across all possible $z$ values must sum to unity:

$$\int q_\phi(z \mid x) dz = 1$$

**Critical distinction - Which variable are we integrating over?**

Notice integration is over $z$ (latent codes), NOT over $x$ (observations). This matters:

**For a single observation** $x_{\text{cat}}$:
- The encoder generates one distribution over latent space: $q_\phi(z \mid x_{\text{cat}})$
- This distribution integrates to 1 **over $z$**: $\int q_\phi(z \mid x_{\text{cat}}) dz = 1$
- Meaning: "For this cat image, total probability across all possible latent codes is 100%"

**For a different observation** $x_{\text{dog}}$:
- The encoder generates a *distinct* distribution: $q_\phi(z \mid x_{\text{dog}})$
- This also integrates to 1 **over $z$**: $\int q_\phi(z \mid x_{\text{dog}}) dz = 1$
- Meaning: "For this dog image, total probability across all possible latent codes is 100%"

**These are independent distributions!** Each observation has its own complete probability distribution over $z$ that integrates to unity. We're not combining probabilities across different observations—each observation defines its own conditional distribution over latent space.

**Analogy from statistics**: Consider height distributions:
- For males: $p(\text{height} \mid \text{male})$ integrates to 1 over all heights
- For females: $p(\text{height} \mid \text{female})$ integrates to 1 over all heights
- Different distributions (different means), yet each integrates to 1 over the height variable
- We don't combine them—they represent distinct conditional distributions

**This is automatic, not engineered—it's mandatory for $q_\phi$ to qualify as a probability distribution.** When designing the encoder architecture, we ensure it outputs parameters of a proper distribution (like Gaussian parameters), which automatically satisfies this requirement.

This normalization property (probabilities sum to 1 **over latent space**) becomes crucial in our derivation—we'll leverage it to cleverly introduce $q_\phi$ into our equations.

**The key insight: $q_\phi(z \mid x)$ directs our attention!**

This is what makes VAEs tractable! Recall our earlier problem: when computing $p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$, most $z$ values are irrelevant with negligible contributions.

**What $q_\phi(z \mid x)$ accomplishes**:
- For a specific cat image $x_{\text{cat}}$, it indicates: "Concentrate on $z$ values near $\mu = [0.5, -0.2, 0.8, ...]$"
- For a specific dog image $x_{\text{dog}}$, it indicates: "Concentrate on $z$ values near $\mu = [-0.3, 0.7, -0.1, ...]$"

**Without $q_\phi$**: We'd sample $z$ randomly from $p(z) = \mathcal{N}(0, I)$ and squander 99.99% of samples on irrelevant regions

**With $q_\phi$**: We sample $z$ from $q_\phi(z \mid x)$, which concentrates probability mass on $z$ values that actually matter for reconstructing $x$!

This explains why $q_\phi(z \mid x)$ is termed the **approximate posterior**—it approximates "which latent codes likely generated this observation?" While we cannot compute the true posterior $p_\theta(z \mid x)$ (requires intractable $p_\theta(x)$), we can learn $q_\phi(z \mid x)$ directly through neural networks!

**What comes next?** We have this approximate posterior $q_\phi(z \mid x)$, but how do we use it for training? This is where mathematical elegance enters.

### Deriving the Evidence Lower Bound (ELBO)

**First: What are we trying to maximize?**

This is crucial: In maximum likelihood training, we maximize the probability of *observed data* $x$, which is the **marginal likelihood** $p_\theta(x)$.

Recall the complete generative model:

$$p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$$

- $p_\theta(x \mid z)$ is the **decoder/likelihood**: "probability of data given latent code"
- $p(z)$ is the **prior**: "probability of latent code"
- $p_\theta(x)$ is the **marginal likelihood** or **evidence**: "total data probability" (integrating over all latent codes)

**Our objective**: Maximize $\log p_\theta(x)$ for each training observation. This says: "adjust model parameters $\theta$ to make observed data highly probable."

**The obstacle**: Computing $p_\theta(x)$ requires that intractable integral over $z$!

**Our approach**: Derive a tractable lower bound on $\log p_\theta(x)$ that we can maximize instead.

Let's begin with the quantity we actually want to maximize—the marginal log-likelihood:

$$\log p_\theta(x)$$

**The mathematical trick**: Multiply both sides by 1, but cleverly—using the fact that our approximate posterior $q_\phi(z \mid x)$ is a proper probability distribution (recall: $\int q_\phi(z \mid x) dz = 1$):

$$\log p_\theta(x) = \log p_\theta(x) \cdot \int q_\phi(z \mid x) dz$$

**Transforming to expectation form:**

Note that $\log p_\theta(x)$ is constant with respect to $z$ (doesn't depend on $z$). Therefore we can move it inside the integral:

$$\log p_\theta(x) \cdot \int q_\phi(z \mid x) dz = \int \log p_\theta(x) \cdot q_\phi(z \mid x) dz$$

**Recall expectation's definition**: For random variable $Z \sim q_\phi(z \mid x)$ and any function $f(Z)$:

$$\mathbb{E}_{q_\phi(z \mid x)} [f(Z)] = \int f(z) \cdot q_\phi(z \mid x) dz$$

Here, $f(Z) = \log p_\theta(x)$ is constant (independent of $z$). Applying expectation:

$$\int \log p_\theta(x) \cdot q_\phi(z \mid x) dz = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x) \right]$$

**Therefore**:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x) \right]$$

> **Note**: For comprehensive coverage of expectation, mathematical properties, and why constants move freely in/out of expectations, see our companion article: [Expected Value & Expectation: Mathematical Foundations]({{ site.baseurl }}{% link _posts/2026-01-01-expected-value-expectation-mathematical-foundations.md %}). That post explores discrete vs continuous expectations, linearity properties, and constant function expectations with detailed examples.

**Why is this transformation useful?** We've now expressed the log-likelihood as an expectation over our approximate posterior $q_\phi$. This enables powerful mathematical manipulations!

**Next step**: Inside this expectation, apply Bayes' rule to decompose $p_\theta(x)$:

$$p_\theta(x) = \frac{p_\theta(x, z)}{p_\theta(z \mid x)} = \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)}$$

Substituting:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)} \right]$$

**The pivotal insight**: Now multiply and divide by our approximate posterior $q_\phi(z \mid x)$ to introduce it:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{q_\phi(z \mid x)} \right]$$

**Separating the logarithms** into three distinct terms:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(z)}{q_\phi(z \mid x)} \right] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right]$$

**Recognizing the structure**: The final term is **KL divergence** between our approximate and true posteriors: $\text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x)) \geq 0$.

**Complete decomposition**:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z))}_{\text{ELBO: } \mathcal{L}(\theta, \phi; x)} + \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

**The key discovery**: Because KL divergence is always non-negative (equals zero only when distributions match perfectly), we have:

$$\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)$$

**We've derived the Evidence Lower Bound (ELBO)!** The first two terms provide a lower bound on log-likelihood. Crucially: these terms are tractable! We can now optimize this lower bound rather than the intractable true likelihood.

## Why This Eliminates the Partition Function

Let's examine the ELBO components:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z))$$

**First term: Reconstruction objective**
$$\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]$$

This is computable! We can:
1. Sample $z \sim q_\phi(z \mid x)$ (straightforward if $q_\phi$ is Gaussian)
2. Evaluate $\log p_\theta(x \mid z)$ (no partition function required!)
3. Estimate expectation via Monte Carlo

**Why does $\log p_\theta(x \mid z)$ avoid the partition function?**

This deserves careful explanation—let's clarify what each network outputs:

**Encoder output structure**: 
- Input: observation $x$ (e.g., a cat image)
- Output: parameters $\mu_\phi(x), \sigma_\phi(x)$ defining $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$
- This is a distribution **over latent codes $z$**

**Decoder output structure**:
- Input: latent code $z$
- Output: parameters $\mu_\theta(z)$ (and possibly $\sigma_\theta(z)$) defining $p_\theta(x \mid z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$
- This is a distribution **over observations $x$**

**Both produce normalized distributions, but over different spaces!** Now let's see why the decoder's distribution avoids partition functions:

**The decoder parameterizes a distribution over observations $x$**:

For continuous image data, the decoder network takes latent code $z$ as input and outputs parameters for a Gaussian distribution over possible images $x$:

$$p_\theta(x \mid z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$$

Here, $\mu_\theta(z)$ represents the decoder network's output—"the most probable image corresponding to latent code $z$."

**Critical distinction: Training vs Generation usage**

Many people get confused here! The decoder serves different purposes during training and generation:

**During TRAINING**:
- We possess actual training data $x_{\text{train}}$ (e.g., a real cat image)
- Encoder provides $z \sim q_\phi(z \mid x_{\text{train}})$
- Decoder outputs $\mu_\theta(z)$ (reconstructed image)
- **Key**: We evaluate $\log p_\theta(x_{\text{train}} \mid z)$ = "How probable is the *actual* training image under the Gaussian centered at $\mu_\theta(z)$?"
- We compute: $\log \mathcal{N}(x_{\text{train}}; \mu_\theta(z), \sigma^2 I)$ which quantifies reconstruction quality

**During GENERATION** (post-training):
- We sample a random latent code: $z \sim p(z) = \mathcal{N}(0, I)$
- Decoder outputs $\mu_\theta(z)$
- **Option 1 (deterministic)**: Use $x_{\text{new}} = \mu_\theta(z)$ directly (most common practice)
- **Option 2 (stochastic)**: Sample $x_{\text{new}} \sim \mathcal{N}(\mu_\theta(z), \sigma^2 I)$ (introduces noise around mean)

**Why Option 1 dominates**: Trained VAEs typically have very small $\sigma^2$. The mean $\mu_\theta(z)$ is already a high-quality image, and adding Gaussian noise would only blur it. Practically, we often just use the mean as the generated image.

**Analogy**: Think of $\mu_\theta(z)$ as the decoder stating "Given these instructions $z$, here's my exact recommendation," while the full Gaussian $\mathcal{N}(\mu_\theta(z), \sigma^2 I)$ states "Here's my recommendation, with some random variation."

**Essential insight: How do we generate complex images using only Gaussians?**

This is where the true elegance appears! The Gaussian distribution $p_\theta(x \mid z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$ is NOT claiming "images follow Gaussian distributions." Instead:

**The mean $\mu_\theta(z)$ is a powerful neural network's output!**

$$\mu_\theta(z) = \text{DecoderNetwork}_\theta(z)$$

This neural network can be arbitrarily sophisticated (many layers, convolutions, etc.) and can approximate ANY function. Therefore:

1. **Input**: Simple latent code $z \in \mathbb{R}^{100}$ (merely 100 numbers)
2. **Neural network**: Complex transformation with millions of parameters
3. **Output**: $\mu_\theta(z) \in \mathbb{R}^{200,000}$ (a complete image!)

**The Gaussian's role**: It models small **noise/uncertainty** around the neural network's output:

$$x \approx \mu_\theta(z) + \text{small Gaussian noise}$$

**Concrete example**: Suppose $z = [0.5, -0.2, 0.8, \ldots]$ encodes "smiling cat."
- The decoder neural network transforms this into $\mu_\theta(z)$ = a specific 256×256 cat image
- The Gaussian indicates: "Actual pixel values are probably very close to this image, perhaps off by a tiny amount due to natural variation"

**The crucial realization**: 
- **Complexity originates from the neural network**, not the Gaussian!
- The neural network can generate arbitrarily complex images
- The Gaussian merely states "there's minor randomness around the neural network's prediction"

**This parallels regression**: When fitting $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$:
- We're NOT claiming $y$ is Gaussian distributed
- We're stating $y$ equals a complex function $f(x)$ (the neural network) plus Gaussian noise
- All complexity resides in $f(x)$; the Gaussian captures measurement noise

**Practically**: Since $\sigma^2$ is typically tiny and the neural network $\mu_\theta(z)$ is very powerful, generated images are essentially just neural network outputs—complex, realistic images, not blurry Gaussian samples!

**Precise decoder output structure:**

For a 256×256 grayscale image (65,536 pixels):

$$p_\theta(x \mid z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$$

where:
- $\mu_\theta(z) \in \mathbb{R}^{65,536}$: One mean per pixel (decoder neural network output)
- $\sigma^2 I$: Diagonal covariance matrix
  - Each pixel has identical small variance $\sigma^2$ (typically $\sigma^2 \approx 0.01$ or smaller)
  - Pixels are **independent** (off-diagonal entries are zero)
  - No pixel correlations

**What this structure means**:
- Each pixel $x_i$ is independently sampled: $x_i \sim \mathcal{N}(\mu_i, \sigma^2)$
- The mean $\mu_i$ for pixel $i$ comes from the neural network
- Each pixel has tiny noise ($\sigma$) around its mean

**Single pixel example**:
- Decoder neural network predicts: "Pixel (10, 15) should have value $\mu = 0.73$"
- Actual pixel value: $x \sim \mathcal{N}(0.73, 0.01^2)$
- Most probable value: $x \approx 0.73$ (just use the mean!)

**Significant limitation**: The independence assumption ($\sigma^2 I$) means VAEs assume pixels don't correlate, which is a simplification. Real images exhibit strong pixel correlations (adjacent pixels tend toward similar values). This is one reason VAEs sometimes produce slightly blurrier images compared to GANs or diffusion models.

**The Gaussian is already normalized!** The multivariate Gaussian formula:

$$p_\theta(x \mid z) = \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2}\right)$$

The normalization constant $(2\pi\sigma^2)^{-D/2}$ is **known and computable**—no summation over all $x$ needed! It's determined by variance $\sigma^2$ and dimensionality $D$.

**For binary data** (like black/white images), the decoder outputs Bernoulli parameters:

$$p_\theta(x \mid z) = \prod_{i=1}^D \text{Bernoulli}(x_i; p_i(\theta, z))$$

where $p_i(\theta, z) = \text{sigmoid}(\text{decoder}(z)_i)$.

**Bernoulli distributions are also normalized!** For each pixel:

$$\text{Bernoulli}(x_i; p_i) = p_i^{x_i}(1-p_i)^{1-x_i}$$

This automatically sums to 1 over $x_i \in \{0, 1\}$: $p_i + (1-p_i) = 1$.

**Contrast with energy-based models**:

- **Energy-based**: $p(x) = \frac{\exp(-E(x))}{Z}$ where $Z = \sum_x \exp(-E(x))$ is intractable
  - We compute energy $E(x)$ but must normalize over *all possible $x$*
  
- **VAE decoder**: $p_\theta(x \mid z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$
  - We directly output parameters of a *known, normalized distribution*
  - The normalization constant is analytically known!

**In practice**: When evaluating $\log p_\theta(x \mid z)$, we simply compute:

$$\log p_\theta(x \mid z) = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2}$$

No summation over $x$, no intractable integral—just straightforward calculation!

**Second term: KL regularization**
$$\text{KL}(q_\phi(z \mid x) \| p(z))$$

When $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ and $p(z) = \mathcal{N}(0, I)$, this has a **closed-form solution**:

$$\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{j=1}^J \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)$$

**No integration required!** No partition function appears!

## The Complete VAE Algorithm

### Architecture

**Encoder** (Recognition Network):
- Input: observation $x$
- Output: parameters of $q_\phi(z \mid x)$, typically $\mu_\phi(x)$ and $\sigma_\phi(x)$

**Decoder** (Generative Network):
- Input: latent code $z$
- Output: parameters of $p_\theta(x \mid z)$, typically Gaussian mean or Bernoulli logits

### Training Objective

Maximize the ELBO (equivalently, minimize negative ELBO):

$$\max_{\theta, \phi} \mathbb{E}_{x \sim \text{data}} \left[ \mathcal{L}(\theta, \phi; x) \right]$$

$$= \max_{\theta, \phi} \mathbb{E}_{x \sim \text{data}} \left[ \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z)) \right]$$

### The Reparameterization Trick

**Challenge**: Backpropagation through the sampling operation $z \sim q_\phi(z \mid x)$ is problematic.

**Solution**: Reparameterize sampling:

Rather than: $z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$

Express as: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$

Now randomness resides in $\varepsilon$ (independent of $\phi$), enabling backpropagation through $\mu_\phi$ and $\sigma_\phi$!

### Training Loop

For each mini-batch of data $\{x^{(i)}\}$:

1. **Encode**: Compute $\mu_\phi(x^{(i)})$ and $\sigma_\phi(x^{(i)})$
2. **Sample**: $z^{(i)} = \mu_\phi(x^{(i)}) + \sigma_\phi(x^{(i)}) \odot \varepsilon^{(i)}$ where $\varepsilon^{(i)} \sim \mathcal{N}(0, I)$
3. **Decode**: Compute $p_\theta(x^{(i)} \mid z^{(i)})$
4. **Compute ELBO**: 
   $$\mathcal{L} = \log p_\theta(x^{(i)} \mid z^{(i)}) - \text{KL}(q_\phi(z \mid x^{(i)}) \| p(z))$$
5. **Backpropagate**: Update $\theta$ and $\phi$ to maximize $\mathcal{L}$

**No partition function anywhere!**

## Intuitive Understanding

### Interpreting the ELBO

The ELBO balances two competing objectives:

1. **Reconstruction objective** $\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]$:
   - Encourages decoder to reconstruct $x$ from latent codes sampled from $q_\phi$
   - Functions like an autoencoder: compress $x$ to $z$, then reconstruct

2. **KL regularization** $\text{KL}(q_\phi(z \mid x) \| p(z))$:
   - Prevents $q_\phi(z \mid x)$ from degenerating to a delta function
   - Forces latent codes to distribute like the prior $p(z) = \mathcal{N}(0, I)$
   - Ensures we can generate new data by sampling $z \sim p(z)$

### Why "Lower Bound"?

Remember:

$$\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

- $\mathcal{L}$ is what we maximize
- The KL term quantifies how far $q_\phi$ is from the true posterior
- Since KL ≥ 0, we have $\mathcal{L} \leq \log p_\theta(x)$

**When maximizing $\mathcal{L}$**, we simultaneously:
1. Push up the true log-likelihood $\log p_\theta(x)$
2. Make $q_\phi$ better approximate $p_\theta(z \mid x)$

### The Variational Gap

The difference:

$$\log p_\theta(x) - \mathcal{L}(\theta, \phi; x) = \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

is the **variational gap**. 

- If $q_\phi$ perfectly matches $p_\theta(z \mid x)$, the gap vanishes and we maximize true likelihood
- Practically, $q_\phi$ has limited expressiveness (often Gaussian), so some gap persists

## Why VAEs Produce Blurry Images

**The ELBO isn't the true likelihood!** 

When maximizing the ELBO:
- We're maximizing a **lower bound** on $\log p_\theta(x)$
- The decoder learns to reconstruct **on average** (expectation over $z \sim q_\phi(z \mid x)$)
- This produces **pixel-space averaging** → blurry images

**Contrasting with GANs**: GANs directly optimize sample quality (via adversarial training), not likelihood. This yields sharper samples but unstable training.

## Sampling from VAEs

Once trained, generating new samples is straightforward:

1. **Sample latent code**: $z \sim p(z) = \mathcal{N}(0, I)$
2. **Decode**: $x = \mu_\theta(z)$ or sample $x \sim p_\theta(x \mid z)$

**No partition function needed!** This is the entire point.

## Relationship to the Partition Function

Let's make the connection explicit. The true generative model is:

$$p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$$

If we attempted direct computation:
- For discrete $z$ with $K$ values: $p_\theta(x) = \sum_{k=1}^K p_\theta(x \mid z_k) p(z_k)$
- Tractable if $K$ is small (say, 10-100)
- But we'd need to evaluate all $K$ terms for every $x$

**VAEs bypass this integral!** Instead:
- The ELBO provides a tractable objective
- We never evaluate $p_\theta(x)$ exactly
- We can still train and generate samples

## VAE Advantages

1. **Tractable training**: No partition function, no adversarial dynamics
2. **Stable optimization**: Standard gradient ascent on ELBO
3. **Learned representations**: Encoder learns meaningful latent codes
4. **Probabilistic framework**: Can estimate (lower bound of) likelihoods
5. **Interpolation capability**: Smooth interpolation in latent space

## VAE Limitations

1. **Blurry samples**: Maximizing ELBO ≠ maximizing sample quality
2. **Variational gap**: Limited $q_\phi$ expressiveness means we're not maximizing true likelihood
3. **Posterior collapse**: Sometimes the model ignores latent variables
4. **Independence assumption**: Standard VAEs assume $p_\theta(x \mid z)$ factorizes (e.g., pixels independent given $z$)

## Modern Improvements

Several techniques enhance VAEs:

### 1. β-VAE (2017)

Weight the KL term:

$$\mathcal{L}_\beta = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \beta \cdot \text{KL}(q_\phi(z \mid x) \| p(z))$$

- $\beta > 1$: Encourages disentangled representations
- Trade-off: Better disentanglement but worse reconstruction

### 2. Hierarchical VAEs

Multiple latent variable layers:

$$p_\theta(x, z_1, z_2, \dots, z_L) = p(z_L) \prod_{l=1}^{L-1} p_\theta(z_l \mid z_{l+1}) \cdot p_\theta(x \mid z_1)$$

More expressive, reduces variational gap.

### 3. Normalizing Flow VAEs

Use invertible transformations for more expressive $q_\phi(z \mid x)$:

$$q_\phi(z \mid x) = q_0(\varepsilon) \left\lvert \det \frac{\partial f^{-1}}{\partial z} \right\rvert$$

where $z = f(\varepsilon)$ and $\varepsilon \sim q_0$.

### 4. VQ-VAE (Vector Quantized VAE)

Discrete latent codes with learned codebook:

$$z \in \{e_1, e_2, \dots, e_K\}$$

Combines VAE ideas with discrete representations.

## The Broader Landscape

VAEs represent one approach among several for avoiding partition functions:

| Approach | How It Avoids Z | Trade-off |
|----------|-----------------|-----------|
| **VAEs** | Optimize ELBO instead of likelihood | Blurry samples, variational gap |
| **GANs** | No explicit density model | No likelihood, unstable training |
| **Normalizing Flows** | Change of variables formula | Restricted architectures |
| **Diffusion Models** | Score matching objective | Slow sampling |
| **Autoregressive** | Factorize $p(x) = \prod p(x_i \mid x_{<i})$ | Sequential generation |

Each approach makes different trade-offs between:
- **Sample quality** vs **likelihood evaluation**
- **Training stability** vs **expressiveness**
- **Sampling speed** vs **model flexibility**

## Key Takeaways

1. **VAEs avoid computing Z** by optimizing a tractable lower bound (ELBO) rather than true likelihood

2. **The ELBO decomposes** into reconstruction and regularization terms, both tractable

3. **The reparameterization trick** enables backpropagation through stochastic sampling

4. **Trade-off**: We gain tractable training but sacrifice exact likelihood and sample sharpness

5. **The variational gap** between ELBO and true likelihood depends on $q_\phi$ expressiveness

6. **Modern improvements** (hierarchical VAEs, normalizing flows, β-VAE) address various limitations

VAEs elegantly demonstrate that we don't always need exact computation—sometimes a good approximation suffices!

## Further Reading

- **Original VAE paper**: Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
- **Tutorial**: Carl Doersch (2016), "Tutorial on Variational Autoencoders"
- **β-VAE**: Higgins et al. (2017), "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- **VQ-VAE**: van den Oord et al. (2017), "Neural Discrete Representation Learning"
- **Understanding disentanglement**: Locatello et al. (2019), "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"
