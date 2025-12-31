---
layout: post
title: "How Variational Autoencoders Avoid Computing the Partition Function"
description: "A deep dive into how VAEs sidestep the intractable partition function Z through the Evidence Lower Bound (ELBO), making generative modeling tractable."
tags: [deep-learning, generative-models, vae, probability, machine-learning]
---

*This post explores one of the most elegant solutions to the partition function problem, building on [The Normalization Constant Problem: Why Computing Z Is So Hard]({{ site.baseurl }}{% link _posts/2025-12-24-normalization-constant-problem.md %}). For broader context, see [Why Discriminative Learning Dominated First]({{ site.baseurl }}{% link _posts/2025-12-25-why-discriminative-learning-came-first.md %}).*

## The Problem We're Solving

Recall from our discussion of the partition function problem: when we want to model a probability distribution $p(x)$ over high-dimensional data (like images), we face an intractable normalization constant:

$$p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}$$

where $Z_\theta = \sum_x \tilde{p}_\theta(x)$ requires summing over an astronomically large number of configurations (e.g., $10^{157,826}$ for 256×256 images).

**The consequence**: We can't:
1. Evaluate $p_\theta(x)$ (need $Z_\theta$)
2. Train via maximum likelihood (need to compute gradients of $\log Z_\theta$)
3. Sample efficiently (most methods need to evaluate probabilities)

**Variational Autoencoders (VAEs)** solve this through a brilliant insight: **don't compute $p(x)$ directly—optimize a lower bound instead!**

## The Key Insight: Latent Variable Models

VAEs introduce **latent variables** $z$—hidden representations that capture the underlying structure of the data.

**The generative story**:
1. Sample a latent code: $z \sim p(z)$ (typically $\mathcal{N}(0, I)$)
2. Generate data from latent: $x \sim p_\theta(x \mid z)$

**In everyday terms**: Think of $z$ as a "recipe" or "blueprint" for creating data:
- **Step 1**: Pick a random recipe from a cookbook (sample $z$ from a simple distribution, like rolling dice with Gaussian probabilities)
- **Step 2**: Follow the recipe to create an image (the decoder neural network reads the recipe $z$ and outputs an image $x$)

For example, if generating faces:
- $z$ might encode: [smiling=0.8, glasses=0.2, age=25, hair_color=brown, ...]
- The decoder takes these instructions and paints a face matching them
- Different random $z$ values → different faces

**The key insight**: Instead of trying to model all possible images directly (which requires the intractable $Z$), we model the *process* of creating images from simple recipes.

**How do we get the full probability $p_\theta(x)$?**

We have a joint distribution over data and latent codes:

$$p_\theta(x, z) = p_\theta(x \mid z) p(z)$$

This says: "the probability of both $x$ and $z$ occurring together equals the probability of $z$ times the probability of $x$ given $z$."

**Quick probability refresher**:
- **Joint distribution** $p(x, z)$: Probability of both $x$ AND $z$ occurring together. Think: "What's the probability someone is tall (x) AND has blue eyes (z)?"
- **Marginal distribution** $p(x)$: Probability of just $x$, regardless of $z$. Think: "What's the probability someone is tall, with any eye color?"
- **Marginalization**: To get the marginal from the joint, sum over all possibilities of the other variable: $p(x) = \sum_z p(x, z)$ (or $\int p(x, z) dz$ in continuous case)

**But we want just $p_\theta(x)$** (the marginal probability of the data). To get this, we **marginalize out** (sum/integrate over) all possible latent codes:

$$p_\theta(x) = \int p_\theta(x, z) dz = \int p_\theta(x \mid z) p(z) dz$$

**Intuition**: To find the total probability of generating image $x$, we consider all possible recipes $z$ that could have produced it:
- Each recipe $z$ has some probability $p(z)$ of being picked
- Given recipe $z$, there's some probability $p_\theta(x \mid z)$ of producing image $x$
- Sum up contributions from all recipes: $\int p_\theta(x \mid z) p(z) dz$

**Important note: Are these normalized probabilities?** YES! Unlike the unnormalized $\tilde{p}(x)$ in energy-based models:
- $p(z) = \mathcal{N}(0, I)$ is a proper, normalized Gaussian distribution
- $p_\theta(x \mid z)$ is also a normalized distribution (e.g., Gaussian or Bernoulli)
- **The problem isn't normalization—it's the integral!** Even though each term is properly normalized, integrating over all possible $z$ is intractable.

**Wait, isn't $z$ much lower dimensional than $x$?** YES! Great observation! Typically $z \in \mathbb{R}^{100}$ while $x \in \mathbb{R}^{200,000}$ (for images). So why is the integral still hard?

**Three reasons**:

1. **We need it for EVERY data point**: During training, for each of millions of images, we'd need to compute $p_\theta(x^{(i)}) = \int p_\theta(x^{(i)} \mid z) p(z) dz$. Even if one integral takes 1 second, doing this millions of times is prohibitive.

2. **We need gradients through it**: Training requires $\frac{\partial}{\partial \theta} \int p_\theta(x \mid z) p(z) dz$. We need to differentiate through the integral, which is expensive numerically.

3. **We don't know which $z$ values matter**: For a given image $x$, most $z$ values give tiny $p_\theta(x \mid z)$. We're integrating over mostly irrelevant regions! Let me explain this crucial point in detail:

   **The problem**: Consider a specific cat image $x_{\text{cat}}$. We want to compute:
   
   $$p_\theta(x_{\text{cat}}) = \int p_\theta(x_{\text{cat}} \mid z) p(z) dz$$
   
   - The prior $p(z) = \mathcal{N}(0, I)$ says: sample $z$ uniformly from a standard Gaussian
   - But most random $z$ values correspond to completely different images (dogs, cars, noise)!
   - For those $z$ values: $p_\theta(x_{\text{cat}} \mid z) \approx 0$ (decoder says: "this $z$ would never produce a cat")
   
   **Example**: Imagine $z \in \mathbb{R}^{100}$:
   - $z_1 = [0.1, -0.3, 0.8, \dots]$ → decoder outputs a cat image similar to $x_{\text{cat}}$ → $p_\theta(x_{\text{cat}} \mid z_1)$ is high
   - $z_2 = [5.2, -3.1, 2.7, \dots]$ → decoder outputs a dog image → $p_\theta(x_{\text{cat}} \mid z_2) \approx 0$
   - $z_3 = [-2.1, 4.5, -1.8, \dots]$ → decoder outputs noise → $p_\theta(x_{\text{cat}} \mid z_3) \approx 0$
   
   **The inefficiency**: If we naively sample $z \sim p(z)$ to approximate the integral via Monte Carlo:
   - 99.99% of samples will be irrelevant (give near-zero $p_\theta(x_{\text{cat}} \mid z)$)
   - We'd need millions of samples to accidentally hit the tiny relevant region
   - This is the **curse of high-dimensional integration**!
   
   **What we really need**: The posterior $p_\theta(z \mid x_{\text{cat}})$ tells us: "which $z$ values are likely to have produced this specific cat image?"
   
   $$p_\theta(z \mid x_{\text{cat}}) = \frac{p_\theta(x_{\text{cat}} \mid z) p(z)}{p_\theta(x_{\text{cat}})}$$
   
   **Chicken-and-egg problem**: To compute the posterior, we need $p_\theta(x_{\text{cat}})$ in the denominator—the very thing we're trying to compute!

**This is where VAEs shine**: They introduce an **approximate posterior** $q_\phi(z \mid x)$ (the encoder):
- The encoder looks at $x_{\text{cat}}$ and says: "the relevant $z$ values are around $\mu = [0.1, -0.3, 0.8, \dots]$"
- Now we can sample $z$ from this focused region where $p_\theta(x_{\text{cat}} \mid z)$ is actually significant
- This makes Monte Carlo estimation tractable—we're sampling from where it matters!

## The Variational Trick: ELBO

Here's where VAEs become clever. Instead of computing $p_\theta(x)$ directly, we derive a **lower bound** that is tractable.

**Understanding Prior and Posterior in VAEs:**

Before we proceed, let's clarify two fundamental concepts:

**Prior $p(z)$**: This represents our initial belief about latent codes *before* seeing any data. In VAEs, we typically use $p(z) = \mathcal{N}(0, I)$ (standard Gaussian). Think of it as: "If I pick a random recipe without knowing what dish I want, what distribution should I sample from?" We choose a simple distribution that's easy to sample from.

**Posterior $p_\theta(z \mid x)$**: This represents our belief about latent codes *after* observing specific data $x$. It answers: "Given that I see this specific cat image, what recipes $z$ likely produced it?" Using Bayes' rule:

$$p_\theta(z \mid x) = \frac{p_\theta(x \mid z) p(z)}{p_\theta(x)} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$

**The posterior problem**: Computing $p_\theta(z \mid x)$ requires $p_\theta(x)$ in the denominator—the intractable integral we're trying to avoid! This creates a circular dependency:
- We need the posterior to efficiently sample relevant $z$ values
- But computing the posterior requires knowing $p_\theta(x)$
- And computing $p_\theta(x)$ requires integrating over all $z$

**VAE's solution**: Introduce an **approximate posterior** $q_\phi(z \mid x)$ (the encoder) that we can actually compute, and use it to derive a tractable objective.

### Building the Approximate Posterior

So here's our strategy: since we can't compute the true posterior $p_\theta(z \mid x) = \frac{p_\theta(x \mid z)p(z)}{p_\theta(x)}$ (it has that pesky $p_\theta(x)$ in the denominator), let's build our own approximate version!

We'll create a **recognition model** (encoder) $q_\phi(z \mid x)$—a neural network that looks at data $x$ and directly outputs parameters describing which $z$ values are relevant. For example, if $q_\phi$ is Gaussian, the encoder outputs mean $\mu_\phi(x)$ and variance $\sigma_\phi^2(x)$.

**Crucially**: $q_\phi(z \mid x)$ is a proper probability distribution over $z$. Let me explain what this means:

**Why does any probability distribution integrate to 1?**

Think about it intuitively: if you have a distribution over possible outcomes, the probabilities of *all possible outcomes* must add up to 100% (or 1 in decimal).

**Example with Gaussian parameters (exactly what VAE encoders do)**:

Suppose our encoder looks at a cat image and outputs:
- Mean: $\mu_\phi(x_{\text{cat}}) = [0.5, -0.2, 0.8, ...]$ 
- Variance: $\sigma_\phi^2(x_{\text{cat}}) = [0.1, 0.15, 0.2, ...]$

These parameters define a Gaussian distribution $q_\phi(z \mid x_{\text{cat}}) = \mathcal{N}(\mu_\phi(x_{\text{cat}}), \sigma_\phi^2(x_{\text{cat}}))$.

**The key property**: This Gaussian distribution, like *all* probability distributions, must integrate to 1:

$$\int q_\phi(z \mid x_{\text{cat}}) dz = \int \mathcal{N}(z; \mu_\phi(x_{\text{cat}}), \sigma_\phi^2(x_{\text{cat}})) dz = 1$$

**What this means**: "The probability that the latent code $z$ takes *some value* (anywhere in latent space) is 100%." 

We're not saying which specific $z$ value is most likely (that's what $\mu$ tells us), we're saying that the total probability across all possible $z$ values must equal 1. This is true for *any* Gaussian, regardless of what $\mu$ and $\sigma$ are!

**Intuition**: *Something* must be the latent code—we can't have a distribution where probabilities sum to more than 100% (impossible) or less than 100% (implying "maybe nothing exists").

**For our VAE**: $q_\phi(z \mid x)$ represents "given image $x$, what's the probability distribution over latent codes $z$?" Since *some* latent code must be responsible, the probabilities across all possible $z$ values must sum to 1:

$$\int q_\phi(z \mid x) dz = 1$$

**Critical clarification - What variable are we integrating over?**

Notice we're integrating over $z$ (the latent code), NOT over $x$ (the data). This is crucial:

**For a single data point** $x_{\text{cat}}$:
- The encoder produces one distribution over latent codes: $q_\phi(z \mid x_{\text{cat}})$
- This distribution integrates to 1 **over $z$**: $\int q_\phi(z \mid x_{\text{cat}}) dz = 1$
- Meaning: "For this cat image, the total probability across all possible latent codes is 100%"

**For a different data point** $x_{\text{dog}}$:
- The encoder produces a *different* distribution: $q_\phi(z \mid x_{\text{dog}})$
- This also integrates to 1 **over $z$**: $\int q_\phi(z \mid x_{\text{dog}}) dz = 1$
- Meaning: "For this dog image, the total probability across all possible latent codes is 100%"

**These are separate distributions!** Each data point gets its own distribution over $z$ that integrates to 1. We're not summing probabilities across different data points—each data point has its own complete probability distribution over latent space.

**Analogy**: Think of it like heights of people:
- For men: $p(\text{height} \mid \text{male})$ integrates to 1 over all possible heights
- For women: $p(\text{height} \mid \text{female})$ integrates to 1 over all possible heights
- These are two different distributions (different means), but each integrates to 1 over the height variable
- We don't add them together—they describe different conditional distributions

**This is not something we choose or compute—it's a fundamental requirement for $q_\phi$ to be a valid probability distribution.** When we design the encoder network, we ensure it outputs parameters of a proper distribution (like a Gaussian), which automatically satisfies this property.

This property (that probabilities sum to 1 **over latent codes**) will be key to our derivation—we'll use it as a clever way to introduce $q_\phi$ into our equations.

**The brilliant insight: $q_\phi(z \mid x)$ tells us WHERE to look!**

This is the key to making VAEs work! Remember our problem from earlier: when computing $p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$, most $z$ values are irrelevant and contribute nothing.

**What $q_\phi(z \mid x)$ does**:
- For a specific cat image $x_{\text{cat}}$, it says: "Focus on $z$ values around $\mu = [0.5, -0.2, 0.8, ...]$"
- For a specific dog image $x_{\text{dog}}$, it says: "Focus on $z$ values around $\mu = [-0.3, 0.7, -0.1, ...]$"

**Without $q_\phi$**: We'd sample $z$ randomly from $p(z) = \mathcal{N}(0, I)$ and waste 99.99% of samples on irrelevant regions

**With $q_\phi$**: We sample $z$ from $q_\phi(z \mid x)$, which concentrates probability mass on the $z$ values that actually matter for reconstructing $x$!

This is why $q_\phi(z \mid x)$ is called the **approximate posterior**—it approximates "which latent codes likely produced this specific data point?" Even though we can't compute the true posterior $p_\theta(z \mid x)$ (requires intractable $p_\theta(x)$), we can learn $q_\phi(z \mid x)$ directly with a neural network!

**Now what?** We have this approximate posterior $q_\phi(z \mid x)$, but how do we use it to train the model? This is where the mathematical magic happens.

### Deriving the Evidence Lower Bound (ELBO)

**Wait, what are we actually trying to maximize?**

This is a crucial point: In maximum likelihood training, we want to maximize the probability of the *observed data* $x$, which is the **marginal likelihood** $p_\theta(x)$.

Recall from earlier that the full generative model is:

$$p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$$

- $p_\theta(x \mid z)$ is the **decoder/likelihood**: "probability of data given latent code"
- $p(z)$ is the **prior**: "probability of latent code"
- $p_\theta(x)$ is the **marginal likelihood** or **evidence**: "total probability of data" (integrating over all possible latent codes)

**What we want**: Maximize $\log p_\theta(x)$ for each observed data point in our training set. This says: "adjust the model parameters $\theta$ so that the observed data becomes highly probable."

**The problem**: Computing $p_\theta(x)$ requires that intractable integral over $z$!

**The strategy**: We'll derive a tractable lower bound on $\log p_\theta(x)$ that we can maximize instead.

Let's start with the quantity we actually want to maximize—the marginal log-likelihood:

$$\log p_\theta(x)$$

**The clever trick**: Let's multiply both sides by 1, but in a sneaky way—using the fact that our approximate posterior $q_\phi(z \mid x)$ is a proper probability distribution (recall: $\int q_\phi(z \mid x) dz = 1$):

$$\log p_\theta(x) = \log p_\theta(x) \cdot \int q_\phi(z \mid x) dz$$

**How do we rewrite this as an expectation?**

Notice that $\log p_\theta(x)$ doesn't depend on $z$ (it's just a constant with respect to $z$). So we can move it inside the integral:

$$\log p_\theta(x) \cdot \int q_\phi(z \mid x) dz = \int \log p_\theta(x) \cdot q_\phi(z \mid x) dz$$

**Now recall the definition of expectation**: For a random variable $Z \sim q_\phi(z \mid x)$ and any function $f(Z)$:

$$\mathbb{E}_{q_\phi(z \mid x)} [f(Z)] = \int f(z) \cdot q_\phi(z \mid x) dz$$

In our case, $f(Z) = \log p_\theta(x)$ is a constant function (doesn't depend on $z$). Applying the expectation definition:

$$\int \log p_\theta(x) \cdot q_\phi(z \mid x) dz = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x) \right]$$

**So we have**:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x) \right]$$

> **Note**: For a deeper dive into expectation, its mathematical properties, and why constants can be pulled in/out of expectations, see our companion post: [Expected Value & Expectation: Mathematical Foundations]({{ site.baseurl }}{% link _posts/2026-01-01-expected-value-expectation-mathematical-foundations.md %}). This post covers discrete vs continuous expectations, linearity of expectation, and expectation of constant functions with detailed examples.

**Why is this useful?** Because now we've expressed the log-likelihood as an expectation over our approximate posterior $q_\phi$. This will allow us to manipulate the equation in powerful ways!

**Next move**: Inside this expectation, let's apply Bayes' rule to decompose $p_\theta(x)$:

$$p_\theta(x) = \frac{p_\theta(x, z)}{p_\theta(z \mid x)} = \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)}$$

Substituting this back:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)} \right]$$

**Here comes the key insight**: We now multiply and divide by our approximate posterior $q_\phi(z \mid x)$ to bring it into the picture:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{q_\phi(z \mid x)} \right]$$

**Rearranging the logarithms**, we can split this into three separate terms:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(z)}{q_\phi(z \mid x)} \right] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right]$$

**Recognizing the pattern**: That last term is actually the **KL divergence** between our approximate posterior and the true posterior: $\text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x)) \geq 0$.

**Putting it all together**:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z))}_{\text{ELBO: } \mathcal{L}(\theta, \phi; x)} + \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

**The breakthrough**: Since KL divergence is always non-negative (it's zero only when the two distributions are identical), we have:

$$\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)$$

**We've discovered the Evidence Lower Bound (ELBO)!** The first two terms give us a lower bound on the log-likelihood. And here's the beautiful part: these terms are tractable to compute! We can now optimize this lower bound instead of the intractable true likelihood.

## Why This Solves the Z Problem

Let's examine the ELBO:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z))$$

**Term 1: Reconstruction term**
$$\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]$$

This is tractable! We can:
1. Sample $z \sim q_\phi(z \mid x)$ (easy if $q_\phi$ is Gaussian)
2. Evaluate $\log p_\theta(x \mid z)$ (no partition function needed!)
3. Estimate the expectation via Monte Carlo

**Why does $\log p_\theta(x \mid z)$ not need a partition function?**

This is crucial to understand, and it's important to clarify what both networks output:

**What the encoder outputs**: 
- Input: data $x$ (e.g., a cat image)
- Output: parameters $\mu_\phi(x), \sigma_\phi(x)$ for the distribution $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$
- This is a distribution **over latent codes $z$**

**What the decoder outputs**:
- Input: latent code $z$
- Output: parameters $\mu_\theta(z)$ (and possibly $\sigma_\theta(z)$) for the distribution $p_\theta(x \mid z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$
- This is a distribution **over data $x$**

**Both output normalized distributions, but over different variables!** Now let's see why the decoder's distribution doesn't need a partition function:

**The decoder parameterizes a distribution over data $x$**:

For image data (continuous), the decoder neural network takes latent code $z$ as input and outputs parameters for a Gaussian distribution over possible images $x$:

$$p_\theta(x \mid z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$$

Here, $\mu_\theta(z)$ is the output of the decoder network—it represents "the most likely image corresponding to latent code $z$."

**Important distinction: Training vs Generation**

This is where many people get confused! The decoder is used differently during training and generation:

**During TRAINING**:
- We have actual training data $x_{\text{train}}$ (e.g., a real cat image)
- Encoder gives us $z \sim q_\phi(z \mid x_{\text{train}})$
- Decoder outputs $\mu_\theta(z)$ (reconstructed image)
- **Key**: We evaluate $\log p_\theta(x_{\text{train}} \mid z)$ = "How likely is the *actual* training image under the Gaussian centered at $\mu_\theta(z)$?"
- We compute: $\log \mathcal{N}(x_{\text{train}}; \mu_\theta(z), \sigma^2 I)$ which measures reconstruction quality

**During GENERATION** (after training):
- We sample a random latent code: $z \sim p(z) = \mathcal{N}(0, I)$
- Decoder outputs $\mu_\theta(z)$
- **Option 1 (deterministic)**: Use $x_{\text{new}} = \mu_\theta(z)$ directly (most common in practice)
- **Option 2 (stochastic)**: Sample $x_{\text{new}} \sim \mathcal{N}(\mu_\theta(z), \sigma^2 I)$ (adds noise around the mean)

**Why Option 1 is common**: The variance $\sigma^2$ is often very small in trained VAEs. The mean $\mu_\theta(z)$ is already a good image, and adding Gaussian noise would just blur it. So in practice, we often just use the mean as the generated image.

**Analogy**: Think of $\mu_\theta(z)$ as the decoder saying "Given this recipe $z$, here's the exact dish I recommend," while the full Gaussian $\mathcal{N}(\mu_\theta(z), \sigma^2 I)$ says "Here's my recommendation, plus/minus some random variation."

**Critical insight: How do we generate complex images with just a Gaussian?**

This is where the magic happens! The Gaussian distribution $p_\theta(x \mid z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$ is NOT saying "images are Gaussian distributed." Instead:

**The mean $\mu_\theta(z)$ is the output of a powerful neural network!**

$$\mu_\theta(z) = \text{DecoderNetwork}_\theta(z)$$

This neural network can be arbitrarily complex (multiple layers, convolutions, etc.) and can approximate ANY function. So:

1. **Input**: Simple latent code $z \in \mathbb{R}^{100}$ (just 100 numbers)
2. **Neural network**: Complex transformation with millions of parameters
3. **Output**: $\mu_\theta(z) \in \mathbb{R}^{200,000}$ (a full image!)

**What the Gaussian does**: It models small **noise/uncertainty** around the neural network's output:

$$x \approx \mu_\theta(z) + \text{small Gaussian noise}$$

**Example**: Suppose $z = [0.5, -0.2, 0.8, \ldots]$ represents "cat with smile."
- The decoder neural network transforms this into $\mu_\theta(z)$ = a specific 256×256 cat image
- The Gaussian says: "The actual pixel values are probably very close to this image, maybe off by a tiny bit due to natural variation"

**Key realization**: 
- **The complexity comes from the neural network**, not the Gaussian!
- The neural network can generate arbitrarily complex images
- The Gaussian just says "there's a bit of randomness around the neural network's prediction"

**This is similar to regression**: When we fit $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$:
- We're NOT saying $y$ is Gaussian distributed
- We're saying $y$ equals a complex function $f(x)$ (the neural network) plus Gaussian noise
- All the complexity is in $f(x)$; the Gaussian is just measurement noise

**In practice**: Since $\sigma^2$ is typically very small and the neural network $\mu_\theta(z)$ is very powerful, the generated images are essentially just the neural network outputs—complex, realistic images, not blurry Gaussian samples!

**Precise structure of the decoder's Gaussian output:**

For a 256×256 grayscale image (65,536 pixels):

$$p_\theta(x \mid z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$$

where:
- $\mu_\theta(z) \in \mathbb{R}^{65,536}$: One mean value per pixel (output of decoder neural network)
- $\sigma^2 I$: Diagonal covariance matrix
  - Each pixel has the same small variance $\sigma^2$ (typically $\sigma^2 \approx 0.01$ or smaller)
  - Pixels are **independent** (off-diagonal entries are 0)
  - No correlation between pixels

**What this means**:
- Each pixel $x_i$ is independently sampled: $x_i \sim \mathcal{N}(\mu_i, \sigma^2)$
- The mean $\mu_i$ for pixel $i$ comes from the neural network
- Each pixel just has a tiny bit of noise ($\sigma$) around its mean

**Example for one pixel**:
- Decoder neural network says: "Pixel (10, 15) should have value $\mu = 0.73$"
- Actual pixel value: $x \sim \mathcal{N}(0.73, 0.01^2)$
- Most likely value: $x \approx 0.73$ (just use the mean!)

**Important limitation**: The independence assumption ($\sigma^2 I$) means VAEs assume pixels don't correlate, which is a simplification. Real images have strong pixel correlations (nearby pixels tend to have similar values). This is one reason why VAEs sometimes produce slightly blurry images compared to other generative models like GANs or diffusion models.

**The Gaussian is already normalized!** The formula for a multivariate Gaussian is:

$$p_\theta(x \mid z) = \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2}\right)$$

The normalization constant $(2\pi\sigma^2)^{-D/2}$ is **known and computable**—we don't need to sum over all $x$ to compute it! It's determined by the variance $\sigma^2$ and dimensionality $D$.

**For binary data** (like black/white images), the decoder outputs Bernoulli parameters:

$$p_\theta(x \mid z) = \prod_{i=1}^D \text{Bernoulli}(x_i; p_i(\theta, z))$$

where $p_i(\theta, z) = \text{sigmoid}(\text{decoder}(z)_i)$.

**Bernoulli distributions are also already normalized!** For each pixel:

$$\text{Bernoulli}(x_i; p_i) = p_i^{x_i}(1-p_i)^{1-x_i}$$

This automatically sums to 1 over $x_i \in \{0, 1\}$: $p_i + (1-p_i) = 1$.

**The key difference from energy-based models**:

- **Energy-based**: $p(x) = \frac{\exp(-E(x))}{Z}$ where $Z = \sum_x \exp(-E(x))$ is intractable
  - We compute an energy $E(x)$ but then need to normalize over *all possible $x$*
  
- **VAE decoder**: $p_\theta(x \mid z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$
  - We directly output parameters of a *known, normalized distribution*
  - The normalization constant is analytically known!

**In practice**: When we evaluate $\log p_\theta(x \mid z)$, we just compute:

$$\log p_\theta(x \mid z) = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2}$$

No summation over $x$, no intractable integral—just a simple calculation!

**Term 2: KL divergence**
$$\text{KL}(q_\phi(z \mid x) \| p(z))$$

When $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ and $p(z) = \mathcal{N}(0, I)$, this has a **closed form**:

$$\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{j=1}^J \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)$$

**No integration needed!** No partition function appears!

## The Complete VAE Algorithm

### Architecture

**Encoder** (Recognition Model):
- Input: data $x$
- Output: parameters of $q_\phi(z \mid x)$, typically $\mu_\phi(x)$ and $\sigma_\phi(x)$

**Decoder** (Generative Model):
- Input: latent code $z$
- Output: parameters of $p_\theta(x \mid z)$, typically mean of Gaussian or logits for Bernoulli

### Training Objective

Maximize the ELBO (minimize negative ELBO):

$$\max_{\theta, \phi} \mathbb{E}_{x \sim \text{data}} \left[ \mathcal{L}(\theta, \phi; x) \right]$$

$$= \max_{\theta, \phi} \mathbb{E}_{x \sim \text{data}} \left[ \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z)) \right]$$

### The Reparameterization Trick

**Problem**: We need to backpropagate through the sampling operation $z \sim q_\phi(z \mid x)$.

**Solution**: Reparameterize the sampling:

Instead of: $z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$

Write: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$

Now the randomness is in $\varepsilon$ (independent of $\phi$), and we can backpropagate through $\mu_\phi$ and $\sigma_\phi$!

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

### What the ELBO Means

The ELBO has two competing objectives:

1. **Reconstruction term** $\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]$:
   - Encourages the decoder to reconstruct $x$ from latent codes sampled from $q_\phi$
   - Like an autoencoder: compress $x$ to $z$, then reconstruct

2. **KL regularization** $\text{KL}(q_\phi(z \mid x) \| p(z))$:
   - Prevents $q_\phi(z \mid x)$ from collapsing to a delta function
   - Forces latent codes to be distributed like the prior $p(z) = \mathcal{N}(0, I)$
   - Ensures we can sample new data by sampling $z \sim p(z)$

### Why Is It Called a "Lower Bound"?

Remember:

$$\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

- $\mathcal{L}$ is what we maximize
- The KL term measures how far $q_\phi$ is from the true posterior
- Since KL ≥ 0, we have $\mathcal{L} \leq \log p_\theta(x)$

**When we maximize $\mathcal{L}$**, we're simultaneously:
1. Pushing up the true log-likelihood $\log p_\theta(x)$
2. Making $q_\phi$ a better approximation to $p_\theta(z \mid x)$

### The Variational Gap

The difference:

$$\log p_\theta(x) - \mathcal{L}(\theta, \phi; x) = \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

is called the **variational gap**. 

- If $q_\phi$ perfectly matches $p_\theta(z \mid x)$, the gap is zero and we're maximizing the true likelihood
- In practice, $q_\phi$ has limited expressiveness (often a Gaussian), so there's always some gap

## Why VAEs Produce Blurry Images

**The ELBO is not the true likelihood!** 

When we maximize the ELBO:
- We're maximizing a **lower bound** on $\log p_\theta(x)$
- The decoder learns to reconstruct **on average** (expectation over $z \sim q_\phi(z \mid x)$)
- This leads to **averaging in pixel space** → blurry images

**Contrast with GANs**: GANs directly optimize sample quality (through adversarial training), not likelihood. This leads to sharper samples but unstable training.

## Sampling from VAEs

Once trained, generating new samples is easy:

1. **Sample latent code**: $z \sim p(z) = \mathcal{N}(0, I)$
2. **Decode**: $x = \mu_\theta(z)$ or sample $x \sim p_\theta(x \mid z)$

**No partition function needed!** This is the whole point.

## Relationship to the Partition Function

Let's make the connection explicit. The true generative model is:

$$p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$$

If we tried to compute this directly:
- For discrete $z$ with $K$ values: $p_\theta(x) = \sum_{k=1}^K p_\theta(x \mid z_k) p(z_k)$
- Still tractable if $K$ is small (say, 10-100)
- But we'd need to evaluate all $K$ terms for every $x$

**VAEs don't compute this integral!** Instead:
- The ELBO provides a tractable objective
- We never evaluate $p_\theta(x)$ exactly
- We can still train and generate samples

## Advantages of VAEs

1. **Tractable training**: No partition function, no adversarial dynamics
2. **Stable optimization**: Standard gradient ascent on ELBO
3. **Learned representations**: The encoder learns meaningful latent codes
4. **Probabilistic**: Can estimate (lower bound of) likelihoods
5. **Interpolation**: Smooth interpolation in latent space

## Limitations of VAEs

1. **Blurry samples**: Maximizing ELBO ≠ maximizing sample quality
2. **Variational gap**: Limited expressiveness of $q_\phi$ means we're not maximizing true likelihood
3. **Posterior collapse**: Sometimes the model ignores the latent variables
4. **Independence assumption**: Standard VAEs assume $p_\theta(x \mid z)$ factorizes (e.g., pixels are independent given $z$)

## Modern Improvements

Several techniques have been developed to improve VAEs:

### 1. β-VAE (2017)

Add a weight to the KL term:

$$\mathcal{L}_\beta = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \beta \cdot \text{KL}(q_\phi(z \mid x) \| p(z))$$

- $\beta > 1$: Encourages disentangled representations
- Trade-off: Better disentanglement but worse reconstruction

### 2. Hierarchical VAEs

Use multiple layers of latent variables:

$$p_\theta(x, z_1, z_2, \dots, z_L) = p(z_L) \prod_{l=1}^{L-1} p_\theta(z_l \mid z_{l+1}) \cdot p_\theta(x \mid z_1)$$

More expressive, reduces variational gap.

### 3. Normalizing Flow VAEs

Use invertible transformations to make $q_\phi(z \mid x)$ more expressive:

$$q_\phi(z \mid x) = q_0(\varepsilon) \left\lvert \det \frac{\partial f^{-1}}{\partial z} \right\rvert$$

where $z = f(\varepsilon)$ and $\varepsilon \sim q_0$.

### 4. VQ-VAE (Vector Quantized VAE)

Use discrete latent codes with a learned codebook:

$$z \in \{e_1, e_2, \dots, e_K\}$$

Combines ideas from VAEs and discrete representations.

## The Bigger Picture

VAEs represent one of several strategies for avoiding the partition function:

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

1. **VAEs avoid computing Z** by optimizing a tractable lower bound (ELBO) instead of the true likelihood

2. **The ELBO decomposes** into reconstruction and regularization terms, both tractable

3. **The reparameterization trick** enables backpropagation through stochastic sampling

4. **Trade-off**: We get tractable training but sacrifice exact likelihood and sample sharpness

5. **The variational gap** between ELBO and true likelihood depends on the expressiveness of $q_\phi$

6. **Modern improvements** (hierarchical VAEs, normalizing flows, β-VAE) address various limitations

VAEs elegantly demonstrate that we don't always need to compute intractable quantities—sometimes a good approximation is enough!

## Further Reading

- **Original VAE paper**: Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
- **Tutorial**: Carl Doersch (2016), "Tutorial on Variational Autoencoders"
- **β-VAE**: Higgins et al. (2017), "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- **VQ-VAE**: van den Oord et al. (2017), "Neural Discrete Representation Learning"
- **Understanding disentanglement**: Locatello et al. (2019), "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"
