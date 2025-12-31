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

The full probability becomes:

$$p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$$

**Still looks hard!** This integral is also intractable because we'd need to integrate over all possible latent codes $z$.

## The Variational Trick: ELBO

Here's where VAEs become clever. Instead of computing $p_\theta(x)$ directly, we derive a **lower bound** that is tractable.

### Step 1: Introduce an Approximate Posterior

We introduce a **recognition model** (encoder) $q_\phi(z \mid x)$ that approximates the true posterior $p_\theta(z \mid x)$.

**Why?** Because the true posterior $p_\theta(z \mid x) = \frac{p_\theta(x \mid z)p(z)}{p_\theta(x)}$ also requires computing $p_\theta(x)$—the very thing we're trying to avoid!

### Step 2: Derive the Evidence Lower Bound (ELBO)

Starting with the log-likelihood we want to maximize:

$$\log p_\theta(x)$$

**Multiply by 1** (in a clever way):

$$\log p_\theta(x) = \log p_\theta(x) \cdot \int q_\phi(z \mid x) dz$$

Since $\int q_\phi(z \mid x) dz = 1$ (it's a probability distribution).

**Bring the log inside** (using properties of expectations):

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x) \right]$$

**Apply Bayes' rule** inside the expectation:

$$p_\theta(x) = \frac{p_\theta(x, z)}{p_\theta(z \mid x)} = \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)}$$

Therefore:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)} \right]$$

**Introduce $q_\phi$** by multiplying and dividing:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z)p(z)}{p_\theta(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{q_\phi(z \mid x)} \right]$$

**Rearrange**:

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p(z)}{q_\phi(z \mid x)} \right] + \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right]$$

The last term is the **KL divergence** $\text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x)) \geq 0$.

**Final result**:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z))}_{\text{ELBO: } \mathcal{L}(\theta, \phi; x)} + \text{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

Since KL divergence is non-negative:

$$\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)$$

**This is the Evidence Lower Bound (ELBO)!**

## Why This Solves the Z Problem

Let's examine the ELBO:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right] - \text{KL}(q_\phi(z \mid x) \| p(z))$$

**Term 1: Reconstruction term**
$$\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) \right]$$

This is tractable! We can:
1. Sample $z \sim q_\phi(z \mid x)$ (easy if $q_\phi$ is Gaussian)
2. Evaluate $\log p_\theta(x \mid z)$ (no partition function needed!)
3. Estimate the expectation via Monte Carlo

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
