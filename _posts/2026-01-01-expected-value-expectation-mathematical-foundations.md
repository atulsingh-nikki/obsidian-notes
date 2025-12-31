---
layout: post
title: "Expected Value & Expectation: Mathematical Foundations"
description: "A comprehensive guide to understanding expectation—from discrete dice rolls to continuous distributions, with applications in machine learning and probability theory."
tags: [probability, statistics, mathematics, machine-learning]
---

*This post provides the mathematical foundation for understanding expectation, a concept central to probability theory and machine learning. This explanation supports the ELBO derivation in [How Variational Autoencoders Avoid Computing the Partition Function]({{ site.baseurl }}{% link _posts/2026-01-01-how-vaes-avoid-computing-partition-function.md %}).*

## What is Expectation?

**Intuitive idea**: The expectation (or expected value) of a random variable is its **average value**, weighted by how likely each outcome is.

Think of it as: "If I repeated this random experiment infinitely many times and took the average of all outcomes, what would I get?"

## Discrete Random Variables

### Definition

For a discrete random variable $X$ that can take values $\{x_1, x_2, \dots, x_n\}$ with probabilities $\{p_1, p_2, \dots, p_n\}$:

$$\mathbb{E}[X] = \sum_{i=1}^n x_i \cdot p_i$$

**In words**: Multiply each possible value by its probability, then sum them all up.

### Example 1: Fair Six-Sided Die

Rolling a fair die, what's the expected value?

**Possible outcomes**: $X \in \{1, 2, 3, 4, 5, 6\}$

**Probabilities**: Each outcome has probability $\frac{1}{6}$

**Calculation**:

$$\mathbb{E}[X] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6}$$

$$= \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5$$

**Interpretation**: On average, rolling a die gives 3.5 (even though you can never actually roll 3.5!).

### Example 2: Biased Coin

A coin shows Heads with probability 0.7 and Tails with probability 0.3. If Heads gives you \$10 and Tails gives you \$0, what's the expected payoff?

**Values**: $X \in \{10, 0\}$

**Probabilities**: $p(\text{Heads}) = 0.7$, $p(\text{Tails}) = 0.3$

**Calculation**:

$$\mathbb{E}[X] = 10 \cdot 0.7 + 0 \cdot 0.3 = 7$$

**Interpretation**: On average, you expect to win \$7 per flip.

## Continuous Random Variables

### Definition

For a continuous random variable $X$ with probability density function (PDF) $p(x)$:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) dx$$

**Key difference from discrete**: We integrate instead of sum, because there are infinitely many possible values.

### Example 3: Uniform Distribution

$X$ is uniformly distributed on $[0, 1]$: $p(x) = 1$ for $x \in [0, 1]$, and $p(x) = 0$ otherwise.

**Calculation**:

$$\mathbb{E}[X] = \int_0^1 x \cdot 1 \, dx = \left[ \frac{x^2}{2} \right]_0^1 = \frac{1}{2}$$

**Interpretation**: The average value is right in the middle of the interval.

### Example 4: Standard Normal Distribution

$X \sim \mathcal{N}(0, 1)$ has PDF $p(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$.

**Result**: $\mathbb{E}[X] = 0$ (by symmetry—the distribution is centered at 0)

**Verification** (for completeness):

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = 0$$

The integral equals zero because the integrand is an odd function ($f(-x) = -f(x)$) integrated over a symmetric interval.

## Expectation of Functions

Often we want the expected value of some function $g(X)$ rather than $X$ itself.

### For Discrete Variables

$$\mathbb{E}[g(X)] = \sum_i g(x_i) \cdot p_i$$

### For Continuous Variables

$$\mathbb{E}[g(X)] = \int g(x) \cdot p(x) dx$$

**Crucial point**: We don't need to find the distribution of $g(X)$—we can compute the expectation directly using the distribution of $X$!

### Example 5: Expected Square

For a die roll $X$, what's $\mathbb{E}[X^2]$?

$$\mathbb{E}[X^2] = \sum_{i=1}^6 i^2 \cdot \frac{1}{6} = \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} \approx 15.17$$

Note: $\mathbb{E}[X^2] \neq (\mathbb{E}[X])^2 = (3.5)^2 = 12.25$ (in general, expectation is not preserved under nonlinear transformations!)

## Properties of Expectation

### 1. Linearity

**Property**: $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$

**Proof for discrete case**:

$$\mathbb{E}[aX + b] = \sum_i (ax_i + b) p_i = a\sum_i x_i p_i + b\sum_i p_i = a\mathbb{E}[X] + b$$

(Since $\sum_i p_i = 1$)

**Example**: If $X$ is a die roll ($\mathbb{E}[X] = 3.5$), then:

$$\mathbb{E}[2X + 3] = 2 \cdot 3.5 + 3 = 10$$

### 2. Expectation of a Constant

**Property**: If $c$ is a constant, then $\mathbb{E}[c] = c$

**Proof for discrete case**:

$$\mathbb{E}[c] = \sum_i c \cdot p_i = c \sum_i p_i = c \cdot 1 = c$$

**Proof for continuous case**:

$$\mathbb{E}[c] = \int c \cdot p(x) dx = c \int p(x) dx = c \cdot 1 = c$$

**Intuition**: A constant is a constant—it doesn't vary with the random variable, so its "average" is just itself!

**This is the property used in VAEs**: When we have $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x)]$, since $\log p_\theta(x)$ doesn't depend on $z$, it's a constant with respect to the expectation over $z$, so:

$$\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x)] = \log p_\theta(x)$$

### 3. Expectation of Sum

**Property**: $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$

This holds **even if $X$ and $Y$ are dependent**! (This is remarkable and very useful.)

### 4. Expectation of Product (for independent variables)

**Property**: If $X$ and $Y$ are independent, then:

$$\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y]$$

**Warning**: This generally does NOT hold if $X$ and $Y$ are dependent!

## Moving Constants In and Out of Expectations

This is crucial for VAE derivations!

### Rule 1: Constants can be pulled out

$$\mathbb{E}[c \cdot X] = c \cdot \mathbb{E}[X]$$

**Example**: $\mathbb{E}[5X] = 5\mathbb{E}[X]$

### Rule 2: Constants with respect to the expectation variable

If $f(x)$ is constant with respect to the random variable $Z$ (i.e., doesn't depend on $Z$):

$$\mathbb{E}_{Z}[f(x)] = f(x)$$

**Example in VAE context**:

$$\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x)] = \log p_\theta(x)$$

Because $\log p_\theta(x)$ doesn't contain $z$—it's constant as far as the expectation over $z$ is concerned!

## Detailed Example: Constant Function Expectation

This is the exact situation in VAEs, so let's be very explicit.

**Setup**: We have a distribution $q_\phi(z \mid x)$ over latent variable $z$, and a quantity $\log p_\theta(x)$ that doesn't depend on $z$.

**Question**: What is $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x)]$?

**Step-by-step**:

$$\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x)] = \int \log p_\theta(x) \cdot q_\phi(z \mid x) dz$$

Since $\log p_\theta(x)$ doesn't involve $z$, we can factor it out:

$$= \log p_\theta(x) \int q_\phi(z \mid x) dz$$

Since $q_\phi(z \mid x)$ is a probability distribution:

$$= \log p_\theta(x) \cdot 1 = \log p_\theta(x)$$

**Conclusion**: Taking the expectation of a constant just gives back the constant!

**Visual intuition**: Imagine you're averaging the number 5 over many trials. No matter what random outcomes you get, you're always looking at 5, so the average is 5.

## Why Do We Use Expectations in Machine Learning?

### 1. Handling Randomness

Many ML algorithms involve randomness (sampling, stochastic gradient descent, dropout). Expectations let us reason about average behavior.

### 2. Monte Carlo Estimation

We can estimate expectations by sampling:

$$\mathbb{E}[g(X)] \approx \frac{1}{N} \sum_{i=1}^N g(x_i) \quad \text{where } x_i \sim p(x)$$

This is how VAEs estimate the ELBO—sample a few $z$ values and average!

### 3. Variational Methods

In VAEs and other variational methods, we manipulate expectations to derive tractable objectives.

### 4. Loss Functions

Many loss functions are expectations:

$$\text{Loss}(\theta) = \mathbb{E}_{(x,y) \sim \text{data}} [\ell(f_\theta(x), y)]$$

We minimize the expected loss over the data distribution.

## Common Mistakes to Avoid

### Mistake 1: $\mathbb{E}[g(X)] \neq g(\mathbb{E}[X])$ (in general)

For nonlinear $g$, expectation doesn't "go through" the function.

**Counterexample**: For a die, $\mathbb{E}[X^2] = 15.17$ but $(\mathbb{E}[X])^2 = 12.25$.

**Exception**: For linear functions $g(x) = ax + b$, we do have $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$.

### Mistake 2: Forgetting to integrate/sum over the correct variable

$$\mathbb{E}_{q(z \mid x)}[f(z)] = \int f(z) q(z \mid x) dz$$

The integration is over $z$, not over $x$!

### Mistake 3: Assuming independence when computing $\mathbb{E}[XY]$

$$\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y] \text{ ONLY if } X \perp Y$$

If $X$ and $Y$ are dependent, this formula doesn't hold!

## Connection to VAE Derivation

In the VAE ELBO derivation, we use these key facts:

1. **Starting point**:
   $$\log p_\theta(x) = \log p_\theta(x) \cdot \int q_\phi(z \mid x) dz$$
   (Multiply by 1, since $\int q_\phi(z \mid x) dz = 1$)

2. **Move constant inside integral**:
   $$= \int \log p_\theta(x) \cdot q_\phi(z \mid x) dz$$
   (Since $\log p_\theta(x)$ doesn't depend on $z$)

3. **Recognize as expectation**:
   $$= \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x)]$$

4. **Apply constant expectation property**:
   $$= \log p_\theta(x)$$

This circular-looking manipulation is valid precisely because we're taking the expectation of a constant function!

## Key Takeaways

1. **Expectation is probability-weighted average**: Sum (or integrate) values times their probabilities

2. **Discrete**: $\mathbb{E}[X] = \sum_i x_i p_i$

3. **Continuous**: $\mathbb{E}[X] = \int x \cdot p(x) dx$

4. **Linearity**: $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$

5. **Constant expectation**: $\mathbb{E}[c] = c$

6. **Constants pull out**: $\mathbb{E}[c \cdot X] = c \cdot \mathbb{E}[X]$

7. **Expectation of functions**: $\mathbb{E}[g(X)] = \int g(x) p(x) dx$ (no need to find distribution of $g(X)$)

8. **Monte Carlo**: Expectations can be estimated by sampling and averaging

Understanding these properties is essential for deriving and understanding modern machine learning algorithms like VAEs, policy gradients in reinforcement learning, and expectation-maximization algorithms!

## Further Reading

- **Classic reference**: "Introduction to Probability" by Bertsekas & Tsitsiklis
- **For ML applications**: "Pattern Recognition and Machine Learning" by Bishop (Chapter 1)
- **Rigorous treatment**: "Probability Theory: The Logic of Science" by Jaynes
- **Interactive visualization**: Try the Central Limit Theorem visualizations to see how averages of random variables behave
