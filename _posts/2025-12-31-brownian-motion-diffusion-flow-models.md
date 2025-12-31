---
layout: post
title: "From Brownian Motion to Modern Generative Models: The Stochastic Foundation of Diffusion and Flow Models"
description: "Exploring how 19th-century physics of random particle motion became the mathematical foundation for cutting-edge AI systems that generate images, audio, and video."
tags: [stochastic-processes, deep-learning, diffusion-models, normalizing-flows, generative-ai]
---

*This post builds on concepts from [Stochastic Processes and the Art of Sampling Uncertainty]({{ site.baseurl }}{% link _posts/2025-02-21-stochastic-processes-and-sampling.md %}). For a broader introduction to differential equations (ODEs, PDEs, SDEs), see [The Landscape of Differential Equations]({{ site.baseurl }}{% link _posts/2025-12-29-differential-equations-ode-pde-sde.md %}). For the essential mathematical framework of stochastic calculus, see [Itô Calculus: Why We Need New Rules for SDEs]({{ site.baseurl }}{% link _posts/2025-12-28-ito-calculus-stochastic-differential-equations.md %}). Familiarity with basic probability and calculus is helpful but not required.*

## Table of Contents

- [The Remarkable Journey: From Pollen to Pixels](#the-remarkable-journey-from-pollen-to-pixels)
- [Brownian Motion: The Mathematics of Randomness](#brownian-motion-the-mathematics-of-randomness)
  - [Historical Context](#historical-context)
  - [Mathematical Definition](#mathematical-definition)
  - [Key Properties](#key-properties)
  - [The Wiener Process](#the-wiener-process)
- [Stochastic Differential Equations: Dynamics Under Uncertainty](#stochastic-differential-equations-dynamics-under-uncertainty)
  - [From ODEs to SDEs](#from-odes-to-sdes)
  - [Itô Calculus: The Mathematics of Noise](#itô-calculus-the-mathematics-of-noise)
  - [Fokker-Planck Equation](#fokker-planck-equation)
- [Diffusion Models: Generative AI Through Forward and Reverse Diffusion](#diffusion-models-generative-ai-through-forward-and-reverse-diffusion)
  - [The Core Insight](#the-core-insight)
  - [Forward Diffusion Process](#forward-diffusion-process)
  - [Reverse Diffusion Process](#reverse-diffusion-process)
  - [Score-Based Generative Models](#score-based-generative-models)
  - [Training Objective](#training-objective)
- [Normalizing Flows: Continuous Transformations and Probability](#normalizing-flows-continuous-transformations-and-probability)
  - [The Change of Variables Formula](#the-change-of-variables-formula)
  - [Flow-Based Models](#flow-based-models)
  - [Continuous Normalizing Flows](#continuous-normalizing-flows)
  - [Neural ODEs](#neural-odes)
- [The Deep Connection: From Brownian Motion to Modern AI](#the-deep-connection-from-brownian-motion-to-modern-ai)
  - [Unified Framework](#unified-framework)
  - [SDE vs ODE Perspectives](#sde-vs-ode-perspectives)
  - [The Probability Flow ODE](#the-probability-flow-ode)
- [Practical Implementations and Applications](#practical-implementations-and-applications)
  - [DDPM: Denoising Diffusion Probabilistic Models](#ddpm-denoising-diffusion-probabilistic-models)
  - [Score-Based Models](#score-based-models)
  - [Flow Matching](#flow-matching)
  - [Rectified Flows](#rectified-flows)
- [Computer Vision Applications](#computer-vision-applications)
- [Comparing the Approaches](#comparing-the-approaches)
- [Key Takeaways](#key-takeaways)
- [Further Reading](#further-reading)

## The Remarkable Journey: From Pollen to Pixels

In 1827, botanist Robert Brown observed pollen grains suspended in water under a microscope, jittering in seemingly random motion. This phenomenon, now called **Brownian motion**, puzzled scientists for decades until Einstein's 1905 paper provided a molecular explanation: the visible particles were being bombarded by invisible water molecules in thermal motion.

Fast forward to 2020-2025, and the same mathematical framework describing those dancing pollen grains now powers **diffusion models**—the AI systems behind DALL-E, Stable Diffusion, and Imagen that generate photorealistic images from text. The connection isn't metaphorical: modern diffusion models literally implement stochastic differential equations derived from Brownian motion theory.

This post explores that remarkable intellectual journey, showing how abstract mathematics from physics became the foundation for one of deep learning's most successful paradigms.

## Brownian Motion: The Mathematics of Randomness

### Historical Context

Brownian motion represents one of mathematics' most beautiful examples of order emerging from chaos:

- **1827**: Robert Brown observes the phenomenon
- **1905**: Einstein derives the diffusion equation from molecular collisions
- **1923**: Norbert Wiener provides the first rigorous mathematical construction
- **1940s-50s**: Itô develops stochastic calculus
- **2015-2020**: Score-based and diffusion models emerge in machine learning

### Mathematical Definition

A **standard Brownian motion** (or Wiener process) $W(t)$ is a continuous-time stochastic process with four defining properties:

1. **$W(0) = 0$** with probability 1
2. **Independent increments**: For $s < t$, $W(t) - W(s)$ is independent of all $W(u)$ for $u \leq s$
3. **Stationary Gaussian increments**: $W(t) - W(s) \sim \mathcal{N}(0, t - s)$
4. **Continuous paths**: $W(t)$ is continuous in $t$ (almost surely)

These simple axioms give rise to remarkable properties including self-similarity, the Markov property, and quadratic variation $(dW)^2 = dt$—the foundation of stochastic calculus.

**For a detailed exploration with visualizations**, see [Mathematical Properties of Brownian Motion: A Visual Guide]({{ site.baseurl }}{% link _posts/2025-12-30-mathematical-properties-brownian-motion.md %}), which includes interactive demonstrations of continuous paths, independent increments, Gaussian distributions, scaling behavior, and quadratic variation.

## Stochastic Differential Equations: Dynamics Under Uncertainty

*For a comprehensive introduction to the landscape of differential equations (ODEs, PDEs, SDEs), see [The Landscape of Differential Equations]({{ site.baseurl }}{% link _posts/2025-12-29-differential-equations-ode-pde-sde.md %}). For a deep dive into why we need new calculus rules for SDEs, see [Itô Calculus: Why We Need New Rules for SDEs]({{ site.baseurl }}{% link _posts/2025-12-28-ito-calculus-stochastic-differential-equations.md %}).*

### From ODEs to SDEs

Deterministic dynamics are described by **ordinary differential equations (ODEs)**:

$$\frac{dx}{dt} = f(x, t)$$

When we add noise, we get **stochastic differential equations (SDEs)**:

$$dx = f(x, t) \, dt + g(x, t) \, dW$$

Here:
- $f(x, t)$ is the **drift coefficient** (deterministic trend)
- $g(x, t)$ is the **diffusion coefficient** (noise strength)
- $dW$ represents increments of Brownian motion

### Itô Calculus: The Mathematics of Noise

Because Brownian motion is non-differentiable, we need special rules. **Itô's lemma** is the stochastic chain rule (for a detailed explanation, see [Itô Calculus]({{ site.baseurl }}{% link _posts/2025-12-28-ito-calculus-stochastic-differential-equations.md %})):

For $Y(t) = h(X(t), t)$ where $dX = f \, dt + g \, dW$:

$$dY = \left(\frac{\partial h}{\partial t} + f \frac{\partial h}{\partial x} + \frac{1}{2} g^2 \frac{\partial^2 h}{\partial x^2}\right) dt + g \frac{\partial h}{\partial x} \, dW$$

The crucial difference from ordinary calculus is the **second-order term** $\frac{1}{2} g^2 \frac{\partial^2 h}{\partial x^2}$, arising from the quadratic variation $(dW)^2 = dt$.

**Example**: For $Y = X^2$ where $dX = \sigma \, dW$:

$$dY = \sigma^2 \, dt + 2 \sigma X \, dW$$

The $\sigma^2 \, dt$ term has no analog in ordinary calculus!

### Fokker-Planck Equation

If $X(t)$ follows the SDE $dx = f(x,t) \, dt + g(x,t) \, dW$, then the probability density $p(x, t)$ evolves according to the **Fokker-Planck equation**:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}[f(x,t) p] + \frac{1}{2} \frac{\partial^2}{\partial x^2}[g^2(x,t) p]$$

This is a **deterministic PDE** governing the evolution of the **probability distribution** of a stochastic process. This duality—stochastic trajectories vs deterministic probability evolution—is central to understanding diffusion models.

## Diffusion Models: Generative AI Through Forward and Reverse Diffusion

### The Core Insight

Diffusion models are built on a brilliant observation:

**Forward process**: Gradually add Gaussian noise to data until it becomes pure noise (easy to define)

**Reverse process**: Learn to reverse this noising process to generate data from noise (the hard part)

The mathematics of Brownian motion provides the rigorous framework for both.

### Forward Diffusion Process

Start with data $\mathbf{x}\_0 \sim p\_{\text{data}}(\mathbf{x})$ (e.g., natural images). Define a forward SDE:

$$d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} \, dt + \sqrt{\beta(t)} \, d\mathbf{W}$$

where $\beta(t) > 0$ is a **noise schedule** and $\mathbf{W}$ is multi-dimensional Brownian motion.

**Properties**:
- The drift term $-\frac{1}{2}\beta(t) \mathbf{x}$ shrinks the signal
- The diffusion term $\sqrt{\beta(t)} \, d\mathbf{W}$ adds noise
- As $t \to \infty$, $\mathbf{x}\_t \to \mathcal{N}(0, I)$ regardless of $\mathbf{x}\_0$

**Discrete-time version** (DDPM):
$$\mathbf{x}_t = \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1} + \sqrt{\beta_t} \, \varepsilon_t$$

where $\varepsilon\_t \sim \mathcal{N}(0, I)$.

**Closed-form solution**: Thanks to Gaussian properties,

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon$$

where $\bar{\alpha}\_t = \prod\_{s=1}^t (1 - \beta\_s)$ and $\varepsilon \sim \mathcal{N}(0, I)$.

### Reverse Diffusion Process

The key theorem (Anderson, 1982) states that the reverse-time SDE is:

$$d\mathbf{x} = \left[-\frac{1}{2} \beta(t) \mathbf{x} - \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + \sqrt{\beta(t)} \, d\bar{\mathbf{W}}$$

where:
- $\bar{\mathbf{W}}$ is reverse-time Brownian motion
- $\nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$ is the **score function**

**The challenge**: We don't know $p\_t(\mathbf{x})$, so we don't know its score!

**The solution**: Train a neural network $\mathbf{s}\_\theta(\mathbf{x}, t) \approx \nabla\_{\mathbf{x}} \log p\_t(\mathbf{x})$ to estimate the score.

### Score-Based Generative Models

The **score function** $\nabla\_{\mathbf{x}} \log p(\mathbf{x})$ points toward higher probability regions:

- High magnitude where data is unlikely (far from modes)
- Points toward data manifold
- Enables sampling via **Langevin dynamics**:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_k) + \sqrt{\epsilon} \, \mathbf{z}_k$$

### Training Objective

**Score matching**: Train $\mathbf{s}\_\theta(\mathbf{x}, t)$ to match the true score by minimizing:

$$\mathbb{E}_{t \sim U(0,T)} \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \lambda(t) \left\| \mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \mid \mathbf{x}_0) \right\|^2 \right]$$

**Key insight**: We can compute $\nabla\_{\mathbf{x}\_t} \log p(\mathbf{x}\_t \mid \mathbf{x}\_0)$ analytically since $p(\mathbf{x}\_t \mid \mathbf{x}\_0) = \mathcal{N}(\sqrt{\bar{\alpha}\_t} \mathbf{x}\_0, (1 - \bar{\alpha}\_t) I)$:

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0}{1 - \bar{\alpha}_t} = -\frac{\varepsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

This reduces to **denoising**: predict the noise $\varepsilon$!

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \varepsilon} \left[ \lambda(t) \left\| \varepsilon_\theta(\mathbf{x}_t, t) - \varepsilon \right\|^2 \right]$$

## Normalizing Flows: Continuous Transformations and Probability

### The Change of Variables Formula

If $\mathbf{z} \sim p\_Z(\mathbf{z})$ and $\mathbf{x} = f(\mathbf{z})$ where $f$ is invertible:

$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \left| \det \frac{\partial f^{-1}}{\partial \mathbf{x}} \right|$$

Equivalently:
$$\log p\_X(\mathbf{x}) = \log p\_Z(\mathbf{z}) - \log \left| \det \frac{\partial f}{\partial \mathbf{z}} \right|$$

**Intuition**: Probability mass is conserved, but the Jacobian determinant accounts for volume expansion/contraction.

### Flow-Based Models

A **normalizing flow** is a sequence of invertible transformations:

$$\mathbf{z}_0 \xrightarrow{f_1} \mathbf{z}_1 \xrightarrow{f_2} \cdots \xrightarrow{f_K} \mathbf{z}_K = \mathbf{x}$$

starting from a simple base distribution (typically $\mathcal{N}(0, I)$).

**Log-likelihood**:
$$\log p\_X(\mathbf{x}) = \log p\_Z(\mathbf{z}\_0) - \sum\_{k=1}^K \log \left| \det \frac{\partial f\_k}{\partial \mathbf{z}\_{k-1}} \right|$$

**Advantages**:
- Exact likelihood computation
- Exact sampling: sample $\mathbf{z}\_0 \sim \mathcal{N}(0, I)$ and apply $f\_1 \circ \cdots \circ f\_K$
- Exact inference: apply $f\_K^{-1} \circ \cdots \circ f\_1^{-1}$

**Challenges**:
- Designing architectures with tractable Jacobians
- Balancing expressiveness vs computational cost

### Continuous Normalizing Flows

Instead of discrete transformations, consider a **continuous** transformation via an ODE:

$$\frac{d\mathbf{z}}{dt} = f_\theta(\mathbf{z}(t), t), \quad \mathbf{z}(0) = \mathbf{x}_0, \quad \mathbf{z}(1) = \mathbf{x}_1$$

The log-likelihood evolves according to the **instantaneous change of variables**:

$$\frac{d \log p_t(\mathbf{z}(t))}{dt} = -\text{tr}\left(\frac{\partial f_\theta}{\partial \mathbf{z}}\right)$$

**Total change**:
$$\log p_1(\mathbf{x}_1) = \log p_0(\mathbf{x}_0) - \int_0^1 \text{tr}\left(\frac{\partial f_\theta(\mathbf{z}(t), t)}{\partial \mathbf{z}}\right) dt$$

### Neural ODEs

**Neural ODEs** (Chen et al., 2018) parameterize $f_\theta$ with a neural network:

$$\frac{d\mathbf{z}}{dt} = \text{NeuralNet}_\theta(\mathbf{z}(t), t)$$

**Training**: Backpropagate through the ODE solver using the adjoint method, avoiding storing intermediate states.

**Connection to ResNets**: A ResNet block $\mathbf{z}_{t+1} = \mathbf{z}_t + f(\mathbf{z}_t)$ can be viewed as an Euler discretization of an ODE with step size 1.

## The Deep Connection: From Brownian Motion to Modern AI

### Unified Framework

The groundbreaking insight (Song et al., 2021): **diffusion models and continuous normalizing flows are two sides of the same coin**.

Every diffusion SDE has a corresponding **probability flow ODE** that generates the same marginal distributions $p\_t(\mathbf{x})$ without stochasticity:

**Forward SDE**:
$$d\mathbf{x} = f(\mathbf{x}, t) \, dt + g(t) \, d\mathbf{W}$$

**Equivalent probability flow ODE**:
$$d\mathbf{x} = \left[f(\mathbf{x}, t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$$

### SDE vs ODE Perspectives

| **Property** | **SDE (Diffusion)** | **ODE (Flow)** |
|---|---|---|
| **Stochasticity** | Stochastic trajectories | Deterministic trajectories |
| **Marginals** | $p\_t(\mathbf{x})$ | Same $p\_t(\mathbf{x})$ |
| **Sampling** | Multiple runs give different outputs | Deterministic for fixed $\mathbf{z}\_0$ |
| **Likelihood** | Requires estimation | Exact via change of variables |
| **Flexibility** | Temperature tuning, partial denoising | Exact inversion |

### The Probability Flow ODE

For the variance-exploding (VE) SDE:
$$d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, d\mathbf{W}$$

The probability flow ODE is:
$$d\mathbf{x} = -\frac{1}{2} \frac{d[\sigma^2(t)]}{dt} \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \, dt$$

For the variance-preserving (VP) SDE:
$$d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} \, dt + \sqrt{\beta(t)} \, d\mathbf{W}$$

The probability flow ODE is:
$$d\mathbf{x} = \left[-\frac{1}{2} \beta(t) \mathbf{x} - \frac{1}{2} \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$$

**Why this matters**:
1. We can sample **deterministically** using ODE solvers (faster, more controllable)
2. We can compute **exact likelihoods** by integrating the log-determinant
3. We get **invertible encoding**: map data to latent codes and back exactly
4. We bridge **score-based models** (SDE) and **flow-based models** (ODE)

## Practical Implementations and Applications

### DDPM: Denoising Diffusion Probabilistic Models

**Algorithm** (Ho et al., 2020):

```python
# Training
for batch in dataloader:
    x_0 = batch
    t = random_timestep()
    epsilon = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * epsilon
    epsilon_pred = model(x_t, t)
    loss = mse_loss(epsilon_pred, epsilon)
    loss.backward()

# Sampling
x_T = torch.randn(batch_size, *image_shape)
for t in reversed(range(T)):
    epsilon_pred = model(x_t, t)
    x_{t-1} = denoise_step(x_t, epsilon_pred, t)
```

**Key innovation**: Simple training objective (predict noise), remarkable sample quality.

### Score-Based Models

**Noise Conditional Score Networks** (Song & Ermon, 2019):

- Train score network $\mathbf{s}\_\theta(\mathbf{x}, \sigma)$ at multiple noise levels
- Sample via **annealed Langevin dynamics**:

```python
for sigma in noise_levels:  # high to low
    for step in range(n_steps):
        score = score_network(x, sigma)
        x = x + step_size * score + sqrt(2 * step_size) * torch.randn_like(x)
```

**Advantage**: Unified treatment via SDE framework.

### Flow Matching

Recent approach (Lipman et al., 2023): instead of learning scores, directly regress the vector field:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\|^2$$

where $\mathbf{x}_t = (1-t) \mathbf{x}_0 + t \mathbf{x}_1$ interpolates between noise and data.

**Advantages**:
- Simple training objective
- No score matching required
- Direct ODE learning

### Rectified Flows

**Rectified flows** (Liu et al., 2022) iteratively straighten trajectories:

1. Train initial flow from noise to data
2. Generate pairs $(\mathbf{x}_0, \mathbf{x}_1)$ by simulating the flow
3. Train new flow on these pairs
4. Repeat

**Result**: Straighter paths → fewer ODE steps → faster sampling.

## Computer Vision Applications

**Image Generation**:
- **Stable Diffusion**: Text-to-image generation via latent diffusion
- **DALL-E 2**: CLIP-guided diffusion
- **Imagen**: Text-conditional diffusion in pixel space

**Image-to-Image Translation**:
- **SDEdit**: Stroke-based image editing
- **InstructPix2Pix**: Instruction-guided editing
- **ControlNet**: Adding spatial control to diffusion models

**Video Generation**:
- **Imagen Video**: Cascade of video diffusion models
- **Make-A-Video**: Text-to-video synthesis
- **Align Your Latents**: Video generation via latent diffusion

**3D Generation**:
- **DreamFusion**: Text-to-3D via score distillation sampling
- **Point-E**: 3D point cloud generation
- **Shap-E**: 3D shape generation

**Medical Imaging**:
- **Anomaly detection**: Model healthy anatomy, detect deviations
- **Super-resolution**: Enhance medical image quality
- **Synthesis**: Generate training data for rare conditions

**Inverse Problems**:
- **Denoising**: Remove noise while preserving structure
- **Inpainting**: Fill missing regions coherently
- **Super-resolution**: Recover high-frequency details

The key advantage: diffusion models provide **principled uncertainty quantification** and **composability** (combine multiple guidance signals).

## Comparing the Approaches

| **Aspect** | **Diffusion Models (SDE)** | **Normalizing Flows (ODE)** |
|---|---|---|
| **Mathematical foundation** | Stochastic differential equations | Ordinary differential equations |
| **Training** | Score matching / denoising | Maximum likelihood |
| **Sampling** | Iterative denoising (slow) | Single ODE solve (faster) |
| **Likelihood** | Approximate | Exact |
| **Sample quality** | State-of-the-art | Good, improving |
| **Controllability** | Flexible (temperature, guidance) | Exact inversion |
| **Architecture constraints** | Flexible (U-Nets, Transformers) | Invertibility, tractable Jacobian |
| **Conceptual simplicity** | Intuitive (denoise images) | Abstract (transform distributions) |
| **Recent innovations** | DDPM, EDM, consistency models | Flow matching, rectified flows |

**Convergence**: Modern research shows these are **equivalent frameworks** (Song et al., 2021). The probability flow ODE bridges them, enabling:
- Training as diffusion, sampling as flow (speed)
- Training as flow, sampling with noise (diversity)
- Hybrid approaches combining advantages

## Key Takeaways

1. **Brownian motion** provides the mathematical foundation for modern generative models through stochastic calculus and SDEs.

2. **Diffusion models** work by:
   - Forward process: gradually add noise (forward SDE)
   - Reverse process: learn to denoise (reverse-time SDE)
   - Training: score matching reduces to denoising

3. **Normalizing flows** transform distributions via:
   - Discrete flows: sequence of invertible transformations
   - Continuous flows: ODEs with tractable change of variables
   - Neural ODEs: parameterize transformations with neural networks

4. **Unified perspective**: Every diffusion SDE has an equivalent probability flow ODE generating the same distributions without stochasticity.

5. **Practical implications**:
   - Diffusion: flexible, high quality, slower sampling
   - Flows: exact likelihood, faster sampling, invertible
   - Hybrid methods: combine advantages

6. **The big picture**: Generative modeling is fundamentally about transforming simple distributions (Gaussian noise) into complex ones (natural images). Brownian motion and stochastic calculus provide the rigorous mathematical framework to do this in a principled, controllable way.

## Related Posts

- [The Landscape of Differential Equations: From ODEs to PDEs to SDEs]({{ site.baseurl }}{% link _posts/2025-12-29-differential-equations-ode-pde-sde.md %}) — Foundational overview of differential equations
- [Mathematical Properties of Brownian Motion: A Visual Guide]({{ site.baseurl }}{% link _posts/2025-12-30-mathematical-properties-brownian-motion.md %}) — Deep dive into Brownian motion properties
- [Itô Calculus: Why We Need New Rules for SDEs]({{ site.baseurl }}{% link _posts/2025-12-28-ito-calculus-stochastic-differential-equations.md %}) — Essential mathematical framework for stochastic calculus
- [Stochastic Processes and the Art of Sampling Uncertainty]({{ site.baseurl }}{% link _posts/2025-02-21-stochastic-processes-and-sampling.md %}) — Broader context for stochastic processes

## Further Reading

**Foundational Mathematics**:
- Øksendal, *Stochastic Differential Equations*
- Karatzas & Shreve, *Brownian Motion and Stochastic Calculus*
- Evans, *Partial Differential Equations* (for Fokker-Planck)

**Diffusion Models**:
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Song et al., "Score-Based Generative Modeling through SDEs" (ICLR 2021)
- Dhariwal & Nichol, "Diffusion Models Beat GANs" (NeurIPS 2021)
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (NeurIPS 2022)

**Normalizing Flows**:
- Rezende & Mohamed, "Variational Inference with Normalizing Flows" (ICML 2015)
- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Grathwohl et al., "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (ICLR 2019)

**Unified Frameworks**:
- Song et al., "Score-Based Generative Modeling through SDEs" (ICLR 2021)
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023)

**Applications**:
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
- Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (NeurIPS 2022)
- Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion" (ICLR 2023)

---

*The journey from Brown's microscope to modern AI systems generating images from text illustrates how deep mathematical understanding can unlock transformative technologies. By grounding generative models in the rigorous framework of stochastic processes, we gain not just better algorithms, but principled ways to understand, control, and improve them.*
