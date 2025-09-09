---
title: "Picture: A Probabilistic Programming Language for Scene Perception (2015)"
aliases: 
  - Picture
  - Probabilistic Programming for Vision
authors:
  - Tejas D. Kulkarni
  - Pushmeet Kohli
  - Josh Tenenbaum
  - Vikash Mansinghka
  - William T. Freeman
year: 2015
venue: "CVPR"
doi: "10.1109/CVPR.2015.7299176"
arxiv: "https://arxiv.org/abs/1502.04623"
code: "https://github.com/probprog/scene-interpretation" # (related repo, not official)
citations: 600+
dataset:
  - Synthetic scene datasets
  - Depth and image benchmarks (custom)
tags:
  - paper
  - probabilistic-programming
  - scene-perception
  - generative-models
fields:
  - vision
  - AI
  - probabilistic-models
related:
  - "[[Generative Vision Models]]"
  - "[[Inverse Graphics]]"
predecessors:
  - "[[Inverse Graphics Models]]"
successors:
  - "[[Gen (Probabilistic Programming, 2019)]]"
  - "[[Neural Scene Representation]]"
impact: ⭐⭐⭐⭐☆
status: "to-read"
---


# Summary
**Picture** introduces a probabilistic programming language tailored for **scene perception** tasks. It allows researchers to specify generative models of visual scenes and use probabilistic inference to invert them, combining **inverse graphics** with probabilistic programming.  

# Key Idea
> Express scene perception as probabilistic inference in generative models of graphics.  

# Method
- Based on **probabilistic programming framework** that combines graphics engines with Bayesian inference.  
- Scenes are described by **stochastic generative programs** (3D objects, lighting, camera parameters).  
- Perception = **inference**: match observed image data to latent scene parameters.  
- Supports **approximate inference** methods: MCMC, variational techniques, neural proposals.  
- Unified framework: same language can model multiple perception tasks.  

# Results
- Demonstrated applications on **3D object recognition, pose estimation, and scene parsing**.  
- Showed flexibility in handling complex occlusions and uncertainty.  
- Compared against discriminative baselines, showing generalisation advantages.  

# Why it Mattered
- First to connect **probabilistic programming** directly with **vision and scene understanding**.  
- Brought the **inverse graphics** paradigm into a programmable, flexible form.  
- Paved the way for future probabilistic programming systems (e.g., Gen) and hybrids with deep learning.  

# Architectural Pattern
- **Generative graphics models + probabilistic inference**.  
- Abstract language level: define priors, rendering, and likelihood in a unified system.  
- Influenced neural–probabilistic hybrids for scene understanding.  

# Connections
- **Contemporaries**: Deep generative models just emerging (VAEs, GANs, 2014–15).  
- **Influence**: Probabilistic programming in AI (Gen, Pyro), neural scene representation models.  

# Implementation Notes
- Computationally heavy due to MCMC-based inference.  
- Requires domain-specific priors for complex scenes.  
- More flexible than fixed architectures, but slower.  

# Critiques / Limitations
- Not real-time; inference can be very slow.  
- Struggles with high-dimensional latent spaces (complex 3D scenes).  
- Superseded in practice by **deep generative vision models**, but influential conceptually.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1502.04623)  
- [Video talk](https://www.youtube.com/watch?v=6hPZbTu8Nnw)  
- [Probabilistic Programming resources](http://probprog.org)  
- [Later system: Gen (MIT)](https://probcomp.csail.mit.edu/gen/)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Transformations in graphics rendering.  
- **Probability & Statistics**: Bayesian inference, priors/posteriors.  
- **Optimization Basics**: Approximate inference methods.  
- **Computer Graphics**: Rendering pipeline basics.  

## Postgraduate-Level Concepts
- **Numerical Methods**: MCMC sampling, variational inference.  
- **Machine Learning Theory**: Probabilistic models vs discriminative models.  
- **Generative Models**: Inverse graphics, scene generation.  
- **Research Methodology**: Program synthesis for scene interpretation.  

---

# My Notes
- Links directly to interests in **generative models + perception**.  
- Conceptually close to diffusion models: both treat perception as **inverting a generative process**.  
- Open question: Can modern generative priors (diffusion, transformers) replace hand-written stochastic programs?  
- Extension: Use **probabilistic programming + deep renderers** for controllable video synthesis.  
