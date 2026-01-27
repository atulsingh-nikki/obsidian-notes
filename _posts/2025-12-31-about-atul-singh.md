---
layout: post
title: "About Atul Singh: Technical Portfolio and Expertise"
description: "Technical notebook documenting deep explorations across computer vision, generative models, machine learning, high-performance computing, and applied mathematics. 69 comprehensive posts serving as both learning material and technical reference."
tags: [portfolio, about, expertise, computer-vision, machine-learning, hpc, color-science]
---

# About Atul Singh

## Computer Vision Engineer | ML Leader | Technical Writer

Welcome to my technical notebook. I'm Atul Singh, and I use this space to document deep dives into computer vision, machine learning, high-performance computing, and the mathematical foundations that underpin modern systems. 

Think of this as my **"second brain"**—comprehensive explorations that help me understand algorithms and techniques deeply, serving as reference material when tackling real-world problems. Each post represents the groundwork needed to make informed decisions in production environments: understanding VAEs deeply enables better architectural choices, mastering contrast metrics guides quality assessment design, and grasping stochastic processes informs robust system development.

### Snapshot
- **Output**: 69 longform technical posts (~90k words) across 2013-2026
- **Roles**: Computer Vision Engineer, ML Leader, Technical Writer
- **Depth**: Multi-part series on contrast, generative models, Kalman filtering, stochastic processes, C++ concurrency
- **Tooling**: Python (NumPy, PyTorch, OpenCV), C++17/20, CUDA, LaTeX
- **Philosophy**: Deep understanding + mathematical rigor + accessible explanations

## What This Notebook Covers

Over **69 technical posts** spanning 2013-2026, I've built a comprehensive technical curriculum covering:

### 🎯 Core Expertise Areas

#### Computer Vision & Image Processing

**Image Contrast & Quality Assessment** (6-part series)
- Authored the most comprehensive publicly available tutorial series on contrast measurement, from grayscale fundamentals to SDR/HDR cross-domain comparison
- Covers perceptual metrics, tone mapping, and when traditional metrics break down
- See: [Understanding Contrast in Images]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}) → [Color Contrast]({{ "/2025/12/27/understanding-color-contrast.html" | relative_url }}) → [Measuring Between Images]({{ "/2025/12/28/measuring-contrast-between-images.html" | relative_url }}) → [Content-Independent Metrics]({{ "/2025/12/29/comparing-contrast-across-different-images.html" | relative_url }}) → [SDR/HDR Comparison]({{ "/2025/12/30/sdr-hdr-contrast-comparison.html" | relative_url }}) → [Unsupervised ML Prediction]({{ "/2025/12/31/unsupervised-learning-contrast-prediction.html" | relative_url }})

**Color Science & Imaging Pipelines**
- Deep expertise in scene-referred workflows, ACES color pipelines, and real-time gamut precomputation
- Logarithmic color spaces, PCA-based analysis, and spectral imaging fundamentals
- Bitmap-based gamut precomputation for real-time color management systems
- See: [ACES Scene-Referred Workflows](https://atulsingh-nikki.github.io/obsidian-notes/2025/01/10/understanding-aces-scene-referred-workflows.html), [Logarithmic Color Spaces & PCA](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/24/logarithmic-color-pca.html), [Bitmap Gamut Precomputation](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/18/bitmap-gamut-precomputation.html)

**Modern CV Research**
- Panoptic segmentation without inductive biases (Pix2Seq-D analysis)
- Knowledge distillation for lightweight video models
- OCR evolution from Tesseract to transformers
- See: [Pix2Seq-D Panoptic Masks](https://atulsingh-nikki.github.io/obsidian-notes/2025/10/13/pix2seqd-panoptic-masks-without-inductive-biases.html), [Adaptive Dual-Teacher Distillation](https://atulsingh-nikki.github.io/obsidian-notes/2025/11/14/adaptive-dual-teacher-distillation-lightweight-video-models.html)

#### High-Performance Computing

**SIMD & GPU Programming**
- Production-grade intrinsics from SSE to AVX2
- GPU kernel programming with grids, blocks, and warps
- Bridging ISA concepts to GPU programming mindsets
- Performance optimization for vision workloads
- See: [SIMD Intrinsics: SSE to AVX2](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/26/intrinsics-simd-avx.html), [GPU Kernel Programming](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/28/gpu-kernel-programming-basics.html), [ISA to GPU Kernels Bridge](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/08/isa-to-gpu-kernel-bridge.html)

**Modern C++ Mastery**
- Complete trilogy on futures, promises, and std::async
- Reference types (lvalue, rvalue, universal) with production examples
- Template metaprogramming and zero-cost abstractions
- See: [Futures & Promises](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/18/understanding-futures-promises-cpp.html) → [Composing Futures](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/19/compound-futures-modern-cpp.html) → [Mastering std::async](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/20/mastering-std-async.html), [C++ Reference Types](https://atulsingh-nikki.github.io/obsidian-notes/2025/10/22/cpp-reference-types-explained.html)

#### Machine Learning & Probabilistic Reasoning

**State Estimation & Filtering** (8-part curriculum)
- Complete Kalman filtering series from Bayesian foundations to nonlinear extensions
- Real-world applications in tracking and control
- EKF, UKF, and particle filter implementations
- See: [Introduction to Kalman Filtering](https://atulsingh-nikki.github.io/obsidian-notes/2024/09/20/introduction-to-kalman-filtering.html) through [Advanced Topics](https://atulsingh-nikki.github.io/obsidian-notes/2024/09/27/advanced-kalman-topics.html)

**Sampling Theory & Uncertainty** (3-part series)
- Stochastic processes and the art of sampling uncertainty
- Advanced techniques: importance sampling, Gibbs, stratified sampling
- Why direct PDF sampling is fundamentally hard
- See: [Stochastic Processes & Sampling](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/21/stochastic-processes-and-sampling.html), [Advanced Sampling Techniques](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/22/advanced-sampling-techniques.html), [Why Direct Sampling Is Hard](https://atulsingh-nikki.github.io/obsidian-notes/2025/10/04/why-direct-sampling-from-pdfs-is-hard.html)

**Generative Models & Variational Inference** (NEW 2026)
- Complete series on the partition function problem and why it's intractable
- How VAEs avoid computing Z through the Evidence Lower Bound (ELBO)
- Mathematical foundations of expectation and variational inference
- Historical context: why discriminative learning dominated first
- ML paradigms: modeling distributions vs learning functions
- See: [The Z Problem]({{ "/2025/12/24/normalization-constant-problem.html" | relative_url }}), [VAEs & ELBO]({{ "/2026/01/01/how-vaes-avoid-computing-partition-function.html" | relative_url }}), [Expected Value Foundations]({{ "/2026/01/01/expected-value-expectation-mathematical-foundations.html" | relative_url }})

**ML Infrastructure**
- PyTorch tensor indexing from 1D to N-dimensional views
- Unsupervised learning for contrast prediction without ground truth
- Distribution shifts in the AI era (from funnels to loops)
- See: [PyTorch Tensor Indexing](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/12/pytorch-indexing-slicing.html), [Unsupervised Contrast Prediction]({{ "/2025/12/31/unsupervised-learning-contrast-prediction.html" | relative_url }}), [Distribution Shifts in AI](https://atulsingh-nikki.github.io/obsidian-notes/2025/11/16/from-funnels-to-loops-distribution-shifts-in-the-ai-era.html)

#### Mathematical Foundations

**Optimization & Analysis**
- Gradient and Hessian intuition for vision/ML systems
- Implicit function theorem with geometric interpretation
- Lagrange multipliers and why intersection methods fail
- See: [From Gradients to Hessians](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/01/from-gradients-to-hessians.html), [Implicit Function Theorem](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/10/understanding-implicit-function-theorem.html), [Lagrange Multipliers Geometry](https://atulsingh-nikki.github.io/obsidian-notes/2025/01/27/why-intersection-fails-lagrange-multipliers.html)

**Applied Mathematics**
- Hidden symmetry of inverse trig functions
- Normalized power sums from elementary math to vision algorithms
- Random vs. stochastic: clarifying variables, processes, and optimization
- See: [Inverse Trig Symmetry](https://atulsingh-nikki.github.io/obsidian-notes/2025/01/15/hidden-symmetry-inverse-trig.html), [Elementary Math to Vision](https://atulsingh-nikki.github.io/obsidian-notes/2025/09/22/elementary-math-to-vision-algorithms.html)

**Stochastic Processes & Diffusion Models** (NEW 2025)
- Brownian motion: mathematical properties and infinite total variation
- Itô calculus and stochastic differential equations (SDEs)
- Connection to modern diffusion and flow-based generative models
- Total variation and its role in understanding stochastic behavior
- See: [Brownian Motion & Diffusion]({{ "/2025/12/31/brownian-motion-diffusion-flow-models.html" | relative_url }}), [Mathematical Properties]({{ "/2025/12/30/mathematical-properties-brownian-motion.html" | relative_url }}), [Itô Calculus]({{ "/2025/12-28/ito-calculus-stochastic-differential-equations.html" | relative_url }})

## Writing Philosophy

My technical writing serves as both learning tool and reference material:

**1. Deep Understanding First**
- Code examples are complete and runnable for hands-on learning
- Mathematical derivations include edge cases and numerical stability considerations
- Every concept comes with the "why" and "when NOT to use" to build intuition

**2. Multi-Layered Accessibility**
- Start with intuition and motivation (the "why")
- Build to mathematical rigor with proper notation (the "how")
- End with practical implementation details (the "what")
- Readers at different levels can extract value

**3. Cross-Domain Synthesis**
- Connect mathematics, perception science, and engineering
- Show how concepts from physics, signal processing, and human vision converge
- Reference standards (SMPTE, ITU-R) and seminal papers for further exploration

### My Approach to Learning
- **First principles**: Understand the foundations before diving into implementations
- **Reproducibility**: Deterministic seeds, pinned environments, and complete code listings
- **Multiple perspectives**: Mathematical rigor + geometric intuition + practical considerations
- **Teaching mindset**: Write as if explaining to my past self—each post ladders from intuition → formalism → implementation → caveats

## Key Technical Series

### 📘 The Contrast Masterclass (6 posts, ~12,000 words)
The most comprehensive public resource on image contrast measurement, covering:
- Grayscale contrast fundamentals (RMS, Michelson, local, frequency-domain)
- Color contrast beyond luminance (chromatic, opponent channels, Lab space)
- Same-content comparison metrics (SSIM, local preservation, perceptual models)
- Content-independent comparison for different images
- SDR/HDR cross-domain comparison with tone mapping
- Unsupervised ML for contrast prediction without ground truth

**Industry Impact**: This series provides a complete reference for anyone building image quality assessment systems, tone mapping algorithms, or dataset curation pipelines.

### 📗 Kalman Filtering Curriculum (8 posts)
From Bayesian foundations to advanced nonlinear extensions:
- Recursive filtering fundamentals
- Complete mathematical derivation
- Python implementation from scratch
- Real-world applications (tracking, navigation, sensor fusion)
- EKF, UKF, and particle filters
- Advanced topics and future directions

**Academic Equivalent**: Graduate-level state estimation course

### 📙 C++ Concurrency Trilogy (3 posts)
Production-grade modern C++ concurrency:
- Futures and promises: when and why
- Composing futures for complex workflows
- Mastering std::async with launch policies
- Bonus: Reference types (lvalue, rvalue, universal)

**Industry Relevance**: Essential for building high-throughput vision pipelines

### 📕 Sampling Theory Arc (3 posts)
Understanding uncertainty in ML and CV:
- Stochastic processes and sampling fundamentals
- Importance, Gibbs, and stratified sampling
- Why direct PDF sampling is hard (and what to do instead)

**Research Applications**: Monte Carlo methods, MCMC, probabilistic inference

### 📔 Generative Models & VAEs (5 posts, ~12,000 words) **NEW 2025-2026**
Complete treatment of the partition function problem and variational inference:
- The curse of dimensionality and why high-dimensional spaces are strange
- The normalization constant (Z) problem and why it's intractable
- Historical context: why discriminative learning came first
- ML paradigms: distributions vs functions
- How VAEs cleverly avoid Z through ELBO derivation
- Mathematical foundations of expectation for variational inference

**Industry Impact**: Essential for understanding modern generative models (VAEs, GANs, diffusion models) and why certain architectural choices exist

### 📓 Stochastic Processes & Diffusion (5 posts) **NEW 2025**
From Brownian motion to modern generative models:
- Brownian motion mathematical properties
- Total variation and why Brownian paths are "badly behaved"
- Itô calculus and stochastic differential equations
- Connection to diffusion models and score-based generative models
- Flow matching and modern generative modeling

**Research Relevance**: Foundation for understanding diffusion models (DDPM, score-based models) and continuous normalizing flows

## Technical Breadth: Full Topic Coverage

### Computer Vision & Imaging (16 posts)
Contrast measurement • Color science • ACES workflows • Gamut mapping • HDR/SDR • Tone mapping • OCR evolution • Panoptic segmentation • Spectral imaging • Vision algorithms

### High-Performance Computing (6 posts)
SIMD intrinsics • GPU kernels • C++ concurrency • ISA architecture • Performance optimization • Template metaprogramming

### Machine Learning (13 posts)
Kalman filtering • PyTorch indexing • Knowledge distillation • Distribution shifts • Unsupervised learning • State estimation • Video models • VAEs • ELBO • Partition function • Generative models

### Mathematics (11 posts)
Optimization (gradients, Hessians, Lagrange multipliers) • Functional analysis (implicit function theorem) • Inverse trig symmetry • Normalized power sums • Brownian motion • Stochastic calculus • Itô's lemma • Total variation

### Generative Models & Variational Inference (5 posts)
Partition function problem • Curse of dimensionality • VAEs • ELBO derivation • Expected value foundations • Discriminative vs generative history

### Sampling & Probability (4 posts)
Stochastic processes • Importance sampling • Gibbs sampling • Direct PDF sampling challenges

### Linguistics & Communication (5 posts)
Technical writing • Word studies (culpable, resent, gripe/complaint/grievance) • Precision in language

## Impact & Reach

**Blog Statistics** (as of Jan 2026):
- **69 technical posts** across 2013-2026
- **~90,000 words** of technical content
- **Multiple comprehensive series** equivalent to graduate-level courses
- **Complete, runnable code examples** in Python, C++, CUDA

**Technical Depth**:
- Posts average 1,800 words with complete mathematical derivations
- Series posts build systematically from foundations to advanced topics
- Every post includes practical implementation guidance

## What Sets This Work Apart

**1. Comprehensive Coverage**
- Most series are the most complete publicly available treatments of their topics
- The contrast measurement series alone rivals textbook chapters in depth

**2. Practical Depth**
- Not just "how it works" but "when it breaks, why, and what to do about it"
- Performance considerations, numerical stability, edge cases
- Awareness of real-world constraints and implementation challenges

**3. Cross-Domain Integration**
- Connects mathematics, perception science, and engineering
- Shows how concepts from different fields reinforce each other
- Builds intuition through multiple perspectives

**4. Accessibility Without Compromise**
- Technical rigor maintained throughout
- Concepts explained from first principles
- Code examples complement mathematical exposition

## Selected Highlights

**Most Comprehensive**:
- [6-part Contrast Measurement Series]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}) — Most complete public resource on image contrast
- [8-part Kalman Filtering Curriculum](https://atulsingh-nikki.github.io/obsidian-notes/2024/09/20/introduction-to-kalman-filtering.html) — From Bayesian foundations to particle filters

**Most Technical**:
- [SDR/HDR Cross-Domain Comparison]({{ "/2025/12/30/sdr-hdr-contrast-comparison.html" | relative_url }}) — Encoding, tone mapping, perceptual metrics
- [GPU Kernel Programming](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/28/gpu-kernel-programming-basics.html) — Grids, blocks, warps explained

**Most Practical**:
- [SIMD Intrinsics: SSE to AVX2](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/26/intrinsics-simd-avx.html) — Production-grade optimization
- [PyTorch Tensor Indexing](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/12/pytorch-indexing-slicing.html) — From 1D slices to N-D views

**Most Original**:
- [Unsupervised Contrast Prediction]({{ "/2025/12/31/unsupervised-learning-contrast-prediction.html" | relative_url }}) — ML without ground truth
- [Elementary Math to Vision Algorithms](https://atulsingh-nikki.github.io/obsidian-notes/2025/09/22/elementary-math-to-vision-algorithms.html) — Connecting normalized power sums

## Skills Demonstrated

### Programming Languages & Tools
- **Python**: NumPy, OpenCV, PyTorch, scikit-learn, SciPy
- **C++**: Modern C++17/20, templates, concurrency, SIMD intrinsics
- **CUDA**: Kernel programming, memory hierarchy, optimization
- **Mathematics**: LaTeX, symbolic computation, numerical methods

### Technical Communication
- **Documentation**: API design, technical specifications, user guides
- **Tutorial Writing**: Multi-level accessibility, code examples, visual aids
- **Research Synthesis**: Distilling papers into actionable knowledge

### Domain Expertise
- **Computer Vision**: Image processing, quality assessment, color science
- **Machine Learning**: State estimation, probabilistic reasoning, unsupervised learning
- **Systems Engineering**: Performance optimization, pipeline design, real-time systems
- **Applied Mathematics**: Optimization, functional analysis, probability theory

## Education Through Writing

If you read through this entire collection systematically, you'll gain:

**Technical Foundations**:
- Graduate-level knowledge in computer vision, probabilistic reasoning, and HPC
- Implementation skills in Python, C++, and CUDA with runnable examples
- Mathematical maturity for reading research papers and understanding proofs

**Practical Skills**:
- Ability to understand and implement performance-critical vision algorithms
- Understanding of when to use supervised vs. unsupervised learning
- Knowledge of color science from capture to display

**Professional Development**:
- How to communicate complex technical concepts clearly
- How to structure multi-part technical narratives
- How to balance rigor with accessibility

**Time Investment**: ~30-40 hours of focused reading  
**Payoff**: Equivalent to multiple graduate-level courses in depth and rigor

## Current Focus (Jan 2026)

**Active Series**:
- ✅ **Image Contrast Masterclass** (6/6 complete) — From grayscale to unsupervised ML
- ✅ **Generative Models & VAEs** (5/5 complete) — From partition function problem to ELBO
- ✅ **Stochastic Processes & Diffusion** (5/5 complete) — From Brownian motion to modern generative models
- 🔄 **Color Science Deep Dive** (ongoing) — ACES, gamut mapping, HDR workflows
- 📝 **Advanced Generative Models** (planned) — Diffusion models, score matching, flow matching

**Upcoming Topics**:
- Denoising diffusion probabilistic models (DDPM)
- Score-based generative models
- Neural radiance fields (NeRF) and 3D reconstruction
- Continuous normalizing flows
- Transformer architectures for vision tasks

## Connect & Explore

**Portfolio Site**: [atulsingh-nikki.github.io/obsidian-notes](https://atulsingh-nikki.github.io/obsidian-notes/)

**Navigation**:
- [Publishing Cadence Summary](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/10/publishing-cadence-summary.html) — All 69 posts organized by month, quarter, year
- [Blog Index](https://atulsingh-nikki.github.io/obsidian-notes/blog/) — Browse by tag or date

**Featured Series**:
- [Generative Models & VAEs]({{ "/2025/12/24/normalization-constant-problem.html" | relative_url }}) (NEW - start here!)
- [Contrast Measurement Masterclass]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }})
- [Stochastic Processes & Diffusion]({{ "/2025/12/31/brownian-motion-diffusion-flow-models.html" | relative_url }}) (NEW)
- [Kalman Filtering Curriculum](https://atulsingh-nikki.github.io/obsidian-notes/2024/09/20/introduction-to-kalman-filtering.html)
- [C++ Concurrency Trilogy](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/18/understanding-futures-promises-cpp.html)

## About This Notebook

This is a living document of continuous learning and knowledge sharing. Every post represents hours of deep research, hands-on implementation, and thoughtful refinement—building a comprehensive technical reference.

**Purpose**: Create detailed explorations that serve both as learning material and future reference. When I need to understand how VAEs work or implement contrast metrics, I return to these posts.

**Philosophy**: Technical excellence comes from understanding fundamentals deeply, connecting concepts across domains, and always asking "why" and "when does this break?"

**Approach**: Every post is written with the care I'd want if I were learning the topic myself—complete code, mathematical rigor, practical considerations, and honest discussion of limitations.

---

*This notebook reflects a multi-year arc (2013-2026) of systematic exploration across computer vision, machine learning, generative models, high-performance computing, and applied mathematics. Each post is both a learning artifact and a reference for future work. The journey continues—new posts added regularly as I dive deeper into the intersection of perception, computation, probabilistic inference, and modern generative modeling.*

**Last Updated**: January 1, 2026  
**Total Posts**: 69  
**Total Words**: ~90,000  
**Active Series**: 9 complete, 2 ongoing

---

## Keep Reading

* [Publishing Cadence Summary](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/10/publishing-cadence-summary.html) — Complete index of all 69 posts
* [Generative Models & VAEs Series]({{ "/2025/12/24/normalization-constant-problem.html" | relative_url }}) — NEW 5-part series on the partition function problem
* [Contrast Measurement Masterclass]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}) — 6-part series from grayscale to ML
* [Stochastic Processes & Diffusion]({{ "/2025/12/31/brownian-motion-diffusion-flow-models.html" | relative_url }}) — NEW 5-part series on Brownian motion to modern generative models
* [Kalman Filtering Curriculum](https://atulsingh-nikki.github.io/obsidian-notes/2024/09/20/introduction-to-kalman-filtering.html) — 8-part state estimation series
* [Blog Home](https://atulsingh-nikki.github.io/obsidian-notes/) — Browse all content

*Want to discuss computer vision, generative models, color science, or HPC optimization? These posts are conversation starters—I'd love to hear your perspective on these topics.*
