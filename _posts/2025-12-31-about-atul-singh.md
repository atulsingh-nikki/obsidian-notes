---
layout: post
title: "About Atul Singh: Technical Portfolio and Expertise"
description: "Technical portfolio showcasing expertise in computer vision, machine learning, high-performance computing, and color science through 45+ technical blog posts and deep dives into production-grade systems."
tags: [portfolio, about, expertise, computer-vision, machine-learning, hpc, color-science]
---

# About Atul Singh

## Computer Vision Engineer | ML Practitioner | Technical Writer

Welcome to my technical portfolio. I'm Atul Singh, and I use this notebook to document deep dives into computer vision, machine learning infrastructure, high-performance computing, and the mathematical foundations that underpin production systems. What you see here isn't just theoretical exploration—it's battle-tested knowledge from building real-world pipelines, optimizing performance-critical code, and solving problems at the intersection of perception, computation, and human vision science.

### Snapshot
- **Output**: 45+ longform technical posts (~80k words) across 19 months
- **Roles**: CV/HPC engineer, ML infrastructure builder, and technical writer
- **Depth**: Multi-part series on contrast, Kalman filtering, C++ concurrency, sampling theory
- **Tooling**: Python (NumPy, PyTorch, OpenCV), C++17/20, CUDA, LaTeX
- **Philosophy**: Production rigor + mathematical grounding + accessible explanations

## What This Portfolio Demonstrates

Over **45+ technical posts** spanning 2024-2025, I've built a comprehensive technical curriculum covering:

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

## Writing Philosophy

My technical writing follows three principles:

**1. Production-Grade Rigor**
- Code examples are complete, tested, and production-ready
- Mathematical derivations include edge cases and numerical stability considerations
- Every "best practice" comes with the "why" and "when NOT to use"

**2. Multi-Layered Accessibility**
- Start with intuition and motivation
- Build to mathematical rigor with proper notation
- End with practical implementation details
- Readers at different levels can extract value

**3. Cross-Domain Synthesis**
- Connect mathematics, perception science, and engineering
- Show how concepts from physics, signal processing, and human vision converge
- Reference standards (SMPTE, ITU-R) and seminal papers

### How I Work
- **Reproducibility first**: deterministic seeds, pinned environments, and complete code listings
- **Instrumentation everywhere**: profiling (CPU/GPU), memory audits, and accuracy/stability checks
- **Design documents**: hypotheses, failure modes, and rollback criteria before implementing
- **Teaching mindset**: each post ladders from intuition → formalism → implementation → caveats

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

## Technical Breadth: Full Topic Coverage

### Computer Vision & Imaging (16 posts)
Contrast measurement • Color science • ACES workflows • Gamut mapping • HDR/SDR • Tone mapping • OCR evolution • Panoptic segmentation • Spectral imaging • Vision algorithms

### High-Performance Computing (6 posts)
SIMD intrinsics • GPU kernels • C++ concurrency • ISA architecture • Performance optimization • Template metaprogramming

### Machine Learning (8 posts)
Kalman filtering • PyTorch indexing • Knowledge distillation • Distribution shifts • Unsupervised learning • State estimation • Video models

### Mathematics (6 posts)
Optimization (gradients, Hessians, Lagrange multipliers) • Functional analysis (implicit function theorem) • Inverse trig symmetry • Normalized power sums

### Sampling & Probability (4 posts)
Stochastic processes • Importance sampling • Gibbs sampling • Direct PDF sampling challenges

### Linguistics & Communication (5 posts)
Technical writing • Word studies (culpable, resent, gripe/complaint/grievance) • Precision in language

## Impact & Reach

**Blog Statistics** (as of Dec 2025):
- **45 technical posts** across 19 months
- **~80,000 words** of technical content
- **Multiple comprehensive series** equivalent to graduate-level courses
- **Production-ready code examples** in Python, C++, CUDA

**Technical Depth**:
- Posts average 1,800 words with complete mathematical derivations
- Series posts build systematically from foundations to advanced topics
- Every post includes practical implementation guidance

## What Sets This Work Apart

**1. Comprehensive Coverage**
- Most series are the most complete publicly available treatments of their topics
- The contrast measurement series alone rivals textbook chapters in depth

**2. Production Perspective**
- Not just "how it works" but "when it breaks, why, and what to do about it"
- Performance considerations, numerical stability, edge cases
- Real-world deployment considerations

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
- Production-ready implementation skills in Python, C++, and CUDA
- Mathematical maturity for reading research papers and understanding proofs

**Practical Skills**:
- Ability to optimize performance-critical vision code
- Understanding of when to use supervised vs. unsupervised learning
- Knowledge of color science from capture to display

**Professional Development**:
- How to communicate complex technical concepts clearly
- How to structure multi-part technical narratives
- How to balance rigor with accessibility

**Time Investment**: ~30-40 hours of focused reading  
**Payoff**: Equivalent to multiple graduate-level courses + production experience

## Current Focus (Dec 2025)

**Active Series**:
- ✅ **Image Contrast Masterclass** (6/6 complete) — From grayscale to unsupervised ML
- 🔄 **Color Science Deep Dive** (ongoing) — ACES, gamut mapping, HDR workflows
- 📝 **ML Infrastructure** (planned) — Training pipelines, distributed systems

**Upcoming Topics**:
- Neural radiance fields (NeRF) and 3D reconstruction
- Advanced tone mapping algorithms
- Real-time ray tracing for computer graphics
- Transformer architectures for vision tasks

## Connect & Explore

**Portfolio Site**: [atulsingh-nikki.github.io/obsidian-notes](https://atulsingh-nikki.github.io/obsidian-notes/)

**Navigation**:
- [Publishing Cadence Summary](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/10/publishing-cadence-summary.html) — All 45 posts organized by month, quarter, year
- [Blog Index](https://atulsingh-nikki.github.io/obsidian-notes/blog/) — Browse by tag or date

**Featured Series**:
- [Contrast Measurement Masterclass]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}) (start here)
- [Kalman Filtering Curriculum](https://atulsingh-nikki.github.io/obsidian-notes/2024/09/20/introduction-to-kalman-filtering.html)
- [C++ Concurrency Trilogy](https://atulsingh-nikki.github.io/obsidian-notes/2025/02/18/understanding-futures-promises-cpp.html)

## About This Portfolio

This portfolio is a living document of continuous learning and knowledge sharing. Every post represents hours of research, implementation, testing, and refinement. The goal isn't just to document what I know—it's to build resources that help others climb the technical learning curve faster.

**Philosophy**: Technical excellence comes from understanding fundamentals deeply, connecting concepts across domains, and always asking "why" and "when does this break?"

**Commitment**: Every post is written with the care I'd want if I were learning the topic myself—complete code, mathematical rigor, practical considerations, and honest discussion of limitations.

---

*This portfolio reflects 19 months of systematic exploration across computer vision, machine learning, high-performance computing, and applied mathematics. The journey continues—new posts added regularly as I dive deeper into the intersection of perception, computation, and human vision science.*

**Last Updated**: December 31, 2025  
**Total Posts**: 45  
**Total Words**: ~80,000  
**Active Series**: 6 complete, 2 ongoing

---

## Keep Reading

* [Publishing Cadence Summary](https://atulsingh-nikki.github.io/obsidian-notes/2025/03/10/publishing-cadence-summary.html) — Complete index of all 45 posts
* [Contrast Measurement Masterclass]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}) — Start the 6-part series
* [Kalman Filtering Curriculum](https://atulsingh-nikki.github.io/obsidian-notes/2024/09/20/introduction-to-kalman-filtering.html) — 8-part state estimation series
* [Blog Home](https://atulsingh-nikki.github.io/obsidian-notes/) — Browse all content

*Want to discuss computer vision, color science, or HPC optimization? These posts are conversation starters—I'd love to hear your perspective on these topics.*
