---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
authors:
  - Ben Mildenhall
  - Pratul Srinivasan
  - Matthew Tancik
  - Jonathan T. Barron
  - Ravi Ramamoorthi
  - Ren Ng
year: 2020
venue: "ECCV 2020"
dataset:
  - Synthetic NeRF dataset
  - Real-world forward-facing images
tags:
  - computer-vision
  - 3d-reconstruction
  - novel-view-synthesis
  - implicit-representations
  - neural-rendering
  - radiance-fields
arxiv: "https://arxiv.org/abs/2003.08934"
related:
  - "[[Volumetric Rendering]]"
  - "[[Implicit Neural Representations]]"
  - "[[GANs (2014)]]"
  - "[[Diffusion Models]]"
  - "[[Instant NeRF (2022)]]"
  - "[[3D Scene Representation]]"
---

# Summary
NeRF introduced a method to represent **3D scenes as continuous neural radiance fields**, trained directly from 2D images. By querying a neural network with 3D coordinates and viewing directions, NeRF learns to output color and density values, which can be volume-rendered to synthesize **novel views** of a scene with high photorealism.

# Key Idea (one-liner)
> Represent a scene as a continuous function (neural network) mapping 3D location + viewing direction → color + density, and use volume rendering for photorealistic novel view synthesis.

# Method
- **Input**: multiple posed 2D images of a static scene.
- **MLP Network**:
  - Input: 3D point (x, y, z) + viewing direction (θ, φ).
  - Output: color (RGB) + volume density (σ).
- **Volume Rendering**:
  - Integrate color and density along rays to form final pixel colors.
  - Differentiable → allows backpropagation from rendered image to network weights.
- **Positional Encoding**:
  - Projects input coordinates into higher frequency space (Fourier features) for better learning of high-frequency details.
- **Optimization**:
  - Gradient descent minimizing reconstruction loss between rendered and ground-truth images.

# Results
- Produced highly realistic **novel view synthesis** from sparse input images.
- Outperformed prior methods in realism and detail.
- Limitations: slow training (hours per scene) and inference (minutes per frame in original).

# Why it Mattered
- Opened a new paradigm in **neural rendering** and **3D representation learning**.
- Inspired a massive wave of NeRF variants (FastNeRF, Instant-NGP, NeRF-W, Mip-NeRF).
- Unified graphics-style rendering with deep learning.
- Key step toward **photorealistic 3D scene reconstruction from images**.

# Architectural Pattern
- [[Implicit Neural Representations]] → continuous scene encoded in MLP.
- [[Positional Encoding]] → capture fine details.
- [[Differentiable Volume Rendering]] → link 2D supervision to 3D representation.
- [[Multi-View Learning]] → supervision from multiple viewpoints.

# Connections
- **Predecessors**:
  - Classical volume rendering and light field methods.
- **Successors**:
  - [[Instant NeRF (2022)]] — massive speedups with hash encoding.
  - [[Mip-NeRF (2021)]] — anti-aliasing and multi-scale representations.
  - [[NeRF in the Wild (2020)]] — handling unstructured photo collections.
- **Influence**:
  - Brought implicit neural representations to mainstream.
  - Now a standard benchmark for 3D vision + graphics.

# Implementation Notes
- Original implementation slow (hours to train, minutes to render).
- Requires known camera poses (SfM or COLMAP preprocessing).
- Needs dense view coverage to avoid artifacts.
- Modern variants accelerate via GPU kernels, hash encoding, or distillation.

# Critiques / Limitations
- High compute/training cost.
- Static scenes only (original version).
- Not scalable to large, unbounded scenes (solved partially by NeRF-W, Mega-NeRF).
- Memory + inference bottlenecks.

# Repro / Resources
- Paper: [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
- Official code: [NeRF GitHub](https://github.com/bmild/nerf)
- Datasets: synthetic Blender scenes, real forward-facing captures.
- Variants: Instant-NGP, Mip-NeRF, NeRF-W.

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - 3D coordinate transformations (world ↔ camera).
  - Matrix multiplications for projection.

- **Probability & Statistics**
  - Volume rendering as weighted expectation along a ray.
  - Density interpreted probabilistically (opacity function).

- **Calculus**
  - Differentiable volume integration.
  - Backpropagation through rendering equations.

- **Signals & Systems**
  - Positional encoding as Fourier feature mapping.
  - Sampling frequency vs aliasing.

- **Data Structures**
  - Rays traced per pixel.
  - Spatial queries as continuous inputs.

- **Optimization Basics**
  - Gradient descent on reconstruction loss.
  - Regularization via augmentation and ray sampling.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Importance sampling rays for efficiency.
  - Curriculum sampling strategies.
  - Gradient stability in volumetric rendering.

- **Numerical Methods**
  - Numerical integration of radiance fields.
  - Monte Carlo sampling for rays.
  - Positional encodings as basis expansion.

- **Machine Learning Theory**
  - Implicit neural representations as function approximators.
  - Generalization across unseen views.
  - Trade-off: inductive bias of MLP vs CNNs.

- **Computer Graphics / Vision**
  - Volume rendering pipeline.
  - Multi-view geometry (pose consistency).
  - Neural rendering vs classical graphics.

- **Neural Network Design**
  - MLP with skip connections.
  - Positional encoding improves high-frequency learning.
  - Ray-based training pipeline.

- **Transfer Learning**
  - Pretraining NeRF on scene priors (not common originally).
  - Extensions to dynamic scenes, large-scale datasets.

- **Research Methodology**
  - Benchmarking on synthetic and real datasets.
  - Ablation: positional encoding, network depth, rendering resolution.
  - Comparison with traditional graphics pipelines.
