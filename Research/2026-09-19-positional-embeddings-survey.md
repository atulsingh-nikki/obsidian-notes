---
title: "Positional Embeddings Across the Literature: Sinusoidal, Learned, Relative, RoPE, ALiBi and Beyond"
aliases:
  - Positional Embeddings
  - Position Encoding in Transformers
  - RoPE
  - ALiBi
authors:
  - Atul Singh
year: 2026
venue: "Research Notes / Survey"
tags:
  - transformers
  - positional-embeddings
  - vision-transformers
  - attention
  - machine-learning
  - deep-learning
  - large-language-models
related:
  - "[[Vision Transformer (ViT): An Image is Worth 16×16 Words]]"
  - "[[DINO: Emerging Properties in Self-Supervised Vision Transformers]]"
  - "[[BYOL (2020)]]"
  - "[[MoCo Momentum Contrast for Unsupervised Visual Representation Learning]]"
  - "[[Self-Attention]]"
impact: ⭐⭐⭐⭐
status: "read"
---

# Summary
This note surveys the main positional embedding families used in modern Transformers: absolute, relative, rotary, and distance-based biases like ALiBi. It connects the design choices to the real ViT geometry, explains why order matters, and highlights where each method works best in practice.

# Key Idea
> Positional encoding is not an optional add-on; it is the mechanism that restores order, locality, and relative geometry to attention.

TL;DR

- Positional information is essential for transformers because attention is permutation-invariant; embeddings inject order or relative relations.
- Methods split into absolute (sinusoidal, learned), relative (biases, attention score transforms), and rotation-based (RoPE/complex embeddings); ALiBi biases attention by distance without learned vectors.
- Choice depends on generalization needs, inductive biases, memory/computation, and whether locality or long-range extrapolation is desired.

Introduction

Transformers process sets of token embeddings with attention, which by itself does not encode token order. Positional embeddings (PE) add structure so the model can reason about positions, distances, and order. This survey catalogs common PE families, shows concise math, highlights pros/cons, and gives practical recommendations for vision (ViT-style) and language models.

1) Absolute Position Embeddings

- Sinusoidal (Vaswani et al., 2017)
  - Math: For position p and channel k: $PE_{p,2k}=\sin(p / 10000^{2k/d})$, $PE_{p,2k+1}=\cos(p / 10000^{2k/d})$.
  - Properties: fixed, parameter-free, enables some extrapolation, supports closed-form relative-phase reasoning.
  - Pros: no extra parameters; extrapolates to longer sequences; smooth frequency basis interpretable as Fourier features.
  - Cons: may be less flexible for learned, dataset-specific position patterns.

- Learned Absolute
  - Math: learn a matrix $E_{pos} \in \mathbb{R}^{L\times d}$ added to token embeddings: $z_0 = x + E_{pos}[p]$.
  - Properties: fully flexible; can memorize position-specific quirks (e.g., BOS, EOS patterns).
  - Pros: strong empirical performance when sequence length is fixed or bounded; easy to implement.
  - Cons: poor extrapolation beyond training lengths; uses extra params.

2) Relative Position Representations (RPR)

- Key idea: Represent relative distance between tokens directly in attention computation rather than absolute add-on vectors.

- Shaw et al. (2018) — relative bias to key/value
  - Add learned bias based on relative distance r to attention logits: $a_{ij}=q_i^T k_j + b_{r_{ij}}$.
  - Pros: models distance-specific interactions; parameter-efficient when bucketing distances.

- Transformer-XL / T5 style (content+position and relative functions)
  - Use decomposed terms so query interacts with relative position embeddings via additional projection matrices; preserves translational equivariance and works well with recurrence/segment-level training.

3) Rotary Positional Embeddings (RoPE)

- Wang et al. (2021) / Su et al. — apply rotation in token subspace to inject relative phase.
  - Math sketch: split each head's query/key vector into pairs and rotate by angle proportional to position: multiply by 2D rotation block R(p) so that $q_p^T k_q$ incorporates $\cos(\Delta p)$ terms.
  - Pros: naturally encodes relative phase, composes across sequence concatenation (extrapolates), parameter-free per-position transform (no extra params per position for learned RoPE embeddings).
  - Cons: implementation slightly more complex; interacts with head dimension and interleaving.

4) ALiBi (Attention with Linear Biases)

- Press et al. (2021) — add a fixed, linearly decaying bias to attention logits proportional to distance:
  - $a_{ij}=q_i^T k_j + slope \times (i-j)$.
  - Properties: no learned position vectors; biases scale per head; encourages locality for distant pairs.
  - Pros: works well for long-context generalization; no learned positional parameters; cheap.
  - Cons: simpler inductive bias—less expressive than learned relative schemes; choice of slope schedule matters.

5) Bucketing and Distance Binning

- Many practical systems bucket relative distances (e.g., clip distances >K into last bin) to control parameter growth and handle long ranges.

6) 2D / Image-Specific Positional Encodings

- Vision needs two-dimensional position info: classic options
  - Flattened absolute learned 2D grid: learn an X and Y embedding then sum: $E_{pos}(x,y)=E_x[x]+E_y[y]$.
  - 2D sinusoidal extensions: apply sin/cos basis separately to rows and cols.
  - Relative 2D biases: encode offsets (dx,dy) into attention biases or buckets.

7) Practical Trade-offs & Recommendations

- If you expect sequences longer than training lengths and need extrapolation: prefer sinusoidal, RoPE, or ALiBi.
- If dataset has strong absolute-position signals and length is bounded: learned absolute often gives best in-distribution accuracy.
- For vision/ViT: 2D learned grid + interpolation for different image sizes is common; RoPE variants or relative 2D biases support better translation equivariance.
- For language models aiming at long context: ALiBi or RoPE give robust extrapolation without huge param growth.

8) Implementation Notes

- Interpolation: learned grids can be interpolated (bicubic) to new resolutions for different image sizes.
- Bucketing: use log-scale buckets for huge ranges; combine with clipping to keep param counts small.
- Numerical stability: when rotating subspaces (RoPE) keep consistent head dims and consider FP precision when using very long sequences.

9) Short Comparative Table (informal)

- Sinusoidal: parameter-free, extrapolates moderately, math-clean.
- Learned absolute: most flexible in-distribution, fails to extrapolate.
- Relative (Shaw/T5): expressive, models distance-specific interactions.
- RoPE: rotation-based, composes for extrapolation, parameter-free.
- ALiBi: linear bias, strong long-context generalization, parameter-light.

10) Open Questions & Research Directions

- Best practices for combining 2D relative position with patch-based ViTs.
- How learned position embeddings interact with data augmentation (cropping, resizing).
- Hybrid schemes: learned tables for short-range, ALiBi/RoPE for long-range — how to balance them.

11) Worked numeric example for ViT

The classic ViT setup makes the geometry concrete. Suppose an image is 224x224 pixels and the patch size is 16x16. Then

- 224 / 16 = 14 patches along each axis
- total patch tokens = 14 x 14 = 196

For a ViT-Base model, the patch embedding dimension is usually D = 768 and the positional embedding matrix is therefore shaped as 196 x 768. That means the model learns a positional vector for each of the 196 locations, and each vector has 768 values. In other words, the positional term is not a single scalar per token; it is a full learned embedding vector.

This is why the equation is often written as

$$
z_0 = [x_{class}; x_1 E; x_2 E; \ldots; x_N E] + E_{pos}
$$

where $E$ is the patch embedding projection and $E_{pos}$ is the learned position matrix. A standard ViT-Base configuration uses roughly 196 x 768 = 150,528 positional parameters in the base grid, before any additional classification token or other task-specific heads are included.

This also explains why the model needs interpolation or resizing logic when the input image size changes: the grid has to match a new number of patches, and a learned fixed-length absolute embedding cannot simply be reused without adaptation.

12) Tiny RoPE and ALiBi example

Below is a minimal sketch of how RoPE and ALiBi differ conceptually.

```python
import numpy as np


def rotate_half(x, angle):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return np.concatenate([
        x1 * np.cos(angle) - x2 * np.sin(angle),
        x1 * np.sin(angle) + x2 * np.cos(angle)
    ], axis=-1)


def rope_like(q, k, pos):
    angle = pos / 10000 ** (np.arange(q.shape[-1]) / q.shape[-1])
    q_rot = rotate_half(q, angle)
    k_rot = rotate_half(k, angle)
    return q_rot, k_rot


def alibi_bias(i, j, slope=0.02):
    return slope * (i - j)
```

The RoPE version rotates query/key pairs by position-dependent angles. The ALiBi version does not add a learned positional vector; it adds a distance-based bias directly to the attention logits.

13) Diagram idea

A useful diagram would show a 14x14 patch grid, then mark a few token positions and show how learned absolute embeddings or relative offsets change the attention score. The value is not just arithmetic but the intuition: positions encode locality and ordering that raw attention otherwise loses.

# Repro / Resources

- Vaswani et al., "Attention Is All You Need" (2017) — sinusoidal PE.
- Shaw et al., "Self-Attention with Relative Position Representations" (2018).
- Dai et al., "Transformer-XL" (2019).
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5) — relative position.
- Su et al. / RoPE papers (2021–2022).
- Press et al., "ALiBi" (2021).
- Companion blog post: [Positional Embeddings in ViT: Why Add Them, and How the Shapes Actually Match](https://atulsingh-nikki.github.io/obsidian-notes/2026/09/19/positional-embeddings-in-vision-transformers/)
- Companion blog post: [Positional Embeddings Across the Literature: Sinusoidal, Learned, Relative, RoPE, ALiBi and Beyond](https://atulsingh-nikki.github.io/obsidian-notes/2026/09/19/positional-embeddings-survey/)
