---
title: "MDP: Multidimensional Vision Model Pruning with Latency Constraint (2025)"
aliases:
  - MDP Pruning
  - Multidimensional Vision Model Pruning
  - MDP 2025
authors:
  - Xinglong Sun
  - Barath Lakshmanan
  - Maying Shen
  - Shiyi Lan
  - Jingde Chen
  - Jose M. Alvarez
year: 2025
venue: arXiv
doi: 10.48550/arXiv.2504.02168
arxiv: https://arxiv.org/abs/2504.02168
code: ""
citations: 0+
tags:
  - paper
  - model-compression
  - pruning
  - latency-aware-optimization
  - vision-transformers
  - cnn
fields:
  - computer-vision
  - efficient-ml
  - optimization
related:
  - "[[Optimal Brain Damage (1989)]]"
  - "[[Deep Compression Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding (2015)|Deep Compression (2015)]]"
predecessors:
  - Latency-aware structural pruning methods (e.g., HALP, Isomorphic pruning)
successors: []
impact: ⭐⭐⭐⭐
status: reading
---

# Summary
**MDP (Multi-Dimensional Pruning)** introduces a unified pruning framework that optimizes multiple structural dimensions together (e.g., channels, attention heads, query/key dimensions, embeddings, and blocks) while enforcing an explicit latency target. Instead of relying on FLOPs or simple linear latency approximations, the method builds a more accurate latency model and solves pruning as a constrained optimization problem.

# Key Idea
> Treat network pruning as a **joint multi-dimensional optimization problem under real latency constraints**, not as independent one-dimensional sparsification.

# Method
- **Pruning space**: jointly searches over several architectural dimensions across both CNNs and Transformers.
- **Latency modeling**: uses a richer latency estimator to capture non-linear interactions between pruned dimensions.
- **Optimization formulation**: frames pruning as a **Mixed-Integer Nonlinear Program (MINLP)** to maximize accuracy subject to latency constraints.
- **Deployment alignment**: explicitly targets wall-clock latency instead of proxy metrics alone.

# Results (reported)
- On **ImageNet / ResNet-50**, MDP reports about **28% speed-up** and **+1.4 top-1 accuracy** compared with prior latency-aware baselines such as HALP.
- Against transformer pruning baselines (e.g., **Isomorphic**), MDP reports an additional **~37% acceleration** with **+0.7 top-1 accuracy**.
- Gains are strongest in high-pruning/high-acceleration regimes where one-dimensional pruning often collapses.

# Why It Matters
- Moves pruning practice from “FLOPs reduction” to **hardware-relevant latency optimization**.
- Handles **Transformer-specific coupling effects** (heads, Q/K dims, embeddings, blocks) that simple models miss.
- Offers a shared framework across **CNN and ViT-style** architectures.

# Strengths
- Unified formulation for multiple pruning granularities.
- Better trade-off between accuracy and runtime in constrained deployment settings.
- Strong empirical improvements in aggressive compression regimes.

# Limitations / Open Questions
- MINLP-based search can be computationally expensive and sensitive to solver/config choices.
- Latency model portability across hardware backends remains an open practical issue.
- Reported gains depend on accurate profiling; mismatch between profiling and deployment stack can reduce benefits.

# Practical Takeaways
- For real-time vision deployment, optimize directly for **latency budgets** instead of only FLOPs/parameters.
- Jointly pruning several dimensions can outperform channel-only or head-only pruning.
- Profiling quality is critical: poor latency models can negate optimization gains.

# Educational Connections
## Undergraduate-level intuition
- FLOPs are not the same as runtime.
- Different model components (channels, heads, blocks) interact; removing one part can change the efficiency of others.

## Graduate-level lens
- Constrained architecture search with mixed discrete-continuous decisions.
- MINLP as a principled way to encode combinatorial pruning under deployment constraints.
- Multi-objective trade-off: accuracy vs latency under hardware-dependent response surfaces.

# My Notes
- This paper is a useful bridge between classic structural pruning and hardware-aware neural architecture optimization.
- Key conceptual shift: optimize over **interacting dimensions jointly**; avoid independent greedy pruning passes.
- Worth testing with quantization pipelines: MDP-style pruning + post-training quantization may yield better edge deployment points.

# Citation
Sun, X., Lakshmanan, B., Shen, M., Lan, S., Chen, J., & Alvarez, J. M. (2025). *MDP: Multidimensional Vision Model Pruning with Latency Constraint*. arXiv:2504.02168.
