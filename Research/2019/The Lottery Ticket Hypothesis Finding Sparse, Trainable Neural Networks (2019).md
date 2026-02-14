---
title: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (2019)"
aliases:
  - Lottery Ticket Hypothesis
authors:
  - Jonathan Frankle
  - Michael Carbin
year: 2019
venue: "ICLR"
arxiv: "https://arxiv.org/abs/1803.03635"
tags:
  - paper
  - pruning
  - sparse-training
  - optimization
fields:
  - deep-learning
status: "read"
---

# Summary
Dense, randomly initialized neural networks contain smaller subnetworks ("winning tickets") that, when trained in isolation **with their original initialization**, can match or exceed the test accuracy of the full network in a similar number of iterations.

# Method in Brief
1. Train a dense network to convergence.
2. Remove a fraction of the smallest-magnitude weights (pruning).
3. Reset surviving weights to their original initialization values.
4. Retrain the pruned network.
5. Repeat pruning + reset iteratively to find very sparse winning tickets.

# Why it Matters
- Challenged the assumption that large dense models are strictly necessary for strong performance.
- Motivated broad follow-up work in pruning, sparse training, and initialization-sensitive optimization.
- Helped shape later research directions on efficient deep learning and model compression.

# Key Takeaways
- **Initialization matters:** the same sparse mask with random re-initialization performs much worse.
- **Sparsity can help:** carefully identified sparse subnetworks can train faster and generalize competitively.
- **Training dynamics matter:** winning tickets depend on both the mask and the early optimization trajectory.

# Limitations
- Original experiments were strongest on relatively small-to-medium-scale settings.
- Finding winning tickets can be computationally expensive due to repeated train-prune-reset cycles.
- Results are sensitive to architecture, optimizer settings, and pruning schedule.

# Follow-up Concepts
- Iterative magnitude pruning (IMP)
- One-shot pruning vs iterative pruning
- Rewinding (resetting to early checkpoints instead of iteration 0)
- Dynamic sparse training and sparse-from-scratch methods
