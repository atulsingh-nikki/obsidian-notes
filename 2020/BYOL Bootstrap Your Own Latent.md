---
title: "BYOL: Bootstrap Your Own Latent"
aliases:
  - BYOL (2020)
authors:
  - Jean-Bastien Grill
  - Florian Strub
  - Florent Altché
  - Corentin Tallec
  - Pierre Richemond
  - Elena Buchatskaya
  - Carl Doersch
  - Bernardo Avila Pires
  - Zhaohan Guo
  - Mohammad Gheshlaghi Azar
  - Bilal Piot
  - Koray Kavukcuoglu
  - Rémi Munos
  - Michal Valko
year: 2020
venue: NeurIPS 2020
dataset:
  - ImageNet (unsupervised pretraining, fine-tuning)
tags:
  - self-supervised-learning
  - representation-learning
  - contrastive-learning
  - computer-vision
  - deep-learning
  - unsupervised-pretraining
arxiv: https://arxiv.org/abs/2006.07733
related:
  - "[[MoCo (2019)]]"
  - "[[SimCLR (2020)]]"
  - "[[SimSiam (2021)]]"
  - "[[CLIP (2021)]]"
  - "[[ResNet (2015)]]"
---

# Summary
BYOL introduced a surprising result: **self-supervised representation learning without negative samples**. It uses two neural networks — an **online network** and a **target network** — trained with a bootstrap mechanism. The online network predicts the representation of the target network for another augmented view of the same image. Over time, this process yields rich, transferable representations.

# Key Idea (one-liner)
> Learn by predicting the representation of another augmented view of the same image — no negative samples required.

# Method
- **Networks**:
  - **Online network**: encoder + projection head + prediction head.
  - **Target network**: encoder + projection head (no predictor).
- **Momentum update**: target network updated as exponential moving average of online network weights.
- **Loss**:
  - Symmetric loss between online prediction and target projection.
  - No contrastive negatives — only positives (two views of same image).
- **Augmentations**: strong data augmentations (cropping, color jitter, blur).
- **Backbone**: typically ResNet-50.

# Results
- Achieved ImageNet linear evaluation accuracy rivaling SimCLR.
- Outperformed SimCLR under comparable compute budgets.
- Showed negatives are not strictly necessary for self-supervised learning.
- Representations transferred well to downstream tasks (detection, segmentation).

# Why it Mattered
- Broke the assumption that contrastive learning *requires negatives*.
- Demonstrated power of momentum updates and prediction heads.
- Influenced follow-ups like SimSiam, DINO, BYOL-Audio.
- Helped unify ideas across self-distillation, consistency learning, and SSL.

# Architectural Pattern
- [[Momentum Encoder]] → target network updated by EMA.
- [[Prediction Head]] → avoids collapse by asymmetric architecture.
- [[Data Augmentation]] → critical for invariance.
- [[ResNet (2015)]] → standard backbone.

# Connections
- **Predecessors**:
  - [[MoCo (2019)]] — momentum encoder + negatives.
  - [[SimCLR (2020)]] — large-batch contrastive.
- **Contemporaries**:
  - [[SwAV (2020)]] — clustering-based SSL.
- **Successors**:
  - [[SimSiam (2021)]] — simplified BYOL without momentum encoder.
  - [[DINO (2021)]] — self-distillation with vision Transformers.
- **Influence**:
  - Inspired SSL in multimodal settings (e.g., audio, video, cross-modal).
  - Foundation for non-contrastive SSL methods.

# Implementation Notes
- Needs strong augmentations, otherwise collapse risk.
- Momentum parameter crucial (e.g., 0.99–0.999).
- Prediction head is key — removing it leads to representational collapse.
- Works with modest batch sizes (unlike SimCLR).

# Critiques / Limitations
- Theoretical explanation for avoiding collapse not fully understood (later explored in SimSiam).
- Sensitive to augmentation choices and optimizer settings.
- Heavier architecture (requires predictor in online branch).

# Repro / Resources
- Paper: [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Official TensorFlow implementation (DeepMind).
- PyTorch reimplementations available (solo-learn, VISSL).
- Dataset: [[ImageNet]]

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Vector embeddings for online vs target views.
  - Projection + prediction mappings.

- **Probability & Statistics**
  - Implicit learning of invariances via data augmentation.
  - No explicit negatives → different distributional assumption than contrastive.

- **Calculus**
  - Gradient flow through asymmetric online branch.
  - EMA update as weighted moving average (not gradient-based).

- **Signals & Systems**
  - Data augmentations as input signal perturbations.
  - EMA update as low-pass filter on weights.

- **Data Structures**
  - Paired augmented views.
  - Parameter queues updated via EMA.

- **Optimization Basics**
  - SGD/Adam with momentum.
  - Collapse prevention via predictor.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Non-contrastive SSL stability.
  - EMA update stabilizes training dynamics.
  - Avoiding trivial solutions without negatives.

- **Numerical Methods**
  - EMA update equations (θ_target ← m·θ_target + (1–m)·θ_online).
  - Memory–compute trade-offs of dual networks.

- **Machine Learning Theory**
  - Collapse avoidance via architectural asymmetry.
  - Relation to knowledge distillation and teacher–student training.
  - Contrastive vs non-contrastive paradigms.

- **Computer Vision**
  - Learned features transfer to detection, segmentation.
  - Strong results with ImageNet pretraining.
  - Invariance from augmentations.

- **Neural Network Design**
  - Dual-network setup (online vs target).
  - Predictor as asymmetry source.
  - Momentum update as stabilizer.

- **Transfer Learning**
  - Fine-tuning BYOL features on downstream datasets.
  - Linear probe evaluation vs full fine-tuning.

- **Research Methodology**
  - Ablations: effect of predictor, momentum, augmentations.
  - Comparisons with SimCLR/MoCo.
  - Benchmarks on ImageNet.
