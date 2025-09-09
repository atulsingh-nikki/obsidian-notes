---
title: "MAE: Masked Autoencoders Are Scalable Vision Learners"
authors:
  - Kaiming He
  - Xinlei Chen
  - Saining Xie
  - Yanghao Li
  - Piotr Dollár
  - Ross Girshick
year: 2021
venue: CVPR 2022 (arXiv 2021)
dataset:
  - ImageNet (unsupervised pretraining, fine-tuning)
tags:
  - self-supervised-learning
  - vision-transformers
  - masked-autoencoders
  - representation-learning
  - computer-vision
  - unsupervised-pretraining
arxiv: https://arxiv.org/abs/2111.06377
related:
  - "[[Masked Language Models (BERT)]]"
  - "[[BYOL Bootstrap Your Own Latent]]"
  - "[[SimCLR A Simple Framework for Contrastive Learning of Visual Representations]]"
  - "[[Vision Transformer (ViT) An Image is Worth 16×16 Words|VIT-2020]]"
  - "[[DINO Emerging Properties in Self-Supervised Vision Transformers]]"
---

# Summary
MAE proposed a **masked image modeling framework** for self-supervised learning of Vision Transformers. Inspired by BERT in NLP, MAE trains a ViT to reconstruct missing patches from masked inputs. This simple pretext task leads to strong scalable representations that transfer well to downstream vision tasks.

# Key Idea (one-liner)
> Mask random image patches, encode the visible ones with a ViT, and train a lightweight decoder to reconstruct the missing patches.

# Method
- **Masking**:
  - Randomly mask 75% of image patches.
  - Input: only 25% of patches are visible to encoder.
- **Architecture**:
  - **Encoder**: ViT processes visible patches only.
  - **Decoder**: lightweight Transformer reconstructs full image from encoded + mask tokens.
- **Loss**:
  - Mean squared error (MSE) on pixel values of reconstructed patches.
- **Efficiency**:
  - Encoder processes fewer tokens (masked input) → scalable training.
- **Training**:
  - Pretrain on ImageNet (unsupervised).
  - Fine-tune on supervised downstream tasks.

# Results
- MAE-pretrained ViTs achieve state-of-the-art on ImageNet classification.
- Strong transfer to detection, segmentation (COCO, ADE20K).
- Very scalable — benefits from larger models (ViT-L, ViT-H).
- Outperforms many contrastive and distillation-based SSL methods.

# Why it Mattered
- Brought **masked autoencoding** to vision, directly inspired by BERT.
- Efficient and scalable — processes only partial inputs.
- Paved way for generative SSL approaches (e.g., iBOT, MaskFeat).
- Established MAE as one of the simplest yet strongest SSL baselines for ViTs.

# Architectural Pattern
- [[Vision Transformer (ViT, 2020)]] as encoder backbone.
- [[Masked Autoencoder]] → mask input patches, reconstruct missing ones.
- [[Lightweight Decoder]] → reconstruction head.
- [[Self-Supervised Pretraining → Fine-Tuning]] pipeline.

# Connections
- **Predecessors**:
  - [[BERT (2018)]] → masked language modeling.
  - [[SimCLR (2020)]], [[BYOL (2020)]], [[DINO (2021)]] → SSL for vision.
- **Contemporaries**:
  - iBOT (2021) → masked prediction + distillation.
- **Successors**:
  - [[MaskFeat (2022)]] → masking applied to features.
  - [[BEiT (2021)]] → masked image modeling with discrete tokens.
- **Influence**:
  - Foundation for many vision-language pretraining methods (BEiT, Flamingo, PaLI).
  - Inspired generative masked modeling for multimodal LMMs.

# Implementation Notes
- High mask ratio (75%) critical — forces meaningful representation.
- Encoder processes only visible tokens → training efficiency.
- Decoder can be small/lightweight → no need for symmetry.
- Pretrained weights available (ViT-Base, Large, Huge).

# Critiques / Limitations
- Pixel-level reconstruction may not capture high-level semantics.
- Decoder discarded after pretraining → wasted compute during training.
- Pretraining still compute-intensive for large models.
- Sensitive to choice of patch size and mask ratio.

# Repro / Resources
- Paper: [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Official PyTorch implementation (Facebook/Meta AI).
- Pretrained weights available in timm & HuggingFace.
- Dataset: [[ImageNet]]

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Embedding masked tokens as vectors.
  - Matrix multiplication in ViT layers.

- **Probability & Statistics**
  - Random masking as sampling.
  - Reconstruction error as expectation minimization.

- **Calculus**
  - Gradients through masked encoder–decoder pipeline.
  - Loss as squared error (MSE).

- **Signals & Systems**
  - Masking patches = undersampling signal.
  - Reconstruction = signal interpolation.

- **Data Structures**
  - Image patches as token sequences.
  - Mask tokens filling missing data.

- **Optimization Basics**
  - SGD/AdamW optimization.
  - Pretraining + fine-tuning pipeline.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Effect of mask ratio on learning stability.
  - Training with partial input tokens.
  - Gradient stability in deep ViTs.

- **Numerical Methods**
  - Efficient batching with masked tokens.
  - Complexity reduced by 75% due to masking.
  - Decoder efficiency vs accuracy trade-off.

- **Machine Learning Theory**
  - Connection to denoising autoencoders.
  - Inductive bias: masked prediction forces context learning.
  - Relation to information bottleneck.

- **Computer Vision**
  - Learned features generalize across detection/segmentation.
  - Reconstruction as proxy for scene understanding.
  - Patch-level vs pixel-level supervision.

- **Neural Network Design**
  - Asymmetric encoder–decoder.
  - ViT encoder, lightweight decoder.
  - Positional encoding critical for patch alignment.

- **Transfer Learning**
  - Pretraining on ImageNet unsupervised.
  - Fine-tuning vs linear probe evaluation.
  - Cross-domain generalization.

- **Research Methodology**
  - Ablations on mask ratio, decoder size, patch size.
  - Benchmarks: ImageNet, COCO, ADE20K.
  - Comparisons vs contrastive SSL (SimCLR, BYOL, DINO).
