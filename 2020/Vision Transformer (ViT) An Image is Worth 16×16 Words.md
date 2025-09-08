---
title: "Vision Transformer (ViT): An Image is Worth 16×16 Words"
aliases:
  - VIT-2020
authors:
  - Alexey Dosovitskiy
  - Lucas Beyer
  - Alexander Kolesnikov
  - Dirk Weissenborn
  - Xiaohua Zhai
  - Thomas Unterthiner
  - Mostafa Dehghani
  - Matthias Minderer
  - Georg Heigold
  - Sylvain Gelly
  - Jakob Uszkoreit
  - Neil Houlsby
year: 2020
venue: ICLR 2021 (arXiv 2020)
dataset:
  - ImageNet-21k
  - JFT-300M
  - ImageNet-1k (fine-tuning)
tags:
  - computer-vision
  - transformers
  - image-classification
  - deep-learning
  - attention-mechanisms
  - transfer-learning
  - pretraining
arxiv: https://arxiv.org/abs/2010.11929
related:
  - "[[ResNet (2015)]]"
  - "[[DenseNet (2016)]]"
  - "[[Transformers (NLP, 2017)]]"
  - "[[DeiT (2021)]]"
  - "[[Hybrid CNN-Transformers]]"
  - "[[Self-Attention]]"
---

# Summary
ViT demonstrated that **pure Transformer architectures** (without convolutions) can match or exceed CNNs in image recognition — provided they are trained on very large datasets. Images are split into **fixed-size patches (16×16)**, linearly embedded, and fed into a Transformer encoder with **self-attention**. Pretraining on massive datasets (e.g., JFT-300M) followed by fine-tuning on ImageNet-1k yielded state-of-the-art results.

# Key Idea (one-liner)
> Treat an image as a sequence of patches and apply a standard Transformer encoder to perform vision tasks.

# Method
- **Patch embedding**: Split image into N patches (e.g., 224×224 → 14×14 patches of 16×16 pixels). Flatten + project to embedding vectors.
- **Position embeddings**: Added to patch embeddings to retain spatial order.
- **Transformer Encoder**: Standard NLP-style Transformer encoder:
  - Multi-head self-attention (MHSA).
  - Feed-forward MLP layers.
  - LayerNorm + residual connections.
- **Classification token (CLS)**: A learnable embedding that aggregates global information.
- **Pretraining**: Requires very large datasets (ImageNet-21k, JFT-300M).
- **Fine-tuning**: On downstream benchmarks like ImageNet-1k.

# Results
- Outperformed ResNets of similar size when trained on large datasets.
- On ImageNet-1k alone (without pretraining), CNNs still stronger.
- Pretraining + fine-tuning crucial to ViT success.
- Sparked a wave of vision Transformers (DeiT, Swin, PVT).

# Why it Mattered
- First proof that **convolutions are not necessary** for state-of-the-art vision.
- Showed **scaling laws** in vision similar to NLP.
- Shifted the field toward **attention-based architectures** in vision.
- Unified language + vision modeling approaches.

# Architectural Pattern
- [[Image Patches as Tokens]] → each patch like a “word.”
- [[Self-Attention]] → global context across all patches.
- [[Residual Connections]] → stabilize very deep models.
- [[Pretraining on Large Datasets]] → crucial for success.

# Connections
- **Predecessors**:
  - [[Transformers (NLP, 2017)]] — original sequence-to-sequence architecture.
  - [[ResNet (2015)]] — previous default backbone.
- **Successors**:
  - [[DeiT (2021)]] — data-efficient ViT, trained only on ImageNet-1k.
  - [[Swin Transformer (2021)]] — hierarchical vision Transformers.
  - [[Hybrid CNN-Transformers]] — combining local CNNs with global attention.
- **Influence**:
  - Transformers became dominant across CV tasks: detection, segmentation, generative modeling (Diffusion, LMMs).

# Implementation Notes
- Needs very large datasets (JFT-300M) or distillation (DeiT).
- Positional embeddings limit scalability to new resolutions (fixed length).
- Transfer learning crucial for small datasets.
- Still compute-heavy compared to CNNs.

# Critiques / Limitations
- Data-hungry → poor performance on small datasets without pretraining.
- Less inductive bias than CNNs (no translation equivariance).
- Computationally expensive (quadratic cost in attention).
- Resolution scaling is tricky due to patch embedding.

# Repro / Resources
- Paper: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Official JAX/Flax implementation (Google Research).
- PyTorch implementations (timm, HuggingFace).
- Pretrained models widely available.

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Matrix multiplications in attention layers.
  - Projection of image patches into embedding vectors.
  
- **Probability & Statistics**
  - Softmax attention weights as probability distributions.
  - Cross-entropy loss for classification.

- **Calculus**
  - Gradients through attention mechanisms.
  - Backprop in residual architectures.

- **Signals & Systems**
  - Patch embedding as downsampling.
  - Positional encodings as signal injection.

- **Data Structures**
  - Sequences of patches as token arrays.
  - Attention matrices as N×N graphs.

- **Optimization Basics**
  - SGD/Adam with weight decay.
  - Pretraining + fine-tuning cycle.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Training stability in very deep attention networks.
  - Gradient scaling and residual connections.
  - Learning rate warmup and schedule tuning.

- **Numerical Methods**
  - Quadratic complexity of attention O(N²).
  - Memory trade-offs in long sequences.
  - Efficient attention approximations (later methods).

- **Machine Learning Theory**
  - Inductive biases: CNNs vs Transformers.
  - Scaling laws: model/data size vs accuracy.
  - Transfer learning effectiveness.

- **Computer Vision**
  - Classification with global receptive field attention.
  - Benchmark comparisons against ResNets.
  - Generalization to detection/segmentation with ViT backbones.

- **Neural Network Design**
  - Patch embedding vs convolutional features.
  - CLS token as global aggregator.
  - Role of residual connections in deep Transformers.

- **Transfer Learning**
  - Pretraining on JFT-300M or ImageNet-21k.
  - Fine-tuning on ImageNet-1k and downstream tasks.
  - Knowledge distillation (DeiT).

- **Research Methodology**
  - Scaling dataset/model size experiments.
  - Ablations on patch size, depth, width.
  - Benchmarking against ResNet baselines.
