---
title: Deep Compression Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding (2015)
aliases:
  - Deep Compression
  - Deep Compression (2015)
  - Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding
authors:
  - Song Han
  - Huizi Mao
  - William J. Dally
year: 2015
venue: ICLR (workshop track)
doi: 10.48550/arXiv.1510.00149
arxiv: https://arxiv.org/abs/1510.00149
code: https://github.com/songhan/Deep-Compression-AlexNet
github: https://github.com/songhan/Deep-Compression-AlexNet
citations: 10,000+
dataset:
  - ImageNet (AlexNet, VGG-16)
  - MNIST (LeNet-300-100, LeNet-5)
tags:
  - paper
  - model-compression
  - pruning
  - quantization
  - huffman-coding
fields:
  - deep-learning
  - efficient-ai
related:
  - "[[Optimal Brain Damage (1989)]]"
  - "[[Deep Residual Learning for Image Recognition]]"
predecessors:
  - "[[Optimal Brain Damage (1989)]]"
successors:
  - "[[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (2017)]]"
  - "[[AMC AutoML for Model Compression and Acceleration on Mobile Devices (2018)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
Deep Compression (Han et al., 2015) introduced a practical **three-stage pipeline** to shrink trained neural networks by **35× to 49×** without hurting accuracy. The method combines weight pruning, shared-weight quantization, and Huffman coding to reduce storage and memory bandwidth, enabling deployment on mobile and embedded hardware.

# Key Idea
> First remove unimportant connections, then represent surviving weights with compact shared codebooks, and finally entropy-code everything for near-lossless storage.

# Method
- **Step 1: Network pruning**
  - Train the dense model normally.
  - Remove low-magnitude weights using threshold-based pruning.
  - Retrain/fine-tune sparse structure to recover accuracy.
- **Step 2: Trained quantization + weight sharing**
  - Cluster remaining non-zero weights (k-means per layer).
  - Replace many unique weights with shared centroid values.
  - Store only centroid indices (fewer bits) + compact codebook.
- **Step 3: Huffman coding**
  - Apply Huffman coding to quantized weight indices and sparse matrix metadata.
  - Exploit non-uniform symbol frequency for additional compression.
- Compression is evaluated on **AlexNet, VGG-16, LeNet-300-100, and LeNet-5**.

# Results
- **AlexNet**: ~240 MB → ~6.9 MB (**~35× smaller**) with no accuracy drop.
- **VGG-16**: ~552 MB → ~11.3 MB (**~49× smaller**) with no accuracy drop.
- **LeNet-5 / LeNet-300-100**: around **10×–12×** additional practical gains in already smaller settings.
- Major gain comes from reduced DRAM access, which often dominates energy usage during inference.

# Why it Mattered
- Established that neural network redundancy is large enough for aggressive post-training compression.
- Shifted optimization focus from only FLOPs to **memory footprint + bandwidth**.
- Became foundational for edge AI and later work in pruning/quantization-aware training and hardware co-design.

# Architectural Pattern
- Dense pretraining → structured/unstructured sparsification.
- Codebook-based parameter sharing.
- Entropy coding as a final systems layer.
- Pipeline mindset: combine multiple small wins for big end-to-end compression.

# Connections
- **Builds on**: earlier pruning work like [[Optimal Brain Damage (1989)]].
- **Influenced**: integer-only quantization, mixed precision, neural architecture search with compression constraints, and compiler/runtime support for sparse or low-bit models.

# Implementation Notes
- Unstructured sparsity can be hard to accelerate on general hardware despite high compression.
- Index overhead matters; compressed sparse formats must be designed carefully.
- Layer-wise sensitivity differs: fully connected layers compress heavily, while some convolutional layers are more sensitive.
- Retraining after each stage is critical for maintaining accuracy.
- **Why setting weights to zero does not “break differentiability”**:
  - In pruning, weights are typically multiplied by a binary mask \(m\in\{0,1\}\), so the effective parameter is \(\tilde w = m\odot w\).
  - For active connections (\(m=1\)), gradients flow normally.
  - For pruned connections (\(m=0\)), gradient to that weight is intentionally zero (it is frozen/removed), which is expected behavior, not a mathematical failure.
  - The network remains differentiable with respect to all remaining active parameters; training proceeds on the reduced parameter subspace.
  - In practice, Deep Compression prunes then fine-tunes, so optimization simply continues over surviving weights.

# Critiques / Limitations
- Compression ratio does not translate linearly to wall-clock speedup.
- Huffman coding is storage-focused; decoding overhead can reduce runtime benefit in some deployments.
- Pipeline is mostly post-training and manually tuned; less automatic than modern quantization-aware training and NAS-based compression.

# Repro / Resources
- [Paper link (arXiv)](https://arxiv.org/abs/1510.00149)
- [Original project page](https://hanlab.mit.edu/projects/deep-compression)
- [Reference implementation](https://github.com/songhan/Deep-Compression-AlexNet)

---

# Educational Connections

## Undergraduate-Level
- Sparse matrices and pointer/index representations.
- Clustering (k-means) for vector quantization.
- Prefix codes / entropy coding basics (Huffman coding).
- Trade-off between approximation error and compression ratio.

## Postgraduate-Level
- Compression-aware objective design and sensitivity analysis.
- Hardware-software co-design for sparse/low-bit inference.
- Rate-distortion style reasoning for neural parameter storage.
- Relationship between redundancy, generalization, and lottery-ticket style hypotheses.

---

# My Notes
- This paper is one of the clearest examples of **systems thinking in deep learning**.
- The 3-stage decomposition still maps well to modern deployment pipelines.
- Open question: for current transformer blocks, what is the best ordering between pruning, quantization, and distillation?
