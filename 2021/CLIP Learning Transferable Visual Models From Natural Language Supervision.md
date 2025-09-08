---
title: "CLIP: Learning Transferable Visual Models From Natural Language Supervision"
aliases:
  - CLIP (2021)
authors:
  - Alec Radford
  - Jong Wook Kim
  - Chris Hallacy
  - Aditya Ramesh
  - Gabriel Goh
  - Sandhini Agarwal
  - Girish Sastry
  - Amanda Askell
  - Pamela Mishkin
  - Jack Clark
  - Gretchen Krueger
  - Ilya Sutskever
year: 2021
venue: ICML 2021 (arXiv 2021)
dataset:
  - 400M image–text pairs scraped from the web
tags:
  - multimodal
  - vision-language
  - contrastive-learning
  - computer-vision
  - natural-language-processing
  - transfer-learning
  - zero-shot-learning
arxiv: https://arxiv.org/abs/2103.00020
related:
  - "[[SimCLR (2020)]]"
  - "[[MoCo (2019)]]"
  - "[[BYOL (2020)]]"
  - "[[Vision Transformer (ViT, 2020)]]"
  - "[[Contrastive Learning]]"
  - "[[Zero-Shot Learning]]"
---

# Summary
CLIP trains visual models using **natural language supervision** by pairing images with their text descriptions. Using a large dataset of **400M image–text pairs**, CLIP learns a shared embedding space where images and their captions align. This enables **zero-shot transfer**: given a natural language prompt, CLIP can classify images without task-specific training.

# Key Idea (one-liner)
> Train image and text encoders jointly with contrastive learning so that matching image–text pairs have similar embeddings.

# Method
- **Dual encoders**:
  - **Image encoder**: ResNet-50 or ViT.
  - **Text encoder**: Transformer (12-layer).
- **Training**:
  - Contrastive loss across a batch of image–text pairs.
  - Maximize similarity for true pairs, minimize for mismatched pairs.
- **Dataset**: 400M image–text pairs scraped from the internet.
- **Inference**:
  - Convert class labels into text prompts (“a photo of a dog”).
  - Zero-shot classification by finding closest embedding.

# Results
- Outperformed ImageNet supervised pretraining on many downstream tasks.
- Enabled **zero-shot learning** across hundreds of benchmarks.
- Showed surprising emergent properties (compositionality, robustness).
- Generalized well without task-specific supervision.

# Why it Mattered
- Shifted paradigm from supervised ImageNet pretraining → large-scale multimodal pretraining.
- Bridged computer vision and NLP into a unified model.
- Enabled **zero-shot recognition** at scale.
- Inspired successors (ALIGN, Florence, OpenCLIP, Flamingo, BLIP).

# Architectural Pattern
- [[Contrastive Learning]] → image–text alignment.
- [[Dual Encoder Architecture]] → separate image/text encoders.
- [[Vision Transformer (ViT, 2020)]] → alternative backbone to CNN.
- [[Zero-Shot Learning]] → classification via text prompts.

# Connections
- **Predecessors**:
  - [[SimCLR (2020)]], [[MoCo (2019)]] — contrastive SSL in vision.
- **Contemporaries**:
  - ALIGN (Google, 2021) — similar large-scale image–text contrastive training.
- **Successors**:
  - [[OpenCLIP]] — community-trained large-scale CLIP variants.
  - [[BLIP (2022)]] — multimodal pretraining with captions + image-text matching.
  - [[Flamingo (2022)]] — multimodal few-shot learner.
- **Influence**:
  - Foundation for generative multimodal models (e.g., DALL·E 2, Stable Diffusion).
  - Prompt engineering and multimodal alignment research.

# Implementation Notes
- Trained on 400M pairs → compute-heavy (distributed training).
- Prompt engineering crucial: phrasing affects zero-shot results.
- OpenCLIP provides pretrained public models (since original dataset not public).
- Works with both CNNs and Transformers as backbones.

# Critiques / Limitations
- Dataset not released → reproducibility gap.
- Biases from web-scale data baked into embeddings.
- Struggles with fine-grained or domain-specific tasks.
- Sensitive to prompt wording.

# Repro / Resources
- Paper: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Official models: OpenAI (partial release).
- Open-source reimplementations: OpenCLIP, LAION datasets.
- Datasets: LAION-400M, LAION-5B (community alternatives).

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Embedding vectors for images and text.
  - Cosine similarity for alignment.

- **Probability & Statistics**
  - Softmax over similarity scores.
  - Negative sampling across batch.

- **Calculus**
  - Gradient updates from contrastive loss.
  - Backpropagation across dual encoders.

- **Signals & Systems**
  - Text tokens as sequential signal input.
  - Image patches as spatial tokens (in ViT).

- **Data Structures**
  - Paired image–text datasets.
  - Embedding matrices.

- **Optimization Basics**
  - Contrastive loss optimization with SGD/Adam.
  - Large-batch distributed training.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Large-batch distributed training stability.
  - Temperature scaling in contrastive loss.
  - Balancing text vs image encoder updates.

- **Numerical Methods**
  - Efficient similarity computation (matrix multiplication).
  - Memory trade-offs in massive batch training.

- **Machine Learning Theory**
  - Cross-modal embedding alignment.
  - Zero-shot learning via shared representation.
  - Emergent properties of scale.

- **Computer Vision & NLP**
  - Joint vision–language representation.
  - Transfer across multimodal tasks.
  - Prompt-based inference.

- **Neural Network Design**
  - Dual encoders (image + text).
  - Use of ViT vs ResNet as image backbone.
  - Role of Transformer for text.

- **Transfer Learning**
  - Zero-shot classification via prompt engineering.
  - Fine-tuning CLIP embeddings for downstream tasks.
  - Domain adaptation via few-shot prompts.

- **Research Methodology**
  - Benchmarks: ImageNet, CIFAR, Oxford Pets, Flowers, SUN397, etc.
  - Evaluation across >30 datasets.
  - Ablations: effect of dataset scale, backbone choice.
