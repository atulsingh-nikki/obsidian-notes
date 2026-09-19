---
title: "DINO: Emerging Properties in Self-Supervised Vision Transformers"
aliases:
  - DINO (2021)
authors:
  - Mathilde Caron
  - Hugo Touvron
  - Ishan Misra
  - Hervé Jégou
  - Julien Mairal
  - Piotr Bojanowski
  - Armand Joulin
year: 2021
venue: ICCV 2021
doi: "10.48550/arXiv.2104.14294"
code: "https://github.com/facebookresearch/dino"
dataset:
  - ImageNet (unsupervised pretraining, fine-tuning)
tags:
  - self-supervised-learning
  - vision-transformers
  - representation-learning
  - computer-vision
  - unsupervised-pretraining
  - non-contrastive-learning
arxiv: https://arxiv.org/abs/2104.14294
related:
  - "[[BYOL (2020)]]"
  - "[[SimCLR (2020)]]"
  - "[[Vision Transformer (ViT, 2020)]]"
  - "[[CLIP (2021)]]"
  - "[[Self-Distillation]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
DINO introduced a **self-supervised learning framework for Vision Transformers (ViTs)** based on self-distillation without labels. It showed that ViTs trained with non-contrastive SSL naturally learn **emergent properties** like object segmentation without explicit supervision.

# Key Idea (one-liner)
> Use a student–teacher self-distillation framework with Vision Transformers to learn meaningful, transferable representations — no labels, no negatives.

# Method
- **Architecture**:
  - Student $g_{\theta_s}$ and teacher $g_{\theta_t}$, identical architecture (ViT backbone $f$ + 3-layer MLP projection head $h$ with a weight-normalized, $\ell_2$-bottlenecked output layer, $K=65536$ dims). No predictor, no batch norm anywhere (ViTs don't use BN by default, so DINO stays fully BN-free).
  - Teacher is **not** a separate learned network — it is built online as an exponential moving average (EMA) of the student: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$, with $\lambda$ on a cosine schedule from 0.996 → 1.
- **Training**:
  - Multi-crop: 2 global views (224², covering >50% of the image) + several local views (96², <50%). All crops go through the student; only the 2 global views go through the teacher — encouraging local-to-global correspondence.
  - Student predicts the (stop-gradient) teacher's output distribution for every pair of different views.
- **Loss**:
  - Each network outputs a $K$-dim distribution via temperature softmax: $P_s(x) = \text{softmax}(g_{\theta_s}(x)/\tau_s)$, similarly $P_t$ with $\tau_t$.
  - Minimize cross-entropy $H(P_t(x), P_s(x')) = -P_t(x)\log P_s(x')$ over all global/local view pairs $x \ne x'$. $\tau_s = 0.1$; $\tau_t$ warms up linearly from 0.04 → 0.07 over the first 30 epochs.
- **Avoiding collapse — centering + sharpening (no negatives, no contrastive loss, no clustering):**
  - Centering: teacher logits get a bias $c$ subtracted, updated as an EMA of batch-mean teacher outputs: $c \leftarrow mc + (1-m)\frac{1}{B}\sum_i g_{\theta_t}(x_i)$. Prevents one dimension from dominating, but pulls the output toward uniform.
  - Sharpening: low $\tau_t$ sharpens the teacher's distribution — the opposite failure mode.
  - Together they cancel out and are *sufficient* to avoid collapse (Sinkhorn-Knopp / batch softmax work about as well but aren't required).
- **Optimization**: AdamW, batch size 1024, base LR $0.0005 \times \text{batch}/256$ with 10-epoch warmup + cosine decay; weight decay cosine from 0.04 → 0.4. BYOL-style augmentations (color jitter, Gaussian blur, solarization).
- **Patch size matters more than model size**: shrinking patches (16→8→5) improves features substantially more than scaling params, at the cost of throughput (ViT-S/8 runs at 180 im/s vs. 1007 im/s for /16; /5 drops to 44 im/s).

# Results
(ImageNet-1k, top-1 accuracy; “im/s” = throughput on a V100)
- **Same-architecture comparison (ViT-S/16, 21M params):** DINO 77.0% linear / 74.5% k-NN — beats BYOL, MoCo-v2, SwAV by **+3.5%** linear and **+7.9%** k-NN with the identical backbone. With ResNet-50 DINO is on par with SwAV/BYOL (75.3% linear / 67.5% k-NN), i.e. the gain is ViT-specific.
- **k-NN ≈ linear probe**, only with ViT: 74.5% vs 77.0% — a basic weighted 20-NN classifier on frozen features nearly matches a trained linear head. This does *not* happen with ResNet-50 or with other SSL methods on ViT.
- **Best full result:** ViT-B/8, 8×8 patches → **80.1% linear / 77.4% k-NN**, with 10× fewer params and 1.4× faster inference than the prior convnet SOTA (SimCLRv2 + wide ResNet-152).
- **Budget setting:** ViT-S/16 trained on two 8-GPU machines for 3 days reaches 76.1% linear — already ahead of comparable-size convnet SSL systems at a fraction of the compute.
- **Emergent segmentation**: thresholding the last-layer [CLS] self-attention (keep 60% of the mass) against PASCAL VOC12 ground truth gives Jaccard similarity 45.9 (DINO ViT-S/8) vs 27.3 (supervised ViT-S/8, same architecture/data) — supervision alone doesn't produce this; it's specific to the self-supervised objective.
- **Video object segmentation (DAVIS-2017)**, zero-shot nearest-neighbor propagation of frozen patch tokens: $\mathcal{J\&F}_m$ = 69.9 (ViT-S/8) / 71.4 (ViT-B/8) — competitive with methods purpose-built for the task, despite DINO never training on video or for dense prediction.
- **Image retrieval / copy detection**: DINO ViT-B/8 features beat supervised ViT-B/16 features on Oxford/Paris retrieval (mAP) and Copydays copy-detection (85.5 vs 76.4 mAP), and transfer better than supervised pretraining when fine-tuned on other classification datasets (Table 6 in the paper, +1–2% over supervised ViT across the board).
- **Ablations that matter (Table 7):** removing the momentum encoder collapses training to 0.1% accuracy — nothing else (multi-crop, CE loss, predictor) can substitute for it. Multi-crop is the second-most critical ingredient (+3.4% linear when added). A BYOL-style predictor barely moves the needle for DINO (unlike BYOL, where it's required to prevent collapse).

# Why it Mattered
- Extended non-contrastive SSL (BYOL/SimSiam) to **Transformers**.
- Proved that ViTs can learn **semantic grouping** (objects, parts) without labels.
- Advanced understanding of emergent attention properties in Transformers.
- Inspired later ViT-based SSL (EsViT, iBOT, MAE).

# Architectural Pattern
- [[Vision Transformer (ViT, 2020)]] as backbone.
- [[Self-Distillation]] via student–teacher with EMA.
- [[Multi-Crop Augmentation]] → strong regularization.
- [[Non-Contrastive SSL]] → collapse avoidance by asymmetry.

# Connections
- **Predecessors**:
  - [[BYOL (2020)]] — non-contrastive SSL with momentum teacher.
  - [[ViT (2020)]] — patch-based Transformer for vision.
- **Contemporaries**:
  - SwAV (2020) — clustering-based SSL.
- **Successors**:
  - [[iBOT (2021)]] — masked image modeling + DINO.
  - [[MAE (2021)]] — masked autoencoders for ViTs.
- **Influence**:
  - Established ViT SSL as viable alternative to CNN SSL.
  - Triggered research on **emergent segmentation from attention maps**.

# Implementation Notes
- Temperature parameter crucial to avoid representation collapse.
- EMA coefficient ~0.99–0.999 stabilizes training.
- Multi-crop strategy provides diversity of views.
- Pretrained weights available for ViT-S/16, ViT-B/16.

# Critiques / Limitations
- Compute-heavy (large ViTs + big batches).
- Sensitive to hyperparameters (temperature, momentum).
- Interpretability of emergent segmentation not fully understood.
- Still slower to train compared to CNN SSL baselines.

# Repro / Resources
- Paper: [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Official code: [Facebook Research DINO](https://github.com/facebookresearch/dino)
- Dataset: [[ImageNet]]
- Pretrained checkpoints widely available.
- Companion blog post: [DINO — Self-Distillation With No Labels](https://atulsingh-nikki.github.io/obsidian-notes/2026/09/19/dino-self-distillation-no-labels-vision-transformers/)

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**
  - Embedding vectors for image patches.
  - Matrix multiplications in Transformer attention.

- **Probability & Statistics**
  - Softmax in attention layers and output distributions.
  - Cross-entropy loss between student and teacher.

- **Calculus**
  - Gradients in EMA updates and attention layers.
  - Optimization with temperature scaling.

- **Signals & Systems**
  - Image patches as discrete tokens (sampling).
  - Multi-crop = multiple signal perturbations.

- **Data Structures**
  - Token sequences for patches.
  - Embedding dictionaries for teacher/student outputs.

- **Optimization Basics**
  - SGD/AdamW optimizers.
  - Momentum EMA updates.

---

## Postgraduate-Level Concepts
- **Advanced Optimization**
  - Avoiding collapse in non-contrastive SSL.
  - EMA updates as stability mechanism.
  - Multi-view consistency training.

- **Numerical Methods**
  - Efficient distributed training of ViTs.
  - Temperature annealing schedules.

- **Machine Learning Theory**
  - Self-distillation framework.
  - Non-contrastive SSL vs contrastive paradigms.
  - Emergent grouping in attention maps.

- **Computer Vision**
  - Learned features transferable to detection/segmentation.
  - Unsupervised semantic segmentation from attention.
  - Relevance to large-scale pretraining.

- **Neural Network Design**
  - ViT as SSL backbone.
  - Student–teacher dual architecture.
  - Multi-crop augmentation pipeline.

- **Transfer Learning**
  - Fine-tuning pretrained DINO ViTs on downstream tasks.
  - Linear probing vs full fine-tuning.
  - Cross-domain transfer.

- **Research Methodology**
  - Ablations on temperature, momentum, augmentations.
  - Benchmarks on ImageNet.
  - Visualizations of attention maps.

---

# My Notes
- How this connects to my projects: relevant to any pipeline needing free segmentation/attention masks from a frozen backbone (e.g., object/mask-refinement work) without training a dedicated segmentation head.
- Open questions: why does k-NN performance on ViT features track linear-probe performance so closely, when it doesn't for convnets or other SSL methods on ViT?
- Possible extensions: iBOT and MAE both build directly on this framework (masked-image-modeling + self-distillation, and masked autoencoding respectively) \u2014 worth comparing DINOv2's scaled-up recipe against this original.
