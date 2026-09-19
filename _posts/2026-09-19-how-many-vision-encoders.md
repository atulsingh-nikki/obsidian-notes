---
layout: post
title: "How Many Vision Encoders Are There?"
description: "A practical map of the open vision-encoder ecosystem: classic backbones, foundation encoders, vision-language models, and task specialists."
tags: [computer-vision, vision-encoders, foundation-models, transformers]
---

A surprisingly difficult question in computer vision is: **how many vision encoders are there?**

The short answer is that nobody can give one exact number. The phrase *vision encoder* covers several different things: a ResNet trained for ImageNet, a self-supervised foundation model such as DINOv2, the image tower inside CLIP, and the visual front end of a multimodal language model. Public model hubs also count fine-tuned checkpoints and quantized copies as separate models.

A useful answer is therefore not a census. It is a map.

## Three levels of counting

There are at least three different levels hiding inside the question. They are not mutually exclusive categories; they are different ways of counting the same ecosystem:

1. **Architectures:** the design families, such as ResNet, ViT, Swin, and ConvNeXt.
2. **Pretraining families:** models trained with a particular objective and dataset, such as CLIP, DINOv2, or MAE.
3. **Checkpoints:** downloadable weights, fine-tunes, conversions, and quantized variants.

For example, DINOv2 is a **pretraining family** built with a Vision Transformer **architecture**, and each released model size or fine-tune is a separate **checkpoint**. The same architecture can therefore appear in several pretraining families, and one pretraining family can produce many checkpoints.

At the architecture level, the number is manageable: a few dozen influential families. At the checkpoint level, the number is already in the thousands, and public repositories continue to grow every day. A search of Hugging Face's narrow image-feature-extraction category alone returns hundreds of entries, before counting models published under different task labels or stored outside that hub.

So the useful working estimate is:

- **A few dozen major families** worth learning.
- **Hundreds to thousands of serious public model variants.**
- **Tens of thousands of checkpoints** if fine-tunes, conversions, and task-specific derivatives are included.

These three dimensions connect to the practical taxonomy in different ways:

| Counting dimension | What it tells us | Where it appears below |
|---|---|---|
| Architecture | How the encoder is built | Classical backbones and Vision Transformers |
| Pretraining family | How the representation learned visual structure | Self-supervised, vision-language, and generalist encoders |
| Checkpoint | Which concrete weights can be downloaded or deployed | Every section, including specialist models |

The taxonomy below is therefore not a second competing classification. It is the practical expansion of the table: first we identify the architecture or training lineage, then we choose the particular checkpoint for the task.

## A practical taxonomy by model role

We can now expand the connection one step further. The first sections below begin with architecture, move through pretraining family, and end with the task role that determines why a checkpoint exists. These groups can overlap. For example, a ViT can be a supervised backbone, a DINOv2 encoder, or the image tower inside a vision-language model.

### 1. Classical supervised backbones

These models were trained mainly for image classification and then reused as feature extractors.

- [ResNet]({{ site.baseurl }}/Research/2015/Deep%20Residual%20Learning%20for%20Image%20Recognition.html) and [ResNeXt]({{ site.baseurl }}/Research/2017/ResNeXt%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks%20(2017).html)
- [EfficientNet]({{ site.baseurl }}/Research/2019/EfficientNet%20Rethinking%20Model%20Scaling%20for%20Convolutional%20Neural%20Networks.html)
- RegNet
- [ConvNeXt]({{ site.baseurl }}/Research/2022/ConvNeXt%20A%20ConvNet%20for%20the%202020s%20(2022).html)
- [Inception]({{ site.baseurl }}/Research/2014/GoogLeNet%20Inception%20v1%20Going%20Deeper%20with%20Convolutions.html) and [DenseNet]({{ site.baseurl }}/Research/2017/Densely%20Connected%20Convolutional%20Networks%20(2017).html)

They remain valuable because they are fast, well understood, and easy to deploy. Their representation is usually strongest near the distribution and label space used during supervised training.

### 2. Vision Transformers

ViT showed that a transformer could process an image as a sequence of patches. Later families improved its data efficiency, locality, scale, or training stability.

- [ViT]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html)
- [DeiT]({{ site.baseurl }}/Research/2021/DeiT%20Training%20Data-Efficient%20Image%20Transformers%20%26%20Distillation%20through%20Attention%20(2021).html)
- [Swin Transformer]({{ site.baseurl }}/Research/2021/Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows%20(2021).html)
- BEiT
- PVT
- MaxViT
- EVA and EVA-CLIP

The key shift was from hand-designed convolutional locality toward learned attention over patch tokens. This made scaling and transfer learning much more natural, but it also increased the importance of data, positional encoding, and training recipe.

### 3. Self-supervised foundation encoders

These models learn visual structure without requiring a class label for every image. They are often used as frozen or lightly fine-tuned backbones for downstream tasks.

- [MoCo]({{ site.baseurl }}/Research/2020/MoCo%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Representation%20Learning.html) and MoCo v3
- [SimCLR]({{ site.baseurl }}/Research/2020/SimCLR%20A%20Simple%20Framework%20for%20Contrastive%20Learning%20of%20Visual%20Representations.html) and [BYOL]({{ site.baseurl }}/Research/2020/BYOL%20Bootstrap%20Your%20Own%20Latent.html)
- [MAE]({{ site.baseurl }}/Research/2021/MAE%20Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.html)
- [DINO]({{ site.baseurl }}/Research/2021/DINO%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers.html) and [DINOv2]({{ site.baseurl }}/Research/2023/DINOv2%20Learning%20Robust%20Visual%20Features%20without%20Supervision%20(2023).html)
- iBOT
- EsViT
- DINOv3

DINO-style models are especially interesting because their features often preserve object parts, boundaries, and semantic grouping even though the training objective is not a conventional segmentation loss. MAE-style models, in contrast, learn by reconstructing masked image content and tend to benefit strongly from scale and fine-tuning.

### 4. Vision-language encoders

A vision-language encoder learns a shared representation for images and text. The classic example is CLIP, which aligns an image with its corresponding caption and separates it from mismatched captions.

- [CLIP]({{ site.baseurl }}/Research/2021/CLIP%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.html)
- OpenCLIP
- SigLIP and SigLIP 2
- ALIGN
- BLIP and BLIP-2 vision towers
- EVA-CLIP

These encoders are useful for zero-shot classification, retrieval, image search, and as the visual input to a larger multimodal system. Their strengths are semantic alignment and transfer across labels. Their weakness is that a representation optimized for image-text similarity may not preserve every detail needed for dense prediction or pixel-accurate editing.

### 5. Generalist multimodal encoders

The next category tries to support many visual tasks or modalities with one reusable model. Apple's work is a useful example of this direction.

[AIMV2](https://arxiv.org/abs/2411.14402), *Multimodal Autoregressive Pre-training of Large Vision Encoders*, describes a family of generalist vision encoders trained with a multimodal autoregressive objective. The paper reports strong results across image classification, localization, grounding, and multimodal image understanding. Its important claim is not merely that a larger encoder scores well, but that one visual representation can transfer across several task types.

Apple's [4M](https://machinelearning.apple.com/research/massively-multimodal) and [4M-21](https://machinelearning.apple.com/research/vision-model) take a different route. They train a unified model over many visual modalities and tasks. 4M-21 includes signals produced by specialist systems such as SAM, DINOv2, ImageBind, and 4DHumans. In this setup, specialist models are not simply competitors; they also become teachers whose outputs are distilled into a more general model.

This is one of the most important patterns in current vision research: **generalist models often absorb specialist knowledge rather than replacing specialist models outright.**

### 6. Task-specific specialists

A specialist is trained to be excellent at a narrower problem. Examples include:

- [SAM]({{ site.baseurl }}/Research/2023/Segment%20Anything%20(SAM,%202023).html) and [SAM 2]({{ site.baseurl }}/Research/2024/SAM%202%20Segment%20Anything%20in%20Images%20and%20Videos%20(2024).html) for promptable segmentation
- Mask2Former for segmentation and panoptic prediction
- [DETR]({{ site.baseurl }}/Research/2020/DETR%20End-to-End%20Object%20Detection%20with%20Transformers%20(2020).html) variants for object detection
- HRNet and ViTPose for pose estimation
- MiDaS and DPT for depth estimation
- [RAFT]({{ site.baseurl }}/Research/2020/RAFT%20Recurrent%20All-Pairs%20Field%20Transforms%20for%20Optical%20Flow%20(2020).html) for optical flow
- VideoMAE and InternVideo for video representation
- TrOCR, Donut, and document models for text and layout
- Medical, satellite, face, and industrial-inspection encoders

Specialists often win when the evaluation requires precise geometry, a specific domain, or a carefully optimized output head. Generalist models usually win on breadth, zero-shot transfer, and the cost of maintaining one system instead of many.

## Generalist versus specialist is not a simple ranking

It is tempting to ask which one is better. The more useful question is: **better for which constraint?**

| Constraint | Generalist encoder | Specialist encoder |
|---|---|---|
| Number of tasks | Strong | Narrower |
| Zero-shot transfer | Usually strong | Usually limited |
| Fine geometric detail | Can be weaker | Often strong |
| Domain-specific accuracy | Variable | Often strong |
| Deployment simplicity | One model can cover many tasks | Multiple models may be required |
| Training and maintenance | Centralized | Repeated per task |
| Interpretability of failure | Broad and harder to diagnose | Easier to localize |

A generalist encoder can be understood as an amortized investment: expensive broad pretraining buys reuse later. A specialist spends its capacity on a smaller target and can therefore be more efficient when the target is known.

## What should we actually learn?

Trying to memorize every checkpoint is a losing strategy. A better curriculum is to understand the transitions between objectives:

1. **Supervised CNNs:** learn what a strong task-trained representation looks like, starting with [ResNet]({{ site.baseurl }}/Research/2015/Deep%20Residual%20Learning%20for%20Image%20Recognition.html).
2. **[ViT]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html):** understand patch tokens, attention, and positional information.
3. **[CLIP]({{ site.baseurl }}/Research/2021/CLIP%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.html):** see how image-text alignment changes transfer behavior.
4. **[MAE]({{ site.baseurl }}/Research/2021/MAE%20Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.html) and [DINOv2]({{ site.baseurl }}/Research/2023/DINOv2%20Learning%20Robust%20Visual%20Features%20without%20Supervision%20(2023).html):** compare reconstruction-based and invariance-based self-supervision.
5. **[SAM]({{ site.baseurl }}/Research/2023/Segment%20Anything%20(SAM,%202023).html):** study how a general promptable model handles dense prediction.
6. **AIMV2 and 4M-21:** examine the move toward generalist and multimodal vision systems.

That sequence is more valuable than a list of model names because it explains why the ecosystem keeps producing new encoders: each family changes the balance between semantic breadth, spatial precision, data efficiency, compute, and deployment cost.

## The answer

There are too many public checkpoints to count meaningfully, but not too many foundational ideas to understand. The open ecosystem contains **dozens of major vision-encoder families, thousands of important variants, and many more derivative checkpoints**.

For practical computer-vision work, the central decision is rarely “which of the thousands should I download?” It is usually:

> Do I need a broad representation that can be adapted to many tasks, or a narrow representation optimized for one task and one data distribution?

That is the generalist-specialist trade-off in one sentence. The future is likely to contain both: generalist encoders for shared visual knowledge, plus specialist heads or specialist teachers for the precision that broad models often miss.

## Continue the vision-encoder series

- [Classical Supervised Vision Backbones]({{ site.baseurl }}/2026/09/19/classical-supervised-vision-backbones.html)
- [Vision Transformers]({{ site.baseurl }}/2026/09/19/vision-transformers.html)
- [Self-Supervised Foundation Encoders]({{ site.baseurl }}/2026/09/19/self-supervised-foundation-encoders.html)
- [Generalist Multimodal Encoders]({{ site.baseurl }}/2026/09/19/generalist-multimodal-encoders.html)
- [Task-Specific Vision Specialists]({{ site.baseurl }}/2026/09/19/task-specific-vision-specialists.html)
