---
layout: post
title: "Vision Transformers: From Image Patches to General-Purpose Encoders"
description: "A practical guide to Vision Transformers, from patch tokens and global attention to DeiT, Swin, and modern foundation-model backbones."
tags: [computer-vision, vision-encoders, transformers, attention, backbones]
---

Convolutional networks dominated computer vision because images have strong local structure. Nearby pixels are related, translation should usually preserve meaning, and the same detector can be reused across an image.

Vision Transformers (ViTs) took a different route. They split an image into patches, treat those patches as tokens, and use the Transformer architecture to learn how the tokens should interact.

That sounds like a small change, but it moved several important design decisions from fixed architectural assumptions into learned attention. It also connected visual representation learning to the scaling patterns that had already transformed natural-language processing.

## What is a Vision Transformer?

A Vision Transformer represents an image as a sequence rather than a feature map. For an image of height $H$ and width $W$, using patches of size $P \times P$, the number of image tokens is:

$$
N = \frac{H W}{P^2}.
$$

Each patch is flattened and projected into an embedding of dimension $D$. A learnable class token is often prepended, and positional information is added because self-attention does not know where a token came from by itself:

$$
Z_0 = [x_{\mathrm{cls}}; E(x_1); E(x_2); \ldots; E(x_N)] + P_{\mathrm{pos}}.
$$

The sequence then passes through standard Transformer encoder blocks. Each block contains multi-head self-attention, a feed-forward network, normalization, and residual connections. For classification, the final class-token representation is sent to a prediction head.

The key move is not that a patch is literally a word. It is that the model can apply a common sequence-processing primitive to visual units while learning their relationships from data.

## The design questions

Vision Transformer research can be understood as a search for better answers to five questions:

- How should an image become a sequence of tokens?
- How can attention remain affordable as resolution increases?
- How can a Transformer learn effectively with less labeled data?
- How should spatial hierarchy be represented for detection and segmentation?
- Which positional representation generalizes across image sizes and aspect ratios?

The major families made different trade-offs across these questions.

## 1. ViT: turn patches into tokens

[ViT]({{ site.baseurl }}/Research/2020/Vision%20Transformer%20(ViT)%20An%20Image%20is%20Worth%2016%C3%9716%20Words.html) showed that a mostly unchanged Transformer encoder could perform image classification. The model divides an image into fixed-size patches, linearly embeds them, adds positional embeddings, and processes the resulting sequence with self-attention.

Global attention lets every patch interact with every other patch in one layer. A patch representing a wheel can directly relate to a patch representing the rest of a vehicle, even when the two are far apart in the image.

The cost is sequence length. Standard self-attention builds an $N \times N$ attention matrix, so its memory and compute grow approximately as:

$$
O(N^2) = O\left(\frac{H^2 W^2}{P^4}\right).
$$

Smaller patches preserve more detail but create many more tokens. Larger patches are cheaper but discard fine spatial information.

**Main contribution:** a pure Transformer can be a competitive visual encoder when pretrained at sufficient scale.

**Trade-off:** the original model is data-hungry, computationally expensive at high resolution, and less naturally suited to dense prediction than a hierarchical CNN.

## 2. DeiT: make ViTs data-efficient

[DeiT]({{ site.baseurl }}/Research/2021/DeiT%20Training%20Data-Efficient%20Image%20Transformers%20%26%20Distillation%20through%20Attention%20(2021).html) addressed the most immediate weakness of the original ViT: it needed enormous pretraining datasets to work well.

DeiT showed that careful augmentation, regularization, training schedules, and knowledge distillation could make a Transformer competitive using ImageNet-1k. Its distillation token gives the student a second route to learn from a teacher model, alongside the ordinary class token.

This was an important lesson. ViT was not only a new architecture; it was also a demanding training recipe. Better optimization and supervision could recover some of the inductive bias that CNNs receive from their structure.

**Main contribution:** a practical recipe for training strong ViTs with ordinary labeled data.

**Trade-off:** distillation and aggressive regularization improve the recipe but do not remove the quadratic attention cost or the need for careful tuning.

## 3. Swin: restore hierarchy and locality

[Swin Transformer]({{ site.baseurl }}/Research/2021/Swin%20Transformer%20Hierarchical%20Vision%20Transformer%20using%20Shifted%20Windows%20(2021).html) changed the basic attention pattern. Instead of attending globally over all patches, it computes attention inside local non-overlapping windows. The windows shift in the next layer, allowing information to cross the previous window boundaries.

Swin also builds a hierarchy. Patch merging progressively reduces spatial resolution while increasing channel dimension, producing feature maps at several scales. This makes the model fit naturally into detection and segmentation systems that expect a feature pyramid.

For a fixed window size, local attention scales approximately linearly with the number of image tokens:

$$
O(N M^2),
$$

where $M$ is the number of tokens in one window. Since $M$ is fixed, the quadratic interaction is confined to small neighborhoods.

**Main contribution:** local shifted attention plus hierarchical features for scalable dense prediction.

**Trade-off:** window partitioning introduces architectural complexity, and long-range interactions require multiple layers or additional mechanisms.

## 4. Pyramid and hybrid Transformers: adapt the token hierarchy

Other families explored different ways to make Transformers useful at multiple resolutions. Pyramid Vision Transformer (PVT) reduced the token count between stages and produced pyramid features for dense tasks. Hybrid models combined convolutional stems or local operators with attention, using convolution where its inductive bias was still valuable.

These designs reflect a practical compromise. Classification can tolerate a single sequence of patch tokens, but detection, segmentation, optical flow, and video analysis usually need spatial detail at several scales. A strong dense-prediction backbone therefore has to preserve location while still gaining attention's flexible context modeling.

## 5. Modern scaling: larger, better-trained Transformer encoders

Later models improved ViTs through better normalization, positional representations, augmentations, optimizers, and pretraining datasets. Some of the most influential changes came from the training objective rather than the block itself:

- [MAE]({{ site.baseurl }}/Research/2021/MAE%20Masked%20Autoencoders%20Are%20Scalable%20Vision%20Learners.md) masks a large portion of image patches and trains a decoder to reconstruct them.
- [DINOv2]({{ site.baseurl }}/Research/2023/DINOv2%20Learning%20Robust%20Visual%20Features%20without%20Supervision%20(2023).html) uses self-supervision to produce broadly transferable visual features.
- [CLIP]({{ site.baseurl }}/Research/2021/CLIP%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.html) aligns image and text representations, giving the image encoder a language-connected semantic space.

These are not all new Transformer architectures. They are examples of different ways to train an architecture. A ViT can be supervised, masked-image pretrained, self-distilled, or aligned with language. Keeping those axes separate prevents a common category mistake: treating "ViT" as if it names both the network structure and the representation's training objective.

## What self-attention contributes

For a sequence of token embeddings, attention computes queries, keys, and values:

$$
\operatorname{Attention}(Q,K,V) =
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
$$

The attention weights determine which tokens contribute to each output token. Unlike a fixed convolution kernel, the interaction pattern can depend on the image content.

This provides several useful properties:

- distant regions can interact without a long chain of local convolutions;
- the same layer can use different interaction patterns for different images;
- representations can preserve relationships between parts and the whole;
- the architecture can share tools with language and multimodal models.

Attention is not automatically a better feature extractor. It is a flexible mechanism whose benefit depends on data, scale, positional information, and the downstream task.

## Why positional information matters

A self-attention layer is permutation-equivariant: if the token order is shuffled and the positional information is not changed accordingly, the model has no intrinsic reason to know that one patch was above another.

Vision Transformers therefore inject position through learned absolute embeddings, relative position biases, rotary representations, or other coordinate-aware mechanisms. This choice affects:

- transfer to a different image resolution;
- extrapolation to longer token sequences;
- sensitivity to translation and scale;
- the model's ability to reason about local geometry.

Position is not a small implementation detail. It is part of how a Transformer acquires a usable visual coordinate system.

## Architecture and pretraining are separate axes

The following distinctions are useful when comparing encoders:

| Dimension | Examples | Main question |
|---|---|---|
| Architecture | ViT, Swin, PVT, hybrid Transformer | How are visual features computed? |
| Objective | Supervised classification, MAE, DINO, CLIP | What signal shapes the representation? |
| Scale | Tiny, base, large, huge | How much capacity and compute are available? |
| Role | Classifier, backbone, image tower | Where is the representation used? |

For example, DINOv2 can use a ViT architecture, and a CLIP model can use either a ViT or a CNN image tower. The name of the architecture does not tell us whether the model has zero-shot language behavior or dense-task transfer.

## What transfers to downstream tasks?

Transformer encoders can be adapted in several ways:

- **Fine-tuning:** update the full encoder and task head.
- **Linear probing:** freeze the encoder and train a small classifier.
- **Parameter-efficient adaptation:** update adapters, low-rank parameters, or selected layers.
- **Dense prediction:** use intermediate features with a decoder, feature pyramid, or mask head.
- **Multimodal use:** connect an image tower to a text encoder or language model.

Early patch embeddings and shallow layers retain local appearance. Middle layers often organize parts and regions. Later layers become more semantic, especially when the pretraining objective rewards category or language alignment.

The best layer is therefore task-dependent. A representation that is excellent for image-level retrieval may not be the best input for boundary-accurate segmentation.

## Limitations

### Token cost

High-resolution inputs create long sequences. This is particularly painful for video, where spatial tokens are multiplied by time. Windowed, sparse, pooled, or approximate attention can reduce the cost, but each changes the interaction pattern.

### Data and training sensitivity

The original ViT has weaker built-in visual assumptions than a CNN. With limited data, it may need stronger augmentation, distillation, pretraining, or architectural locality to generalize well.

### Spatial detail

Patchification is a form of spatial compression. Small objects and thin boundaries can disappear before later layers see them. Smaller patches preserve detail but increase memory and latency.

### Deployment complexity

Theoretical FLOPs do not fully predict runtime. Attention kernels, tensor shapes, memory bandwidth, window operations, and accelerator support all matter. A smaller CNN can still beat a Transformer in real-time deployment.

### Interpretability caution

Attention maps are useful diagnostics, but they are not automatically explanations. A high attention weight does not prove that a token caused the prediction or that the highlighted region is the model's complete evidence.

## When Vision Transformers make sense

They are especially attractive when:

- large-scale pretraining or a strong public checkpoint is available;
- global context matters more than a strictly local computation pattern;
- the same encoder will support several tasks;
- image and language representations need to share a common design;
- the deployment target has efficient Transformer kernels;
- the team needs a foundation-model backbone rather than a narrow classifier.

A CNN remains a sensible choice when data is scarce, latency is tightly constrained, or the task benefits from strong local and translation-aware inductive biases.

## The historical lesson

The progression from ViT to DeiT and Swin is not simply a replacement of convolution with attention:

- **ViT:** patches and global attention can scale visual recognition.
- **DeiT:** training strategy and distillation can make that idea data-efficient.
- **Swin:** locality and hierarchy are necessary for efficient dense prediction.
- **MAE and DINO-style models:** the pretraining objective can matter as much as the block design.
- **CLIP:** an image encoder can become part of a shared semantic space with language.

The field did not discard the lessons of CNNs. It reintroduced locality, pyramids, multiscale features, and efficient operators wherever they solved a real problem. The most useful modern encoders are often hybrids in spirit, even when their central block is attention.

## Takeaway

Vision Transformers changed the unit of visual computation from a fixed local filter to a learned interaction among image tokens. That shift made global context, scaling, multimodal alignment, and flexible pretraining more natural.

Their success depends on more than the Transformer block. Patch size, positional encoding, hierarchy, data scale, objective, and hardware all shape the result. ViT is best understood not as a single model that defeated CNNs, but as a design language for building visual encoders with different assumptions about data, context, and transfer.

## Recommended reading

- [Attention, Transformers, and GPT](https://medium.com/@trevormcguire/attention-transformers-and-gpt-b3adbbb4a950) - a readable bridge from general attention to Transformer architectures.
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - the original Vision Transformer paper.

## Recommended video

- [Stanford CS231N Lecture 8: Attention and Transformers](https://www.youtube.com/watch?v=RQowiOF_FvQ) - a university lecture connecting attention mechanics to modern vision models.

When those assumptions match the task and the available compute, a Transformer can serve as a classifier, a dense-prediction backbone, or the visual foundation for a larger multimodal system.

## Continue the vision-encoder series

- [How Many Vision Encoders Are There?]({{ site.baseurl }}/2026/09/19/how-many-vision-encoders.html)
- [Classical Supervised Vision Backbones]({{ site.baseurl }}/2026/09/19/classical-supervised-vision-backbones.html)
- [Self-Supervised Foundation Encoders]({{ site.baseurl }}/2026/09/19/self-supervised-foundation-encoders.html)
- [Generalist Multimodal Encoders]({{ site.baseurl }}/2026/09/19/generalist-multimodal-encoders.html)
- [Task-Specific Vision Specialists]({{ site.baseurl }}/2026/09/19/task-specific-vision-specialists.html)