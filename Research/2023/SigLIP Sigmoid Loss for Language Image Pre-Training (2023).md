---
title: "SigLIP: Sigmoid Loss for Language Image Pre-Training (2023)"
aliases:
  - SigLIP
  - Sigmoid Loss for Language Image Pre-Training
authors:
  - Xiaohua Zhai
  - Basil Mustafa
  - Alexander Kolesnikov
  - Lucas Beyer
  - Oliver F. P. G. et al.
year: 2023
venue: "ICCV 2023"
doi: 10.48550/arXiv.2303.15343
arxiv: https://arxiv.org/abs/2303.15343
code: https://github.com/google-research/big_vision
citations: 0
dataset:
  - WebLI
  - ImageNet
tags:
  - paper
  - deep-learning
  - computer-vision
  - multimodal
  - vision-language
  - contrastive-learning
fields:
  - vision
  - nlp
  - representation-learning
related:
  - "[[CLIP Learning Transferable Visual Models From Natural Language Supervision (2021)|CLIP]]"
  - "[[ViT An Image Is Worth 16x16 Words Transformers for Image Recognition at Scale (2020)|Vision Transformer]]"
  - "[[DINOv2 Learning Robust Visual Features without Supervision (2023)|DINOv2]]"
predecessors:
  - "[[CLIP Learning Transferable Visual Models From Natural Language Supervision (2021)|CLIP]]"
successors:
  - "[[SigLIP 2 Multilingual Vision-Language Encoders with Improved Semantic Understanding Localization and Dense Features (2025)|SigLIP2]]"
impact: ⭐⭐⭐⭐☆
status: reading
---

# Summary
**SigLIP** replaces CLIP's batchwise softmax contrastive objective with a pairwise **sigmoid loss** for image-text pre-training. The model still uses separate image and text encoders, but every image-text pair in a batch is treated as an independent binary prediction: should these two items match?

This change removes the need to normalize over all examples in the batch. It makes the objective less dependent on very large effective batch sizes, simplifies distributed training, and gives a strong quality-to-compute trade-off for zero-shot image classification and retrieval.

# Key Idea
> Train each image-text pair as an independent match or non-match with a sigmoid loss instead of forcing every example to compete in a batchwise softmax.

# Method

## Dual encoders

SigLIP follows the CLIP pattern:

- an image encoder $f_\theta(x)$ maps an image to an embedding;
- a text Transformer $g_\phi(t)$ maps a caption or class prompt to an embedding;
- learned projections put both embeddings in the same dimension;
- a scaled dot product produces an image-text compatibility logit.

For an image $x_i$ and text $t_j$:

$$
s_{ij} = \exp(\tau)\frac{f_\theta(x_i)^\mathsf{T}g_\phi(t_j)}
{\lVert f_\theta(x_i)\rVert_2\lVert g_\phi(t_j)\rVert_2} + b.
$$

Here, $\tau$ is a learned logit scale and $b$ is a learned bias. The bias matters because the model is deciding whether a pair is a positive or negative example, rather than only ranking pairs relative to one another.

## Pairwise sigmoid objective

For a batch of $N$ images and $N$ texts, the diagonal pairs are positive and the off-diagonal pairs are usually treated as negatives. Let $y_{ij}=1$ for a matched pair and $y_{ij}=-1$ otherwise. The loss is:

$$
\mathcal{L} = \frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}
\log\left(1+\exp\left(-y_{ij}s_{ij}\right)\right).
$$

This is binary logistic loss applied to the full image-text similarity matrix. Unlike CLIP's cross-entropy loss, it does not require each row and column to sum to a probability distribution or to identify exactly one positive within the batch.

## What changes relative to CLIP?

| Property | CLIP | SigLIP |
|---|---|---|
| Objective | Batchwise softmax cross-entropy | Pairwise sigmoid loss |
| Competition | Examples compete within rows and columns | Each pair is classified independently |
| Batch dependence | Large batches provide more negatives and affect normalization | Batch size is less central to the objective |
| Positive assumption | One designated match per image and text | Positive and negative pair labels can be handled directly |
| Distributed training | Logits often need global batch gathering | No global softmax normalization is required |

SigLIP still benefits from informative negatives. The distinction is that the loss does not mathematically require the batch to be one globally normalized classification problem.

# Results

- SigLIP achieved competitive or better zero-shot transfer than comparable CLIP-style models at similar model sizes and compute budgets.
- The model performed strongly on ImageNet zero-shot classification and a broad set of downstream vision-language benchmarks.
- The sigmoid objective remained effective with smaller batches than the softmax contrastive objective, which is important when memory or communication is constrained.
- The paper demonstrated scaling across image encoders and text encoders rather than relying on one special architecture.

The central result is not a single architectural trick or benchmark record. It is that changing the normalization structure of the loss can improve training efficiency while preserving the useful shared embedding space learned from image-text pairs.

# Why it Mattered

CLIP made natural-language supervision a practical source of transferable visual knowledge. SigLIP showed that the familiar contrastive recipe was not tied to a batchwise softmax. A pairwise classification view can be simpler to distribute and can make better use of the available training setup.

This was especially influential in later open vision-language checkpoints. SigLIP and SigLIP2 encoders became common components in image-text retrieval, visual question answering, document understanding, image classification, and multimodal language models.

# Architectural Pattern

SigLIP is a useful example of a **loss-level innovation** around a mostly familiar architecture:

1. Encode each modality independently.
2. Project both modalities into a shared embedding space.
3. Compute all image-text pair scores in a batch.
4. Assign positive labels to aligned pairs and negative labels to mismatched pairs.
5. Apply logistic loss to the scores.

The system is still a dual encoder, so image and text embeddings can be precomputed and indexed. The changed loss affects training, not the basic deployment interface.

# Connections

- **Predecessor:** [CLIP]({{ site.baseurl }}/Research/2021/CLIP%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision%20(2021).html), which established large-scale image-text contrastive pre-training.
- **Representation backbone:** [ViT]({{ site.baseurl }}/Research/2020/ViT%20An%20Image%20Is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale%20(2020).html) is a common image encoder family for SigLIP checkpoints.
- **Related objective:** standard logistic loss, viewed over all entries of an image-text similarity matrix.
- **Successor:** [SigLIP2](https://arxiv.org/abs/2502.14786) extends the recipe with improved data, multilingual training, knowledge distillation, and additional objectives for stronger semantic, localization, and dense features.

# Implementation Notes

- Use stable binary-cross-entropy-with-logits rather than explicitly computing `log(sigmoid(x))`.
- Compute the similarity matrix in chunks if the batch is large; the matrix costs $O(N^2)$ memory and compute.
- Keep the learned temperature and bias in the checkpoint. They are part of the calibrated matching function, not disposable training metadata.
- For zero-shot classification, encode class-name prompts such as “a photo of a {label}”, normalize the image and text embeddings consistently, and compare image-to-prompt logits.
- Batch construction still matters. Random web captions create many imperfect negatives, and false negatives can teach the model that semantically related items should be dissimilar.
- When fine-tuning, preserve the preprocessing expected by the checkpoint: image resolution, interpolation, normalization, tokenizer, and prompt format can materially change results.

# Critiques / Limitations

- The sigmoid loss does not remove the need for high-quality data. Noisy captions, duplicated images, and demographic or geographic bias remain major sources of error.
- Treating every off-diagonal pair as negative can create false negatives when two captions describe the same visual concept or two images are semantically related.
- A smaller batch is easier to train, but reducing the number or diversity of negatives too far can still weaken the learned embedding space.
- Zero-shot classification depends on prompt wording and label coverage. A strong embedding model is not the same as a calibrated classifier.
- The model aligns image-level semantics more naturally than fine spatial detail. Dense prediction, counting, text reading, and compositional relations generally need additional task-specific components.
- Like CLIP, SigLIP can inherit web-scale data biases and may associate visual concepts with stereotyped language.

# Repro / Resources

- [Paper: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [Official Big Vision code](https://github.com/google-research/big_vision)
- [Hugging Face SigLIP documentation](https://huggingface.co/docs/transformers/model_doc/siglip)
- [WebLI dataset description](https://arxiv.org/abs/2209.06794)

---

# Educational Connections

## Undergraduate-Level Concepts

- **Linear algebra:** image and text are represented as vectors; a dot product measures compatibility.
- **Probability and statistics:** sigmoid converts a logit into a match probability, while binary cross-entropy penalizes incorrect pair labels.
- **Optimization:** the temperature controls the sharpness of similarities, and the bias shifts the match threshold.
- **Data structures:** a batch produces an $N \times N$ similarity matrix whose diagonal contains intended matches.
- **Transfer learning:** class names become text queries, allowing zero-shot classification without a classifier trained on the target labels.

## Postgraduate-Level Concepts

- **Contrastive learning:** compare a normalized batchwise softmax objective with independent pairwise logistic classification.
- **Negative sampling:** off-diagonal pairs supply negatives, but their quality and semantic diversity determine the useful training signal.
- **Distributed systems:** avoiding a global softmax reduces all-gather pressure and decouples the loss from the exact global batch size.
- **Scaling laws:** model size, data quality, image-text diversity, and compute interact; a better loss does not substitute for a better data distribution.
- **Multimodal transfer:** the shared embedding space is useful for retrieval and zero-shot recognition, while generative or reasoning systems usually need a connector and a language model.

---

# My Notes

- SigLIP is a reminder that a model's capabilities can change substantially when the **training geometry** changes, even when the encoder architecture looks familiar.
- The most useful practical distinction is deployment: dual encoders support cheap embedding retrieval, while cross-encoders or generative VLMs provide richer but more expensive interaction.
- Open question: how should pairwise objectives represent graded similarity, multiple valid captions, and hard semantic negatives without collapsing into noisy binary labels?
- Possible extension: compare SigLIP embeddings with DINOv2 features for tasks that require both language alignment and fine spatial correspondence.