---
layout: post
title: "SAM 2 as a Vision Encoder: Reusing Promptable Features Beyond Segmentation"
description: "SAM 2 is a promptable video segmenter, but its image encoder and memory are a reusable vision backbone. What capabilities that reuse unlocks — dense prediction, tracking, domain transfer, distillation — and where SAM features fall short."
tags: [computer-vision, vision-encoders, segmentation, foundation-models, video-understanding, transfer-learning]
---

Most people meet SAM 2 through its interface: click a point, get a mask, and have that mask follow an object through a video. That framing — [how promptable segmentation evolved]({{ site.baseurl }}/2026/09/19/sam-sam2-sam3-evolution.html) — is the product story.

There is a second story that matters more if you are building systems. SAM and SAM 2 are trained on enormous amounts of segmentation supervision, and the largest, most expensive part of each model is an **image encoder** that turns pixels into dense feature maps. Once you have that encoder, you can ignore the promptable head entirely and treat SAM 2 as a general **vision encoder** — a backbone whose features you route into your own decoders.

This post is about that reuse: which of SAM 2's capabilities get *enhanced or extended* when you use SAM (or SAM 2) as a vision encoder, what its features are actually good at, and where they quietly fail.

## The part of SAM 2 that is a vision encoder

It helps to separate SAM 2 into the piece that is expensive to train from the pieces that are cheap.

- **Image encoder** — a [Hiera](https://arxiv.org/abs/2306.00989) hierarchical Vision Transformer, MAE-pretrained, that maps a frame to multi-scale feature maps. This is where almost all the parameters and FLOPs live. (SAM 1 used a plain MAE-pretrained ViT-H/L/B instead.)
- **Prompt encoder** — tiny; embeds points, boxes, and masks.
- **Memory attention + memory bank + memory encoder** — SAM 2's addition. It conditions the *current* frame's features on embeddings of past frames and past predictions, which is what lets a mask persist through occlusion and motion.
- **Mask decoder** — lightweight; turns conditioned features plus prompt embeddings into masks.

"Using SAM 2 as a vision encoder" means keeping the image encoder (and optionally the memory stack) and discarding the mask decoder, then attaching your own task head to the feature maps. Because the encoder dominates the compute, you inherit a heavily-trained backbone almost for free.

The reason this is interesting is *what that backbone was trained to represent*. Unlike a classification backbone ([classical supervised backbones]({{ site.baseurl }}/2026/09/19/classical-supervised-vision-backbones.html)) or a contrastive/self-distillation encoder ([self-supervised foundation encoders]({{ site.baseurl }}/2026/09/19/self-supervised-foundation-encoders.html)), SAM's encoder was optimized so that a *lightweight* head can cut out any object given a spatial prompt. That objective pushes an unusual bias into the features.

## What SAM features are unusually good at

The training objective — "produce a clean mask for whatever is prompted, for almost any object" — shapes the representation in a specific direction:

- **Boundary and localization precision.** The features carry sharp, spatially-aligned information about where objects begin and end. A trivial decoder recovers crisp masks, which means the encoder is doing the spatial heavy lifting.
- **Class-agnostic objectness.** SAM was trained on ~1.1B masks spanning objects, parts, and stuff, with no class labels. The features encode "this is a coherent thing" without committing to *what* it is. That generality transfers to categories and domains never seen at training time.
- **Dense, high-resolution feature maps.** SAM runs its encoder at high input resolution and keeps spatial detail, which is exactly what dense-prediction heads want.
- **Multi-scale structure (SAM 2 / Hiera).** Hiera emits a feature pyramid, so it drops into FPN-style detection and segmentation heads without bolting on an extra neck.

In short, SAM's encoder is a *spatial* specialist. That is precisely the axis where classification- and language-aligned encoders are weakest.

## What SAM features are not good at

The same objective that makes SAM spatially sharp leaves a gap: **global semantics**.

Because a prompt tells the model *which* region to segment, the encoder never has to decide *what a region is* on its own. So when you probe SAM features for image classification or open-vocabulary recognition, they underperform encoders like DINOv2 (self-distillation) or CLIP (language-aligned), which were explicitly trained to make whole-image or region semantics linearly readable. (If "linearly readable" is unfamiliar, see [what a linear probe measures]({{ site.baseurl }}/2026/09/17/what-is-a-linear-probe.html).)

This is not a bug; it is the direct consequence of the objective. It is also why so many recent systems pair encoders rather than pick one: SAM for **where**, CLIP or DINOv2 for **what**. That "route different questions to different backbones" pattern is the subject of [how many vision encoders you actually need]({{ site.baseurl }}/2026/09/19/how-many-vision-encoders.html).

| Property | SAM / SAM 2 encoder | DINOv2 | CLIP |
|---|---|---|---|
| Boundary / localization | Very strong | Moderate | Weak |
| Class-agnostic objectness | Very strong | Moderate | Weak |
| Global semantic classification | Weak | Strong | Strong |
| Open-vocabulary / text alignment | None (visual only) | None | Strong |
| Native multi-scale features | Yes (Hiera) | No (plain ViT) | No |
| Temporal conditioning | Yes (SAM 2 memory) | No | No |

## Capabilities that reuse unlocks

With that profile in mind, here is what actually gets *enhanced* when SAM 2's encoder becomes your backbone.

### 1. Dense prediction that transfers to new domains

Because the features are class-agnostic and boundary-sharp, a small head trained on top transfers well to segmentation tasks in domains SAM never saw. The clearest example is medical imaging: freezing or lightly fine-tuning the SAM encoder and retraining the decoder (the MedSAM line of work) adapts promptable segmentation to CT, MRI, and pathology far more cheaply than training a segmenter from scratch. The encoder already knows "coherent region with a boundary"; you are only teaching it which regions matter.

The same logic applies to depth, normals, edges, and matting — dense heads that benefit from spatially precise features and do not need the encoder to name anything.

### 2. Video segmentation and tracking, via memory-conditioned features

This is the capability most directly *enhanced by SAM 2 specifically*. SAM 2's memory attention produces features for the current frame that are already conditioned on the object's history. If you tap those conditioned features (not just the raw per-frame encoder output), you get a representation that is **temporally stable** — useful for video object segmentation, tracking-by-segmentation, and label propagation across frames. SAM 2 turned a per-image encoder into a *streaming* one, and downstream video tasks inherit that stability. This connects to the broader tracking story in [task-specific vision specialists]({{ site.baseurl }}/2026/09/19/task-specific-vision-specialists.html).

### 3. Open-vocabulary pipelines, with SAM as the mask engine

SAM's encoder has no text alignment, but it composes cleanly with a model that does. The Grounded-SAM pattern is: a text-grounded detector (Grounding DINO) or CLIP proposes *where* a named concept is, and SAM turns that box or point into a precise mask. Here SAM is the promptable **mask engine** downstream of a semantic model — its spatial features do the part they are best at, and the semantic model supplies the "what." (SAM 3 later folds concept prompting inside; the [evolution post]({{ site.baseurl }}/2026/09/19/sam-sam2-sam3-evolution.html) covers that.)

### 4. Cheap distillation into deployable encoders

Because the encoder is the costly component, a lot of engineering value comes from **distilling it**. MobileSAM and EfficientSAM replace SAM's ViT-H encoder with a much smaller student trained to reproduce its embeddings, keeping the same lightweight decoder. You keep most of the promptable quality at a fraction of the latency — a direct win for on-device and interactive use. The reusable-encoder framing is exactly what makes this possible: the decoder barely changes, so distillation only has to match one interface.

### 5. Interactive feature caching

A practical systems capability: SAM's encoder runs *once* per image, and then many prompts reuse the cached feature map through the cheap decoder. Treating the encoder as a standalone module makes this explicit — encode once, serve many interactions (or many downstream heads) against the same features. For an interactive annotation tool or a multi-task pipeline, that amortization is the whole ballgame.

## A practical recipe

If you are going to use SAM 2 as an encoder, a few things matter in practice:

- **Freeze first, fine-tune selectively.** Start with a frozen encoder and train only your head; it is a strong baseline and tells you how much the pretrained features already carry. Unfreeze (or add adapters / LoRA) only if the domain gap is large, e.g. medical or overhead imagery.
- **Use the pyramid (SAM 2).** With Hiera you get multi-scale features — feed them to an FPN-style head rather than upsampling a single scale.
- **Mind the resolution and positional embeddings.** SAM encoders expect high input resolution; changing it means interpolating positional embeddings, which can degrade features if done carelessly. See [positional embeddings in ViTs]({{ site.baseurl }}/2026/09/19/positional-embeddings-in-vision-transformers.html).
- **Probe before you commit.** Run a linear probe for your target signal (semantic class, depth, boundary) on the frozen features. If semantics probe poorly — likely — add a semantic encoder rather than fighting SAM to produce something it was never trained for.
- **Pair, don't force.** The strongest systems use SAM for spatial precision and a DINOv2/CLIP-style encoder for semantics, fusing the two. Trying to make one backbone do both usually underperforms the pair.

## When to reach for the SAM 2 encoder

Use it when your task is **spatially dominated**: promptable or interactive segmentation, class-agnostic mask proposals, dense prediction, video object segmentation, or any pipeline where "precise where" is the hard part and "what" is supplied elsewhere.

Reach for a different backbone when your task is **semantically dominated**: classification, retrieval, captioning, or open-vocabulary recognition that must stand alone. There, DINOv2 or CLIP features start ahead, and SAM's spatial sharpness is not the bottleneck.

## Takeaway

SAM 2's headline capability is promptable video segmentation. But the reusable asset underneath is an image encoder trained, at enormous scale, to make objects spatially separable — plus, in SAM 2, a memory stack that makes those features temporally stable. Treat that encoder as a vision backbone and you extend SAM 2's reach well past its own decoder: domain-transferred dense prediction, stable video features, open-vocabulary pipelines where SAM is the mask engine, and small distilled encoders for deployment.

The one thing not to expect is semantics for free. SAM tells you *where* with unusual precision; it was never asked to tell you *what*. Build around that split and the encoder earns its place; ignore it and you will spend a lot of compute rediscovering the split the hard way.

## Related posts

- [SAM, SAM 2, and SAM 3: How Promptable Segmentation Evolved]({{ site.baseurl }}/2026/09/19/sam-sam2-sam3-evolution.html)
- [How Many Vision Encoders Do You Actually Need?]({{ site.baseurl }}/2026/09/19/how-many-vision-encoders.html)
- [Self-Supervised Foundation Encoders]({{ site.baseurl }}/2026/09/19/self-supervised-foundation-encoders.html)
- [Task-Specific Vision Specialists]({{ site.baseurl }}/2026/09/19/task-specific-vision-specialists.html)
