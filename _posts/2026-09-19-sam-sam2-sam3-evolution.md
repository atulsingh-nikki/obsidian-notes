---
layout: post
title: "SAM, SAM 2, and SAM 3: How Promptable Segmentation Evolved"
description: "A deep dive into the evolution from image prompting in SAM to video memory in SAM 2 and open-vocabulary concept segmentation in SAM 3."
tags: [computer-vision, segmentation, foundation-models, video-understanding, multimodal-learning]
---

The Segment Anything family changed the way many people think about visual segmentation. Before SAM, segmentation was usually framed as a task-specific prediction problem: train a model for semantic classes, object instances, medical structures, or a known video dataset.

SAM introduced a different interface. Give the model a point, box, or another prompt, and it returns a mask for the object that the prompt refers to. SAM 2 kept that interaction model but added time: the selected object could be propagated through a video. SAM 3 expanded the meaning of “anything” again: the prompt could describe a concept in text or show an example, and the model could find, segment, and track every matching instance.

The progression is therefore not just SAM 1, then a larger SAM 2, then a larger SAM 3. Each version changes the unit of interaction:

- **SAM:** segment the object indicated by a visual prompt.
- **SAM 2:** segment and track the prompted object across frames.
- **SAM 3:** find all instances matching a concept, then segment and track them.

## The common abstraction: promptable segmentation

Let $x$ be an image or video, $p$ a prompt, and $m$ the predicted mask. A promptable segmentation model learns a function:

$$
m = f_\theta(x, p).
$$

The prompt changes what the model should segment without requiring a new task-specific classifier for every object category. The model is not asked to output one fixed set of semantic classes. It is asked to respond to a request.

That request can be visual or linguistic:

- a positive point inside an object;
- negative points on nearby background or another object;
- a bounding box;
- a previous mask;
- a short text phrase;
- an image exemplar showing the object to find.

This interface is the through-line across the family. The major changes are what counts as a prompt, whether time is part of the input, and whether the model returns one prompted object or all matching instances.

## 1. SAM: make image segmentation interactive

[SAM]({{ site.baseurl }}/Research/2023/Segment%20Anything%20(SAM,%202023).html) introduced a promptable image-segmentation foundation model trained with the SA-1B dataset: roughly 11 million images and more than one billion masks.

Its design separates three responsibilities:

1. An image encoder computes a reusable representation of the image.
2. A prompt encoder converts points, boxes, or masks into prompt features.
3. A lightweight mask decoder combines image and prompt features to predict candidate masks and quality scores.

This separation is important for interaction. The expensive image encoding can be computed once, while new point or box prompts can be tested quickly through the prompt encoder and mask decoder.

### What SAM solved

SAM made several capabilities available in one interface:

- segment an object from a single click;
- refine the result with positive and negative clicks;
- use a box to disambiguate an object;
- produce multiple plausible masks when a prompt is ambiguous;
- transfer to object categories and image domains that were not explicit training classes.

The model's broad behavior came from the combination of architecture, prompt design, and data engine. The dataset was not just a larger collection of ordinary segmentation labels. It was built to support interactive mask generation at scale.

### SAM's limitations

SAM was fundamentally an image model. It could produce a mask for a frame, but it did not natively maintain object identity or memory across time. A video pipeline had to call it repeatedly, propagate prompts externally, or combine it with a video-object-segmentation system.

SAM could also struggle with very thin structures, transparency, ambiguous boundaries, small objects, and domains far from its image distribution. Its generality was valuable, but it did not eliminate the need for specialized segmentation systems.

## 2. SAM 2: add time and memory

[SAM 2]({{ site.baseurl }}/Research/2024/SAM%202%20Segment%20Anything%20in%20Images%20and%20Videos%20(2024).html) extended the promptable interface from images to videos. The central change was not simply processing more frames. It was giving the model a memory mechanism so that information from earlier frames could guide segmentation in later frames.

For a video with frames $x_1, x_2, \ldots, x_T$, SAM 2 maintains a state $h_t$:

$$
h_t = \operatorname{Update}(h_{t-1}, x_t, p_t),
$$

and predicts a mask using the current frame and memory:

$$
m_t = f_\theta(x_t, p_t, h_{t-1}).
$$

The memory can contain visual features and mask information from prior frames. A user may prompt the object once, then correct it later with another point or box. The model uses those interactions to update the tracked segmentation.

### What SAM 2 added

SAM 2 unified several workflows:

- interactive image segmentation;
- interactive video segmentation;
- mask propagation through time;
- object tracking guided by segmentation;
- correction and refinement during a video session.

This was a major conceptual step. A mask was no longer only a static output. It became part of an evolving interaction state.

### Why memory changes the problem

Frame-by-frame segmentation treats each image mostly independently. Memory allows the model to use temporal continuity:

- the object's appearance in earlier frames;
- its prior mask and location;
- changes in scale and viewpoint;
- temporary blur or partial occlusion;
- user corrections from previous moments.

However, memory also creates new failure modes. An incorrect mask can contaminate later predictions. Long occlusions can make identity ambiguous. Scene cuts, fast motion, and similar objects can cause drift or identity switches.

### SAM 2's training shift

SAM 2 introduced the SA-V video data engine and dataset, extending the data-generation philosophy of SA-1B into the temporal domain. The model needed examples of object motion, occlusion, appearance change, and interactive correction, not just isolated masks.

This illustrates a recurring foundation-model lesson: adding a capability requires changing the data engine as well as the network. Video behavior cannot be learned reliably from image masks alone.

## 3. SAM 3: segment concepts, not only prompted objects

[SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) changes the task from “segment the object I indicate” to “find and segment every object that matches this concept.” The concept can be specified by a short noun phrase, an image exemplar, or a combination of text and visual evidence.

Examples include:

- “yellow school bus”;
- “players wearing white”;
- a box around one example of a particular object;
- a text phrase plus an exemplar to narrow the intended concept.

SAM 3 calls this **Promptable Concept Segmentation (PCS)**. Given a concept prompt, it returns masks and unique identities for all matching instances in an image or video.

The difference from SAM 2 is subtle but fundamental. SAM 2 can track an object selected by a point or box. SAM 3 can use a concept to discover multiple matching objects, then segment and track them.

## SAM 3's model design

According to the released paper and implementation, SAM 3 consists of a detector and a tracker that share a vision encoder. The detector is DETR-based and conditioned on text, geometry, and image exemplars. The tracker extends the SAM 2 transformer encoder-decoder and memory-based video design.

This decoupling reflects two different computational problems:

- **Detection and concept matching:** Which objects in the current image match the prompt?
- **Tracking and propagation:** How should the identities and masks of those objects persist through video?

Trying to force both behaviors into one undifferentiated module can create task interference. A detector must search broadly; a tracker must preserve identity efficiently. SAM 3 separates them while allowing them to share visual features.

### The presence token

SAM 3 introduces a presence token or presence head to distinguish recognition from localization. The model needs to answer two related questions:

1. Is an object matching this concept present?
2. If it is present, where is it and what is its mask?

Separating presence from localization helps with closely related prompts and negative prompts. For example, “a player in white” and “a player in red” may share many visual features, but the model must decide whether each concept is present before producing masks.

This is an important evolution from ordinary prompt-conditioned mask prediction. Concept segmentation requires explicit handling of absence as well as presence.

## SAM 3's data engine: from masks to concepts

SAM 1 scaled image masks through SA-1B. SAM 2 extended mask collection to videos through SA-V. SAM 3 requires an additional level of annotation: the relationship between a phrase or exemplar and every matching instance.

The SA-Co data engine is designed around concept segmentation. The released material describes more than four million unique concepts across images and videos, with hard negatives and instance identities. The associated SA-Co benchmarks evaluate whether a model can find all objects matching a noun phrase, including cases where the prompt has no matching object.

This data changes what the model learns:

- **SAM 1:** which pixels belong to the prompted object;
- **SAM 2:** how those pixels and identities evolve over time;
- **SAM 3:** which instances match a concept, where they are, and how they persist.

The negative examples matter. A concept model cannot simply generate plausible masks whenever it sees a prompt. It must also know when the requested concept is absent.

## The evolution in one table

| Version | Primary input | Prompt types | Core output | Main new capability |
|---|---|---|---|---|
| SAM | Image | Points, boxes, masks | One or more candidate masks | Interactive zero-shot image segmentation |
| SAM 2 | Image or video | Visual prompts plus corrections | Masks with temporal identity | Memory-based video segmentation and tracking |
| SAM 3 | Image or video | Text, exemplars, points, boxes, masks | All matching masks and identities | Open-vocabulary concept segmentation |

The evolution is best understood as an expansion along three axes:

1. **Input time:** image to video.
2. **Prompt semantics:** visual indication to visual indication plus language and exemplars.
3. **Output cardinality:** the prompted object to all matching instances.

## What stayed constant?

Despite the new capabilities, the family retains a common philosophy:

- segmentation should be promptable rather than tied to a fixed label set;
- the image encoder should be reusable across interactions;
- users should be able to correct predictions incrementally;
- foundation-scale data should support transfer beyond the training distribution;
- the system should expose masks as useful primitives for larger workflows.

The continuity matters. SAM 3 is not a completely separate replacement for SAM 1 and SAM 2. It retains visual prompts and interactive refinement, adds concept prompts, and carries forward video tracking.

## What changed in the user experience?

### SAM

The user points to an object and receives a mask. The interaction is direct and local.

### SAM 2

The user points to an object in an image or frame and receives a mask that can persist through a video. The interaction becomes temporal.

### SAM 3

The user describes or shows a concept and receives masks for all matching instances. The interaction becomes semantic and exhaustive. Visual clicks remain available for correction and refinement.

This is a move from **selection** to **search**. SAM 1 and SAM 2 primarily answer “what is this prompted object?” SAM 3 can answer “where are all the objects matching this description?”

## What changed technically?

### From mask decoding to detection plus tracking

SAM 1 centers on prompt encoding and mask decoding. SAM 2 adds memory to propagate masks through time. SAM 3 adds a detector for concept-conditioned instance discovery and retains a tracker for temporal propagation.

### From visual prompts to multimodal prompts

SAM 1 and SAM 2 are naturally visual-prompt models. SAM 3 treats text and image exemplars as first-class prompts, while still accepting points, boxes, and masks.

### From one object to a concept set

The output is no longer implicitly one selected object. A concept may match zero, one, or many instances. The model must return masks, scores, and consistent identities for the matching set.

### From generic masks to concept discrimination

SAM 3 must distinguish related descriptions, understand hard negatives, and decide when a concept is absent. That requires more than better boundary prediction.

## What did not disappear?

SAM 3 does not make specialized segmentation obsolete. It remains possible for a domain-specific model to outperform a general foundation model on:

- very fine boundaries;
- medical or scientific structures;
- rare industrial defects;
- calibrated geometry;
- strict latency or memory budgets;
- a fixed set of labels with abundant training data.

Nor does open-vocabulary prompting guarantee perfect language grounding. Short phrases can be ambiguous, culturally dependent, or visually under-specified. A model can correctly identify a concept but still produce a mask that is too coarse for editing or measurement.

## Failure modes across the versions

### SAM: prompt ambiguity

A point may lie near multiple objects. A box may contain several plausible regions. The model can return multiple masks, but the user or downstream system still has to choose the intended interpretation.

### SAM 2: temporal drift

An early error can be propagated. Occlusion, fast motion, scene cuts, and similar objects can cause the tracker to lose identity or attach a mask to the wrong target.

### SAM 3: concept ambiguity and exhaustive recall

A phrase may match objects with different appearances. The model must balance precision and recall across all instances, including small or partially occluded objects. It must also avoid hallucinating a mask when the concept is absent.

The failure mode becomes more global as the interface becomes more powerful. An incorrect click affects one object. An incorrect concept interpretation can affect every matching object in a scene or video.

## Why SAM 3 is more than “SAM with text”

Adding text to a prompt interface would not by itself solve concept segmentation. SAM 3 also needs:

- a detector that can search for all matching instances;
- a training dataset with concept-instance relationships;
- hard negatives and no-match examples;
- a presence decision separate from localization;
- identity management for multiple objects;
- a tracker that can propagate the discovered set through video.

The real evolution is from **prompt-conditioned segmentation** to **prompt-conditioned perception**. The model is asked not only to draw a mask, but to discover, enumerate, identify, and follow the objects satisfying a concept.

## Practical selection guide

| Need | Best starting point |
|---|---|
| Click-to-mask image interaction | SAM |
| Prompt once and track through video | SAM 2 |
| Find every instance matching text | SAM 3 |
| Find objects from a visual example | SAM 3 |
| Fine-tuned fixed-domain segmentation | A task-specific specialist, possibly initialized from SAM |
| Maximum control over boundaries | SAM or SAM 2 plus iterative prompts, or a specialized model |
| Low-latency edge deployment | A smaller distilled or task-specific model |

In a production pipeline, the right answer may be a combination. SAM 3 can discover objects, SAM 2-style tracking can maintain them, and a specialized refinement model can improve boundaries for the final edit or measurement.

## Broader lessons for vision foundation models

The SAM family illustrates several general lessons:

### Interfaces can be as important as architectures

The promptable interface made segmentation useful to people and downstream systems that did not want to retrain a classifier for every object category.

### Data engines define capability

SA-1B enabled broad image masks. SA-V enabled temporal propagation. SA-Co enables concept-conditioned instance discovery. New capabilities require data that expresses the capability.

### Memory changes visual interaction

Video memory turns a static prediction into a stateful session. It creates continuity, but it also creates drift and error accumulation.

### Generality has levels

SAM 1 generalizes over objects and images. SAM 2 generalizes over objects, images, and time. SAM 3 generalizes over concepts and instances in images and videos. “Zero-shot” is meaningful only after specifying the prompt type, domain, and output required.

### Specialists still matter

Foundation models can provide strong initialization, annotation, discovery, and tracking. A specialist may still be the right final model when precision, calibration, or latency dominates.

## Takeaway

SAM began by making image segmentation interactive and promptable. SAM 2 extended that interaction through time with memory and video tracking. SAM 3 made the prompt semantic and exhaustive: describe or show a concept, then find, segment, and track every matching instance.

The family evolved along a clear path:

$$
\text{prompted object} \;\longrightarrow\; \text{tracked object} \;\longrightarrow\; \text{prompted concept set}.
$$

The deepest change is not simply better masks. It is a change in what the model is asked to understand: from the pixels belonging to one indicated object, to the temporal identity of that object, to the set of objects that satisfy a visual or linguistic concept.

That progression makes SAM useful as more than a segmentation model. It becomes an interaction primitive for annotation, editing, search, video understanding, and multimodal computer-vision systems.

## Related posts

- [Task-Specific Vision Specialists]({{ site.baseurl }}/2026/09/19/task-specific-vision-specialists.html)
- [Generalist Multimodal Encoders]({{ site.baseurl }}/2026/09/19/generalist-multimodal-encoders.html)
- [Self-Supervised Foundation Encoders]({{ site.baseurl }}/2026/09/19/self-supervised-foundation-encoders.html)
- [Vision Transformers]({{ site.baseurl }}/2026/09/19/vision-transformers.html)