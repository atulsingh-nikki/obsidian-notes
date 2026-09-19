---
layout: post
title: "What Have We Learned by Probing Supervised Vision Backbones?"
description: "A practical survey of the experiments used to understand classical supervised CNN representations: linear probes, layer-wise analysis, transfer learning, visualization, and controlled evaluation."
tags: [computer-vision, vision-encoders, cnn, probing, representation-learning, interpretability]
---

A classical supervised backbone is usually introduced through its headline number: ImageNet top-1 accuracy, parameter count, FLOPs, or inference speed. But those numbers do not tell us what the network has learned internally.

To understand a backbone, researchers have repeatedly asked a deeper question:

> What information is available in its features, where is that information located, and how much of it transfers to another task?

This is the history of probing supervised vision backbones.

The important caveat is that “probing” is not one experiment. It is a family of tests, each answering a different question. A linear probe asks whether information is easily readable. Transfer learning asks whether the representation is useful for a new task. Visualization asks what patterns activate the features. Controlled probes ask whether the result comes from the representation or from the probe itself.

## What is being probed?

Let a supervised backbone transform an image $x$ into features at several depths:

$$
h_1 = f_1(x), \quad h_2 = f_2(h_1), \quad \ldots, \quad h_L = f_L(h_{L-1}).
$$

A probe takes one of these representations, freezes the backbone, and trains a small model to predict a new target $y$:

$$
\hat{y} = g(h_l).
$$

The choice of layer $l$, target $y$, and probe $g$ determines what the experiment means. A probe on an early convolutional map is asking a different question from a probe on the final pooled feature vector.

For supervised CNNs, the recurring targets have been:

- object identity and category;
- texture, color, and material;
- edges, contours, and local geometry;
- object parts and spatial layout;
- depth, surface normals, and scene geometry;
- segmentation, detection, pose, and other downstream tasks.

## 1. Linear probes: is the information accessible?

The most common probe is a linear classifier trained on frozen features. Given a representation $h \in \mathbb{R}^d$, the probe predicts:

$$
\hat{y} = \operatorname{softmax}(Wh + b).
$$

Because the probe is deliberately weak, a strong result means that the target is **linearly accessible**. The backbone has already organized the information into a form that a simple decision boundary can use.

This is useful for comparing representations. For example, a researcher can compare a ResNet feature, an EfficientNet feature, and a self-supervised feature using the same frozen-data protocol and the same linear head.

But a linear probe does not prove that the backbone uses the information during its original classification task. It also does not prove that the information is disentangled, causally important, or robust to distribution shift.

The dedicated guide [What Is a Linear Probe?]({{ site.baseurl }}{% post_url 2026-09-17-what-is-a-linear-probe %}) develops this distinction in detail.

## 2. Layer-wise probing: the representation changes with depth

A backbone is not one representation. It is a sequence of representations.

Early layers usually preserve local detail. They respond to edges, orientations, colors, and textures. Middle layers combine those signals into motifs, parts, and repeated structures. Late layers become increasingly tied to the categories and invariances needed by the supervised objective.

Layer-wise probing makes this progression measurable. The same target can be decoded from every stage of the network, producing a curve rather than a single score.

Typical patterns include:

- low-level properties are easiest to decode from early and middle layers;
- object category becomes more accessible deeper in the network;
- fine spatial detail may decline after repeated downsampling;
- semantic invariance increases with depth, while exact local information may be discarded;
- intermediate features can be better than final features for dense prediction.

This is why detection and segmentation systems often take features from several backbone stages rather than using only the final classification representation.

A final classification vector can be excellent for “what is in the image?” while being a poor representation for “where is each object boundary?”

## 3. Transfer learning: can the representation do useful work?

Linear probing measures accessibility under a restricted head. Transfer learning measures usefulness under a more realistic downstream setup.

The classic experiment is:

1. initialize from supervised ImageNet weights;
2. attach a new task head;
3. freeze the backbone or fine-tune selected layers;
4. evaluate on detection, segmentation, recognition, depth, or another task;
5. compare against random initialization or another pretrained backbone.

The history of supervised backbones is full of this kind of evidence. ResNet became important not only because it won ImageNet, but because its features transferred well to detection and segmentation. EfficientNet explicitly reported transfer results beyond ImageNet. ResNeXt showed that its cardinality-based features could serve as detection backbones.

Transfer experiments probe a broader property than linear readout:

> Does the representation provide a useful starting point for learning a new task with limited additional data and compute?

The cost is that transfer results mix together several factors: the backbone, the new head, optimization settings, augmentation, fine-tuning schedule, and dataset compatibility. A transfer score is therefore evidence of practical utility, not a pure measurement of representation content.

## 4. Task transfer graphs: probing relationships between tasks

[Taskonomy]({{ site.baseurl }}/Research/2018/Taskonomy%20Disentangling%20Task%20Transfer%20Learning%20(2018).html) expanded transfer probing from isolated comparisons into a map of relationships between visual tasks.

Instead of asking only whether ImageNet features transfer to one target, Taskonomy trained task-specific networks and measured how well representations from one task could support another. The study covered tasks such as depth, surface normals, edges, segmentation, and keypoints.

The result was a directed task-transfer graph. It revealed that some tasks are useful sources of supervision for others, and that geometric tasks can support semantic tasks more effectively than the reverse in certain settings.

This is a powerful way to probe a supervised backbone because it asks not only whether information exists, but whether the representation has the right structure for another visual problem.

## 5. Probing spatial information

Classification pretraining creates a tension. A classifier should become insensitive to changes that do not affect the label, but a dense vision task needs exact spatial information.

Researchers probe this tension through tasks such as:

- predicting object position from a feature;
- reconstructing spatial layouts;
- estimating depth or surface normals;
- detecting edges and contours;
- predicting segmentation masks;
- matching features across image locations or views.

Early convolutional features tend to retain high spatial resolution but have limited semantic context. Late features have broad context and strong semantics but may be spatially coarse or invariant to details that dense tasks need.

Feature pyramids, skip connections, and multi-scale decoders are engineering responses to this probing result. They do not assume that one layer contains everything. They combine early detail with late semantic context.

## 6. Visualization: what patterns activate the backbone?

Another line of work probes the features by visualizing them rather than training a new classifier.

Common methods include:

- optimizing an input image to maximize a unit or channel activation;
- inspecting activation maps over natural images;
- projecting feature embeddings into two dimensions;
- using class-activation maps to identify influential regions;
- comparing nearest neighbors in feature space.

These methods often show the transition from local to semantic processing. Early filters resemble oriented edges, color contrasts, or texture detectors. Deeper units respond to object parts and category-specific patterns.

Visualization is useful for forming hypotheses, but it is not a complete measurement. An optimized image may contain artifacts, and a two-dimensional embedding can create apparent clusters that are not stable in the original feature space.

The safest use of visualization is as a companion to quantitative probes, not as a replacement for them.

## 7. Control tasks: is the probe doing the work?

A probe can be too powerful for the claim being made. Even a modest classifier can memorize patterns when the representation is high-dimensional and the dataset is small.

[Designing and Interpreting Probes with Control Tasks]({{ site.baseurl }}/Research/2019/Designing%20and%20Interpreting%20Probes%20with%20Control%20Tasks%20(2019).html) formalized a useful sanity check. Alongside the real task, train the same probe on a matched control task with randomized labels.

Define selectivity as:

$$
\text{selectivity} = \text{real-task accuracy} - \text{control-task accuracy}.
$$

If the probe performs almost as well on random labels as on the real labels, its raw accuracy is not strong evidence that the representation contains meaningful structure.

For vision backbones, analogous controls can include:

- shuffled labels with the same class distribution;
- spatially permuted targets;
- background-only or object-only versions of the same images;
- targets matched for frequency and sample count but unrelated to the intended property.

This changes the interpretation of probing from “train a classifier and report accuracy” to “design a controlled measurement.”

## 8. What the evidence says about supervised backbones

Across these probing methods, a consistent picture has emerged.

### Supervised backbones learn useful hierarchies

Their features are not random collections of filters. Information becomes progressively more contextual and semantic with depth.

### Classification is a strong but selective teacher

ImageNet-style supervision creates features that transfer broadly, but it favors category recognition. It does not guarantee preservation of every spatial, geometric, or domain-specific property.

### The middle of the network is often a useful compromise

Intermediate features can contain both local detail and semantic context. This helps explain why multi-stage backbones are so effective for detection and segmentation.

### Architecture changes what can be transferred

Inception, ResNet, ResNeXt, DenseNet, and EfficientNet do not only differ in benchmark accuracy. Their connectivity, scale, feature reuse, and capacity allocation affect the geometry and usability of their representations.

### No single probe is decisive

Linear accuracy, transfer performance, visualizations, and task-transfer graphs each provide partial evidence. A credible conclusion usually requires several of them.

## A practical probing protocol

For a new supervised backbone, a compact but serious evaluation could be:

1. Extract frozen features from early, middle, and late stages.
2. Run linear probes for category, color, texture, position, and geometry.
3. Compare against shuffled-label control tasks.
4. Run frozen-backbone transfer to one classification and one dense task.
5. Repeat with partial and full fine-tuning.
6. Inspect nearest neighbors and activation maps.
7. Report compute, resolution, dataset size, and optimization details.
8. Test on a shifted domain if deployment robustness matters.

The important output is not one leaderboard number. It is a profile showing where the backbone is semantic, where it is spatial, how much adaptation it needs, and how much of the result can be attributed to the probe.

## What probing still cannot tell us

Even a careful probing suite leaves open questions:

- Does the backbone actually use the decoded feature for its prediction?
- Is the information stored in a clean factorized subspace or distributed across many directions?
- Does the representation support novel combinations of factors?
- Is the feature causal, or merely correlated with a shortcut in the dataset?
- Will the result survive a change in camera, domain, resolution, or label policy?

Those questions require interventions, counterfactual data, compositional tests, and out-of-distribution evaluation. Probing is the first rung of the ladder, not the final explanation.

## Takeaway

The understanding of supervised backbones has been built through a sequence of increasingly careful probes:

- **Linear probes** test what is easily readable.
- **Layer-wise probes** reveal the progression from local to semantic features.
- **Transfer experiments** test practical reuse.
- **Task-transfer graphs** test relationships between visual objectives.
- **Visualization** suggests what units and regions respond to.
- **Control tasks** test whether the measurement is being inflated by the probe.

Together, these experiments explain why classical supervised backbones remain useful. They learn structured, hierarchical representations that can be repurposed well, even though their knowledge is shaped by a narrow supervised objective.

The right conclusion is therefore not that a backbone “contains everything” or “understands the image.” It is more precise:

> A supervised backbone makes some visual properties accessible at particular layers and scales, and probing is how we discover which ones.
