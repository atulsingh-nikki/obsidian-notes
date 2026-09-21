---
layout: post
title: "Spatial Attention: Weight the Important Locations"
description: "How spatial attention focuses image features on useful regions, and how it differs from general self-attention."
tags: [deep-learning, attention, spatial-attention, computer-vision]
---

Spatial attention assigns different weights to locations in an image or feature map. It may be implemented with self-attention, a convolutional gate, or a task-specific saliency module.

Let $X\in\mathbb{R}^{H\times W\times C}$. A spatial gate can produce one weight $g_{h,w}$ per location and apply it across channels:

$$
	ilde{X}_{h,w,:}=g_{h,w}X_{h,w,:}.
$$

Full spatial self-attention is richer because a location can retrieve features from other locations. A lightweight gate is cheaper and often sufficient when the goal is emphasis rather than arbitrary location-to-location communication.

### A concrete example

For portrait matting, a spatial module can emphasize hair boundaries and the person region while reducing a flat background. In document understanding, it can prioritize text lines or table cells instead of treating the whole page uniformly.

## Where it is used

- Image recognition and visual saliency.
- Segmentation, matting, and object localization.
- Document understanding and text-region detection.
- Image restoration and enhancement.

## What problem did it solve?

Global pooling and uniform feature processing can dilute small but important regions. Spatial attention gives the model a way to emphasize foreground, boundaries, text, or other informative locations.

## Benefits

- Preserves task-relevant spatial focus.
- Can improve small-object and boundary sensitivity.
- Makes visual feature weighting explicit.

## Demerits

The model may learn shortcuts and focus on a correlated but irrelevant region. Spatial weighting alone does not guarantee global object understanding or accurate segmentation boundaries.

## What it made possible

Spatial attention helped visual models allocate capacity to relevant regions instead of treating every pixel or patch equally, especially in dense and weakly supervised tasks.

## Continue

Spatial attention chooses locations. [Channel attention]({{ site.baseurl }}/2026/09/21/channel-attention-deep-dive.html) chooses which feature types to emphasize at those locations.