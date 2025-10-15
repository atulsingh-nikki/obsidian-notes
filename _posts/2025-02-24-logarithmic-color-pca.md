---
layout: post
title: "Logarithmic Color Spaces, PCA, and the lαβ Intuition"
description: "Explore why Reinhard et al.'s log + PCA recipe for lαβ color space works, and experiment with an interactive sample image."
tags: [color, perception, vision, pca]
---

The lαβ color model from Reinhard, Ashikhmin, Gooch, and Shirley (2001) fuses two simple ideas: compress channel intensities with a logarithm so that multiplicative illumination becomes additive, then rotate the color axes with principal component analysis (PCA) to decorrelate them. As described in their paper “Color Transfer between Images,” the combination mimics how our eyes notice relative brightness and largely independent chromatic contrasts.[^reinhard]

## Why log transforms feel perceptually linear

Light that is twice as bright does not look twice as bright. Human vision roughly follows a Weber–Fechner response: we register ratios better than absolute differences. Taking the logarithm of the long-, medium-, and short-cone signals (LMS) converts multiplicative illumination changes into additive offsets. Equal increments in log space approximate equal perceptual steps, so simple adjustments feel balanced across the range.

## PCA for maximal decorrelation

Raw RGB (or LMS) channels are correlated because natural images often carry shared structure—think of highlights that raise all three channels together. PCA measures the covariance of those log-encoded responses and rotates the axes to align with independent directions of variation: overall lightness, a blue–yellow opponent axis, and a red–green opponent axis. After the rotation the channels respond more independently, which stabilizes edits like white balancing or contrast stretching.

## Building intuition with a sample image

The visualization below synthesizes a small 96×96 sample image in the browser so that the demo stays self-contained. The generated frame blends gradients, saturated patches, and mixed hues. Everything happens in linear light:

1. The original RGB values are converted from sRGB to linear radiance.
2. We compute the covariance and eigenvectors of those linear samples to obtain the PCA rotation.
3. We also map the linear values to LMS, apply a log10 transform, and rotate with the canonical lαβ matrix for comparison.
4. For display, the “log image” panel shows a straightforward per-channel log encoding that compresses dynamic range while keeping colors recognizable.

Interact with the plots: rotate them, zoom in, and notice how the data clouds align with each coordinate system.

<div id="lab-color-playground" class="color-pca-demo">
  <div class="color-pca-demo__images">
    <div>
      <canvas id="color-pca-original" width="96" height="96" aria-label="Sample RGB image"></canvas>
      <p class="color-pca-demo__caption">Original sample (sRGB)</p>
    </div>
    <div>
      <canvas id="color-pca-log" width="96" height="96" aria-label="Log-encoded RGB image"></canvas>
      <p class="color-pca-demo__caption">Per-channel log encoding</p>
    </div>
  </div>
  <div class="color-pca-demo__plots">
    <div id="color-pca-rgb" aria-label="Scatter plot of linear RGB values"></div>
    <div id="color-pca-pca" aria-label="Scatter plot after PCA rotation"></div>
    <div id="color-pca-logplot" aria-label="Scatter plot in log lαβ coordinates"></div>
  </div>
  <div class="color-pca-demo__stats" id="color-pca-stats" aria-live="polite"></div>
</div>

## Takeaways

* Logs turn lighting changes into additive shifts, letting you edit exposure or shadows with simple offsets.
* PCA finds the decorrelated axes that map onto intuitive opponent channels.
* Together they produce the lαβ space that underpins color transfer techniques, where matching the mean and variance of each channel yields perceptually smooth results.

[^reinhard]: Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). [Color Transfer between Images](https://ieeexplore.ieee.org/document/946629). *IEEE Computer Graphics and Applications*, 21(5), 34–41.

<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script defer src="{{ '/assets/js/log-pca-interactive.js' | relative_url }}"></script>
