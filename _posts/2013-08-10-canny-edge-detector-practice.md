---
layout: post
title: "Canny Edge Detector in Practice (circa 2013)"
description: "Parameter recipes, tuning notes, and performance tips for deploying the Canny edge detector on 2013-era hardware."
tags: [computer-vision, image-processing]
---

Canny’s blend of Gaussian smoothing, gradient estimation, non-maximum suppression, and hysteresis linking made it the go-to detector for general-purpose pipelines in 2013. Below is a practical cheat-sheet for replicable results on CPUs and modest GPUs from that period.

## Parameter recipes

- **Gaussian sigma (σ)**: 1.0–1.4 for VGA/720p inputs; 1.6–2.0 for 1080p when you want broader edges and better noise suppression.
- **High/low thresholds**: A common ratio was 2:1 or 3:1. Many OpenCV users started around `high = 80–120` and `low = 40–60` for 8-bit images, adjusting upward for sharper textures or downward for smoother scenes.
- **Gradient operator**: Sobel (3×3) was the default for its noise robustness; Scharr (3×3) delivered slightly better rotation fidelity and was favored when texture orientation mattered.

## Practical tuning workflow

1. **Normalize lighting first**: Apply a mild bilateral filter or CLAHE before Canny when illumination is uneven.
2. **Pick sigma, then thresholds**: Choose σ to match the expected edge scale; only then sweep thresholds to balance recall vs. false positives.
3. **Use non-maximum suppression diagnostics**: Visualize gradient angles to verify that edges are being thinned rather than broken.
4. **Guard against speckle**: If salt-and-pepper noise is present, precede Canny with a 3×3 median filter.

## Performance notes (2013 hardware)

- **SIMD mattered**: SSE2/AVX intrinsics in OpenCV 2.x gave 2–4× speedups over naive loops.
- **GPU ports were early-stage**: CUDA implementations existed but incurred PCIe overhead; they paid off mainly for batch processing or >1080p frames.
- **Edge density considerations**: High-texture scenes could flood hysteresis; clamping the high threshold or applying a light Gaussian blur first kept runtime predictable.

## When to use LoG instead

If edge polarity is ambiguous or you need rotational symmetry without gradient direction bookkeeping, Laplacian of Gaussian (LoG) can still be competitive. In 2013-era tests, Canny generally won on localization and thin edges, while LoG produced slightly smoother, more continuous contours on very noisy inputs.

## Example OpenCV 2.x call

```cpp
cv::Mat edges;
cv::GaussianBlur(img, img, cv::Size(0, 0), 1.4);
cv::Canny(img, edges, 50, 120, 3, true); // thresholds, aperture size, use L2 gradient
```

Keep the thresholds proportional when changing σ or aperture size. For production, profile with and without the extra GaussianBlur—OpenCV’s internal smoothing can suffice for many scenes.
