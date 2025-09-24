---
title: "SLIC Superpixels: Simple Linear Iterative Clustering (2010)"
aliases:
  - SLIC
  - SLIC Superpixels
authors:
  - Radhakrishna Achanta
  - Appu Shaji
  - Kevin Smith
  - Aurelien Lucchi
  - Pascal Fua
  - Sabine Süsstrunk
year: 2010
venue: "IEEE CVPR Workshops (later TPAMI 2012 extended)"
doi: "10.1109/CVPR.2010.5540158"
citations: 12,000+
tags:
  - paper
  - superpixels
  - image-segmentation
  - clustering
  - computer-vision
fields:
  - vision
  - segmentation
  - representation-learning
related:
  - "[[Mean Shift Segmentation (Comaniciu & Meer, 2002)]]"
  - "[[Efficient Graph-Based Image Segmentation (Felzenszwalb & Huttenlocher, 2004)]]"
  - "[[SEEDS Superpixels (Van den Bergh et al., 2012)]]"
predecessors:
  - mean shift
  - graph-based segmentation (FH)
successors:
  - SEEDS
  - ERS
  - deep superpixel learning
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**SLIC (Simple Linear Iterative Clustering)** is an algorithm for generating **superpixels**—compact, nearly uniform regions that group pixels with similar color and proximity. It adapts k-means clustering to image space by working in a **5D space (Lab color + pixel coordinates)**.  

It became the de facto standard for superpixels due to its simplicity, speed, and memory efficiency.

# Key Idea
> Reformulate k-means in a **5D feature space** (L, a, b, x, y) and constrain clusters to local neighborhoods → efficient, spatially compact superpixels.

# Method
- Represent each pixel by `(L, a, b, x, y)`.  
- Initialize cluster centers on a regular grid.  
- Each cluster only searches within a local 2S×2S region (S = cluster spacing).  
- Distance metric:  
  $$
  D = \sqrt{d_{lab}^2 + \left(\frac{d_{xy}}{S}\right)^2 \cdot m^2}
 $$  
  where `m` = compactness parameter.  
- Iterate assignment & update steps until convergence.  
- Output: superpixels with roughly equal size & compactness.  

# Results
- Produces **regular, uniform, compact superpixels**.  
- Much faster and simpler than prior superpixel algorithms.  
- Widely available (OpenCV, scikit-image).  

# Why it Mattered
- Made superpixels **practical**: efficient, scalable, and easy to implement.  
- Superpixels became a standard preprocessing step for object recognition, tracking, segmentation, and CRFs.  
- Still one of the most widely used segmentation primitives.  

# Architectural Pattern
- k-means clustering in 5D feature space.  
- Constrained local search → efficiency.  

# Connections
- Related to mean shift and FH segmentation.  
- Successor to irregular, costly superpixel methods.  
- Predecessor to SEEDS, ERS, and deep learning superpixels.  

# Implementation Notes
- Key parameter: number of superpixels `K` and compactness `m`.  
- Complexity: O(N) per iteration (linear in pixel count).  
- Works best in Lab color space for perceptual uniformity.  

# Critiques / Limitations
- Fixed grid initialization can bias shape.  
- Sensitive to parameter `m`.  
- Not semantic: purely low-level grouping.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Superpixels = groups of pixels → reduce image complexity.  
- K-means clustering basics.  
- Trade-off: color similarity vs spatial compactness.  

## Postgraduate-Level Concepts
- Distance metrics in high-dimensional space.  
- Efficiency improvements (restricting neighborhood search).  
- Applications: CRFs, graph cuts, region proposals.  
- Extensions: hierarchical or deep-learned superpixels.  

---

# My Notes
- SLIC = **the workhorse of superpixels**.  
- Elegance: one modification of k-means → solved efficiency + compactness.  
- Still widely cited and used, even in the deep learning era for region-level reasoning.  
- Key legacy: made superpixels *mainstream* in CV pipelines.  

---
