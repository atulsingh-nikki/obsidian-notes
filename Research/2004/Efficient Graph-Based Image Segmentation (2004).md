---
title: "Efficient Graph-Based Image Segmentation (2004)"
aliases:
  - FH Segmentation
  - Graph-Based Segmentation
authors:
  - Pedro F. Felzenszwalb
  - Daniel P. Huttenlocher
year: 2004
venue: "International Journal of Computer Vision (IJCV)"
doi: "10.1023/B:VISI.0000022288.19776.77"
citations: 20,000+
tags:
  - paper
  - image-segmentation
  - graph-cuts
  - clustering
  - computer-vision
fields:
  - vision
  - segmentation
  - graph-theory
related:
  - "[[Normalized Cuts and Image Segmentation (Shi & Malik, 2000)]]"
  - "[[Mean Shift Segmentation (Comaniciu & Meer, 2002)]]"
predecessors:
  - "[[Normalized Cuts and Image Segmentation (Shi & Malik, 2000)]]"
successors:
  - modern region-proposal methods
  - superpixel algorithms (e.g. SLIC, SEEDS)
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
Felzenszwalb & Huttenlocher (2004) proposed a **computationally efficient graph-based segmentation algorithm**. Unlike normalized cuts, it does not require solving eigenvalue problems. Instead, it uses a **greedy region-merging criterion** that compares internal variation of a region against edge weights between regions.

# Key Idea
> Build a graph over pixels with edges weighted by similarity (color, intensity, position).  
> Iteratively merge regions if the inter-region difference is smaller than their internal variation, using a dynamic threshold that adapts to region size.

# Method
- Pixels = nodes; edges weighted by color/intensity differences.  
- Sort edges by weight.  
- Start with each pixel as its own component.  
- For each edge (u,v):  
  - If `weight(u,v) ≤ min(Int(C1)+τ(C1), Int(C2)+τ(C2))`, merge components C1 and C2.  
- `Int(C)` = internal variation of region C (max edge weight in its MST).  
- `τ(C)` = adaptive threshold depending on region size.  
- Output = set of regions (segments).  

# Results
- Very fast: **O(n log n)** complexity (near-linear).  
- Produced visually meaningful segmentations (balanced detail and coherence).  
- Widely adopted in computer vision for preprocessing, region proposals, and superpixels.  

# Why it Mattered
- Provided a **practical, scalable alternative** to spectral methods like Normalized Cuts.  
- Still one of the most used classical segmentation algorithms.  
- Foundation for many pipelines (object proposals, image parsing, stereo).  

# Architectural Pattern
- Graph-based clustering via greedy merging.  
- Adaptive threshold ensures segmentation at multiple scales.  

# Connections
- Alternative to spectral clustering (Normalized Cuts).  
- Predecessor to **superpixel methods** like SLIC (2010).  
- Still integrated into libraries (OpenCV, scikit-image).  

# Implementation Notes
- Key parameter: scale constant `k` controlling granularity.  
- Sensitive to noise → often applied after smoothing (Gaussian blur).  
- Efficient even on large images.  

# Critiques / Limitations
- Greedy algorithm can miss global optima.  
- Sensitive to parameter choice.  
- No semantic information, purely low-level grouping.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Graph representation of image (nodes = pixels).  
- Greedy region merging vs global eigenvalue optimization.  
- Adaptive thresholding for segmentation.  

## Postgraduate-Level Concepts
- Comparison with spectral clustering (global vs local criteria).  
- Complexity analysis: O(n log n).  
- Relationship to minimum spanning trees.  
- Multi-scale segmentation via parameter `k`.  

---

# My Notes
- FH segmentation = **practical workhorse** still in use.  
- Unlike Ncuts, it scales to megapixel images easily.  
- Often used to generate proposals or initialize more complex models.  
- A key example of “simple but effective” in vision.  

---
