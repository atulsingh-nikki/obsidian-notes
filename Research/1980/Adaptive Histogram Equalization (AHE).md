---
title: Adaptive Histogram Equalization (AHE)
aliases:
  - AHE
authors:
  - Introduced in 1980s medical imaging research
year: ~1987–1990
venue: Medical Imaging / IEEE Transactions
tags:
  - algorithm
  - image-processing
  - contrast-enhancement
  - medical-imaging
fields:
  - vision
  - local-contrast
related:
  - "[[Histogram Equalization (HE)]]"
  - "[[CLAHE Contrast Limited Adaptive Histogram Equalization|CLAHE]]"
predecessors:
  - Histogram Equalization
successors:
  - CLAHE
impact: ⭐⭐⭐⭐
status: read
---

# Summary
**Adaptive Histogram Equalization (AHE)** enhances **local contrast** by applying histogram equalization in small contextual regions (tiles) rather than globally. This reveals detail in both bright and dark areas of an image.

# Key Idea
> Instead of one global CDF, compute and apply local CDFs over image tiles → adapt to local lighting/contrast conditions.

# Method
- Divide image into contextual tiles.  
- Apply histogram equalization independently within each tile.  
- Interpolate at tile boundaries to reduce block artifacts.  

# Results
- Brings out local details invisible under global HE.  
- Useful in medical imaging and low-light conditions.  

# Limitations
- Strongly amplifies noise in homogeneous regions.  
- Produces unnatural artifacts when applied directly.  
- Motivated the development of **CLAHE** (contrast limiting).  

# Educational Connections
- Undergrad: Difference between global vs local contrast methods.  
- Postgrad: Trade-offs between adaptivity and noise robustness.  

---
