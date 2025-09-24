---
title: "CLAHE: Contrast Limited Adaptive Histogram Equalization"
aliases:
  - CLAHE
  - Contrast Limited AHE
authors:
  - Zuiderveld, Karel (popularized in "Graphics Gems IV", 1994)
year: 1994
venue: Graphics Gems IV (Academic Press)
citations: 20,000+
tags:
  - algorithm
  - image-processing
  - contrast-enhancement
  - computer-vision
  - medical-imaging
fields:
  - vision
  - image-enhancement
  - computational-photography
related:
  - "[[Adaptive Histogram Equalization (AHE)]]"
  - "[[Retinex Methods]]"
  - "[[Histogram Equalization (HE)|HE]]"
predecessors:
  - Histogram Equalization (HE)
  - Adaptive Histogram Equalization (AHE)
successors:
  - Modern local contrast methods
  - Deep learning based contrast enhancement
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**CLAHE (Contrast Limited Adaptive Histogram Equalization)** is an image enhancement method that improves local contrast by applying histogram equalization adaptively on small image regions (tiles), while limiting amplification to avoid noise over-enhancement.  

Originally proposed in the medical imaging community, CLAHE became widely used in digital photography, computer vision, and preprocessing pipelines (e.g., OpenCV, MATLAB implementations).

# Key Idea
> Divide the image into contextual regions, apply local histogram equalization, and **clip histograms at a threshold** before redistribution to prevent over-amplification of noise.

# Method
1. Divide image into non-overlapping tiles (e.g., 8×8).  
2. For each tile: compute histogram and apply histogram equalization.  
3. **Clip histogram bins** above a clip limit (contrast limiting).  
   - Redistribute clipped pixels uniformly across bins.  
4. Interpolate between neighboring tiles to avoid block artifacts.  

# Results
- Enhances local details in dark/bright areas.  
- Reduces risk of noise amplification compared to AHE.  
- Produces natural-looking contrast improvements.  

# Why it Mattered
- Solved key weakness of **AHE** (over-enhancement).  
- Widely adopted in **medical imaging** (X-ray, MRI) and **photography**.  
- Still a standard baseline for local contrast enhancement.  

# Architectural Pattern
- Local histogram equalization + clipping + bilinear interpolation.  

# Connections
- Predecessor: **AHE** (Adaptive Histogram Equalization).  
- Related to **Retinex methods** for illumination correction.  
- Often compared with **Gamma correction** and modern deep-learning contrast methods.  

# Implementation Notes
- Parameters:  
  - `clipLimit` = contrast limiting threshold.  
  - `tileGridSize` = size of contextual regions.  
- Available in OpenCV (`cv::createCLAHE`) and MATLAB.  
- Computationally efficient for real-time use.  

# Critiques / Limitations
- Parameter sensitivity: poor choices can under/over enhance.  
- Still global per-channel; may distort color balance.  
- Can’t adapt to semantic scene structure.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Histogram equalization basics.  
- Local vs global contrast enhancement.  
- Why clipping prevents noise amplification.  

## Postgraduate-Level Concepts
- Trade-offs: tile size vs detail preservation.  
- Relation to perceptual models of vision.  
- Applications in medical image preprocessing and low-light photography.  
- Comparison with modern deep learning-based contrast correction.  

---

# My Notes
- CLAHE = **practical sweet spot**: better than global HE, safer than AHE.  
- Great example of how a small heuristic (clipping histograms) makes a method robust.  
- Still used in deep-learning pipelines as preprocessing for low-light / medical images.  
- Legacy: the “classic” non-deep local contrast enhancer.  

---
