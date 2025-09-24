---
title: "Histogram Equalization (HE)"
aliases:
  - HE
  - Global Histogram Equalization
authors:
  - Early image processing community (no single canonical paper; widely used since 1960s–70s)
year: ~1970s
venue: "Image Processing / Computer Vision literature"
tags:
  - algorithm
  - image-processing
  - contrast-enhancement
  - computer-vision
fields:
  - vision
  - image-enhancement
related:
  - "[[Adaptive Histogram Equalization (AHE)]]"
  - "[[CLAHE: Contrast Limited Adaptive Histogram Equalization]]"
predecessors:
  - Linear contrast stretching
successors:
  - AHE
  - CLAHE
impact: ⭐⭐⭐⭐
status: "read"

---

# Summary
**Histogram Equalization (HE)** is a **global image contrast enhancement method**. It redistributes pixel intensity values so that the image histogram is approximately uniform, enhancing global contrast especially in low-contrast images.

# Key Idea
> Map intensity values using the **cumulative distribution function (CDF)** of the image histogram → stretches frequent intensities across the full dynamic range.

# Method
- Compute histogram of grayscale image.  
- Compute CDF.  
- Use CDF as mapping function to remap intensity levels.  
- Output image has enhanced global contrast.  

# Results
- Effective for images with poor contrast.  
- Simple, efficient, widely used baseline.  

# Limitations
- Global: may over-enhance or wash out local regions.  
- Amplifies noise in homogeneous areas.  
- Color images: applied per channel → may cause color distortions.  

# Educational Connections
- Undergrad: CDF mapping, histogram basics.  
- Postgrad: Contrast enhancement in different color spaces, perceptual evaluation.  

---
