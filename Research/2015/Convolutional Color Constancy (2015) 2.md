---
title: Convolutional Color Constancy (2015)
aliases:
  - CCC
  - Convolutional Color Constancy
authors:
  - Jonathan T. Barron
year: 2015
venue: CVPR
doi: 10.1109/CVPR.2015.7298832
arxiv: https://arxiv.org/abs/1412.4334
citations: 1000+
dataset:
  - Gehler-Shi dataset
  - Cheng et al. dataset
tags:
  - paper
  - color-constancy
  - white-balance
  - computer-vision
fields:
  - vision
  - computational-photography
  - image-processing
related:
  - "[[Bayesian Color Constancy (Gehler et al., 2008)]]"
  - "[[Fast Fourier Color Constancy (2017)|Fast Fourier Color Constancy]]"
predecessors:
  - "[[Bayesian Color Constancy (2008)]]"
successors:
  - "[[Fast Fourier Color Constancy (2017)|Fast Fourier Color Constancy]]"
impact: ⭐⭐⭐⭐☆
status: read
---

# Summary
**Convolutional Color Constancy (CCC)** reframes color constancy (automatic white balance) as a **convolutional classification problem**. Images are projected into **log-chroma histograms**, and illuminant estimation becomes the task of convolving these histograms with learned filters to predict the correct white balance.  

# Key Idea
> Treat color constancy as a **filtering task**: apply learned convolutional filters on log-chroma histograms to locate the most likely illuminant.

# Method
- Represent input as a **2D histogram in log-chroma space**.  
- Train **linear convolutional filters** to map histograms → illuminant likelihood.  
- Pose as a **structured prediction problem**: illuminant = histogram location.  
- Regularization to avoid overfitting small datasets.  

# Results
- On Gehler-Shi dataset: achieved **state-of-the-art accuracy** at the time.  
- Outperformed Bayesian and learning-based predecessors.  
- Still slower (~500 ms per image) than practical deployment needs.  

# Why it Mattered
- First to successfully treat color constancy as a **convolutional problem**.  
- Laid groundwork for FFCC (FFT-based speedup).  
- Showed effectiveness of discriminative learning for AWB.  

# Architectural Pattern
- Input histogram → learned convolutional filters → illuminant estimate.  
- Output: **point estimate** of illuminant (no uncertainty).  

# Connections
- Predecessor to **FFCC (2017)**.  
- Related to Bayesian color constancy methods.  
- Inspired probabilistic extensions (FFCC).  

# Implementation Notes
- Requires histogram computation (expensive vs FFCC).  
- Computational bottleneck: spatial convolutions.  
- Best suited for offline or research usage, not real-time deployment.  

# Critiques / Limitations
- No probabilistic modeling → no uncertainty estimate.  
- Slow (hundreds of ms per image).  
- Still assumes a global single illuminant.  

# Repro / Resources
- [Paper PDF](https://arxiv.org/abs/1412.4334)  
- MATLAB implementation available in Barron’s repo.  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Image Processing**: histograms in color space.  
- **Convolutions**: linear filters applied to 2D arrays.  
- **Classification**: treating illuminant as class label.  

## Postgraduate-Level Concepts
- **Structured prediction** over histograms.  
- **Regularization for high-dimensional filters**.  
- **Comparisons with Bayesian MAP estimation**.  
- **Bridging to probabilistic models** (FFCC).  

---

# My Notes
- CCC = **first modern learning-based AWB model**.  
- FFCC’s brilliance was just speeding this up (FFT).  
- CCC shows how “classic” CNN-style thinking entered low-level vision.  

---
