---
title: Advancing White Balance Correction through Deep Feature Statistics and Feature Distribution Matching
aliases:
  - Feature Distribution Matching WB Correction
  - EFDM White Balance 2025
authors:
  - Furkan Kınlı
  - Barış Özcan
  - Furkan Kıraç
year: 2025
venue: Journal of Visual Communication and Image Representation
doi: 10.1016/j.jvcir.2025.104412
tags:
  - paper
  - white-balance
  - color-constancy
  - deep-learning
  - vision transformers
fields:
  - computational-photography
  - image processing
  - vision
related:
  - "[[Fast Fourier Color Constancy (2017) 2|Fast Fourier Color Constancy]]"
  - "[[Convolutional Color Constancy (2015)|Convolutional Color Constancy]]"
  - "[[Advancing White Balance Correction through Deep Feature Statistics and Feature Distribution Matching|Feature Distribution Matching WB Correction]]"
predecessors:
  - CCC
  - FFCC
  - other AWB deep methods (e.g. “Modeling Lighting as Style for AWB Correction” 2022)
successors:
  - future AWB methods leveraging distribution matching, possibly diffusion or style transfer-based
---

# Summary  
This work proposes a new method for automatic white balance (WB) correction that explicitly uses **deep feature statistics** and **feature distribution matching**. They treat lighting conditions as a style factor and introduce a loss function (Exact Feature Distribution Matching, EFDM) that aligns predicted and ground-truth feature distributions—including higher-order moments (mean, variance, skewness, kurtosis). They show that integrating this loss into transformer-based or UNet-based architectures improves correction, especially under complex and multi-illuminant lighting.

---

# Key Idea  
- Model lighting/illumination as a *style factor*, affecting distributional statistics of deep features.  
- Instead of just pixel-based or low-order statistical losses, align full feature distributions (many moments) between predicted image and ground truth.  
- Use vision transformer ([CLS] token) and architectures such as Uformer/UNet, but EFDM acts as a *loss objective*, not necessarily changing inference complexity.  

---

# Method  
- Input images from challenging WB datasets (e.g. LSMI) containing single- and multi-illuminant conditions.  
- Image processed to get predicted white balanced output.  
- Extract deep features via a pretrained or jointly trained ViT backbone (using the [CLS] token) from both predicted and ground-truth images.  
- Compute EFDM loss: align empirical cumulative distribution functions of feature values (not just mean/variance) → ensures alignment of higher order stats.  
- Train architectures like Uformer and UNet with standard WB correction losses (pixel-wise) + EFDM loss.  

---

# Results  
- On the LSMI dataset: the model with EFDM achieves lower error (MAE) in multi-illuminant scenarios than baselines.  
- Also shows more balanced performance: doesn’t drop too much when going from single-illuminant to multi-illuminant (via a metric called MSR, Multi-to-Single Ratio).  
- Ablation studies confirm that higher order statistics and EFDM loss contribute meaningfully beyond just mean/variance alignment.  

---

# Strengths & Why It Matters  
- Moves AWB correction forward by explicitly modeling feature distribution shifts—not only simple color/intensity changes.  
- Makes WB models more robust under realistic lighting (mixed sources, varied spectra).  
- EFDM is architecture-agnostic at inference; the extra cost is in training.  

---

# Limitations / Caveats  
- Still assumes global corrections; local illuminant variation across scene may remain hard.  
- Reliant on quality of ground truth WB and feature extractor.  
- EFDM adds computational overhead during training.  

---

# Connections & Trends  
- Builds on work like *Feature Distribution Statistics as Loss Objective for Robust WB Correction* by the same authors. :contentReference[oaicite:0]{index=0}  
- Aligns with trend of treating lighting as style (style transfer / domain adaptation) in AWB correction.  
- Related to earlier WB works like CCC, FFCC.  

---

# My Notes  
- This feels like a next step: from pixel/color histogram alignment → deep features/distribution alignment.  
- EFDM might help generalize across camera sensors (since feature distributions vary with sensor).  
- Could be interesting to combine with local illuminant estimation or spatial mapping to handle non-uniform lighting.  

---
