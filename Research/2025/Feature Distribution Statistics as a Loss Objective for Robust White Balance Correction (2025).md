---
title: Feature Distribution Statistics as a Loss Objective for Robust White Balance Correction (2025)
aliases:
  - FDM Loss WB Correction
  - EFDM WB Kınlı & Kıraç 2025
authors:
  - Furkan Kınlı
  - Furkan Kıraç
year: 2025
venue: Machine Vision and Applications, Vol 36(3), Article 58
doi: 10.1007/s00138-025-01680-1
tags:
  - white-balance
  - color-constancy
  - deep-learning
  - feature-distribution
  - illuminant estimation
fields:
  - computational-photography
  - image-processing
  - vision
related:
  - "[[Advancing White Balance Correction through Deep Feature Statistics and Feature Distribution Matching]]"
  - "[[Convolutional Color Constancy (2015) 2|Convolutional Color Constancy]]"
  - "[[Fast Fourier Color Constancy (2017) 2|Fast Fourier Color Constancy]]"
predecessors:
  - CCC, FFCC
successors:
  - future WB methods that use distributional feature losses
---

# Summary  
This work introduces a new loss objective—**Exact Feature Distribution Matching (EFDM)**—to improve robustness of white balance (WB) correction, especially in multi-illuminant and non-uniform lighting conditions. The core idea is aligning feature distributions (not just pixel-wise or low-order statistics) between predicted WB outputs and ground truth, using higher-order moments (mean, variance, skewness, kurtosis). They integrate this with architectures like Uformer and UNet, and show improvements on the LSMI dataset over standard loss functions like MSE or simpler statistical alignment (mean/variance only). :contentReference[oaicite:0]{index=0}

---

# Key Idea  
- Lighting is treated as a “style” factor that changes the distribution of deep features across images.  
- Instead of only matching pixel values, match entire feature distributions to make WB correction more perceptually robust.  
- Use EFDM loss applied to the [CLS] token of a Vision Transformer (ViT) to get global summary of feature distribution. :contentReference[oaicite:1]{index=1}

---

# Method  

- **Backbone architectures**: Uformer (transformer-based) and UNet (CNN). Both are used with the new loss. :contentReference[oaicite:2]{index=2}  
- **Feature extraction**: Use a pretrained ViT; extract the [CLS] token features from both predicted corrected image and ground truth image. :contentReference[oaicite:3]{index=3}  
- **Loss functions**:  
  - EFDM loss: matches empirical cumulative distribution functions (eCDFs) of feature values—so not just mean and variance, but skewness & kurtosis. :contentReference[oaicite:4]{index=4}  
  - Baseline/Ablation: compare to MSE, or mean/variance alignment (e.g. AdaIN-style). :contentReference[oaicite:5]{index=5}  
- **Color space / target modeling**: Input images are transformed (chromatic channels UV in some color space) etc. They reconstruct corrected RGB via predicted illuminant maps. :contentReference[oaicite:6]{index=6}  
- **Evaluation metrics**: Mean absolute error (MAE) in color correction; introduced a “Multi-to-Single Ratio” (MSR) to measure how well models generalize from single to multi-illuminant settings. :contentReference[oaicite:7]{index=7}

---

# Results  

- Demonstrated improved performance over standard loss-based methods (MSE etc.) in diverse lighting, especially multi-illuminant scenes. :contentReference[oaicite:8]{index=8}  
- The EFDM loss gives more balanced error across camera devices (LSMI dataset has samples from several cameras). :contentReference[oaicite:9]{index=9}  
- Qualitative improvements: fewer artifacts in regions with mixed lighting, better visual plausibility in shadows/highlights. :contentReference[oaicite:10]{index=10}  

---

# Why It Matters / Strengths  

- Tackles a weakness of many WB correction methods: they often assume global uniform illuminant or rely only on low-order statistics, which break under complex lighting.  
- EFDM makes the model more robust to distributional shifts in features caused by lighting.  
- Uses transformer feature ([CLS] token) to capture global scene style without heavy overhead at inference (loss-only change).  
- Loss‐objective agnostic of architecture → can plug into different backbones.  

---

# Limitations / Open Issues  

- Global [CLS] token gives global corrective behavior, but **local variation** (e.g. shadows, spatially varying illuminants) may still pose problems.  
- Higher order matching (skewness, kurtosis) requires enough data / good training to avoid instability.  
- Some “residual color shifts” in extreme lighting or mixed sources remain. :contentReference[oaicite:11]{index=11}  
- The EFDM is a training-time cost; inference complexity doesn’t increase much, but training may be heavier.  

---

# Architectural Pattern  

- Input image → predicted illumination map (e.g. UV or chromatic channels) → generate corrected RGB → extract deep features via ViT (for predicted & GT) → compute EFDM loss → backprop through WB correction network.  
- Alternative backbone: CNN (UNet) vs transformer (Uformer).  

---

# Critiques / Considerations  

- How much does matching entire feature distribution help vs just matching up to 2nd moment? They do ablation, but still, marginal returns vs complexity?  
- Dependence on ViT pretrained feature extractor: are biases introduced by that model (trained on generic images) limiting this alignment?  
- How does this method behave under unknown camera sensors / sensors with very different color response curves?  

---

# Educational Connections  

## Undergraduate Level  
- Why matching only mean/variance is insufficient in some tasks.  
- What skewness, kurtosis are, intuitively (e.g. asymmetry, tails of distribution).  
- [CLS] token in ViT as global summary.  

## Graduate Level  
- Empirical CDF (eCDF) based distribution matching; EFDM loss mathematics.  
- Generalization to multi-illuminant lighting; evaluation metric MSR.  
- Comparing CNN vs transformer backbones under this loss.  

---

# Possible Extensions / My Notes  

- Could try patch-wise EFDM to handle local illuminants better.  
- Explore if using multiple feature levels (not just [CLS]) gives richer alignment.  
- Could combine EFDM with spatial priors (illumination maps, depth) to correct mixed lighting more precisely.  
- Evaluate how this loss interacts with perceptual color differences (e.g. CIEDE2000) — is distribution matching perceptually aligned?

---

## Citations  
“Feature distribution statistics as a loss objective for robust white balance correction.” Furkan Kınlı & Furkan Kıraç. Machine Vision and Applications, 2025. :contentReference[oaicite:12]{index=12}  

---
