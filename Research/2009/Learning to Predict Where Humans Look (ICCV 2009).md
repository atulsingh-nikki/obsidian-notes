---
title: Learning to Predict Where Humans Look (ICCV 2009)
aliases:
  - Judd et al. 2009
  - Learning to Predict Human Fixations
  - MIT Saliency Benchmark Paper
authors:
  - Tilke Judd
  - Krista A. Ehinger
  - Frédo Durand
  - Antonio Torralba
year: 2009
venue: IEEE International Conference on Computer Vision (ICCV)
tags:
  - paper
  - saliency
  - human-attention
  - eye-tracking
  - computer-vision
fields:
  - cognitive-vision
  - image-understanding
  - human-computer-interaction
impact: ⭐⭐⭐⭐⭐
status: read
related:
  - "[[Beyond Pixels: Predicting Human Gaze (Xu et al., 2014)]]"
  - "[[Foraging with the Eyes Dynamics in Human Visual Gaze and Deep Predictive Modeling|Foraging with the Eyes]]"
  - "[[DeepGaze I Deep Features for Predicting Human Eye Fixations (2015)|DeepGaze 1]]"
---

# 🧠 Summary
The first large-scale **computational saliency** model trained directly on **human eye-tracking data**.  
It moved the field from hand-crafted heuristics toward **data-driven attention prediction**, combining low-, mid-, and high-level features to estimate fixation probability across an image.

> “Humans look at faces, text, and objects, not just local contrast.” — Judd et al., 2009

---

# 🧩 Key Insights

| Aspect | Contribution |
|--------|---------------|
| **Dataset** | 1,003 natural images, 15 viewers × 3 s each → ~1 M fixations. |
| **Feature Hierarchy** | Intensity, color, orientation, horizon, face, object, and depth cues combined. |
| **Learning Approach** | Linear SVM trained on fixation vs. non-fixation pixels. |
| **Result** | Outperformed classic saliency models (e.g., Itti & Koch 1998) by > 30 %. |
| **Key Finding** | High-level semantics (faces, people, text) are critical for human attention. |

---

# 🔬 Methodology

### 1️⃣ Feature Extraction
- **Low-level:** intensity, color opponency, orientation (Gabor filters).  
- **Mid-level:** horizon line, depth gradient, local contrast.  
- **High-level:** face detections (Viola–Jones), people, text, and objectness maps.

### 2️⃣ Learning
- Labeled fixation pixels (positive) vs non-fixated (negative).  
- Linear SVM → weights over all features → saliency map.  
- Cross-validation per subject to test generalization.

### 3️⃣ Evaluation Metrics
- AUC ROC between predicted map and human fixations.  
- Compared to Itti–Koch (1998), GBVS (2006), Torralba’s contextual prior.

---

# 📊 Results Summary
- Achieved best quantitative fit to human fixations among 2009 models.  
- Demonstrated importance of **top-down features** (faces/text).  
- Introduced the MIT Saliency Benchmark dataset → foundation for later DeepGaze, SALICON etc.

---

# 🌍 Impact and Legacy
- First proof that saliency is not purely bottom-up.  
- Anchored attention modeling in machine learning rather than hand-tuned maps.  
- Inspired modern deep saliency networks (e.g., DeepGaze I/II, SalGAN, SAM).  
- Dataset still used as benchmark for fixation prediction research.

---

# ⚙️ Key Equation
Linear combination of feature maps:
\[
S(x,y) = \sum_{i} w_i \, f_i(x,y)
\]
where \(S\) = saliency score, \(f_i\) = feature map, \(w_i\) = learned weight.

---

# 💡 Discussion Highlights
- Human attention follows semantic priors (“look at people and faces”).  
- Scene layout bias: center and horizon lines are natural anchors.  
- Eye movements show consistent patterns across subjects (≈ 70 % overlap).  
- Provided early bridge between psychophysics and machine vision.  

---

# 🧩 Connections

| Lineage | Progression |
|----------|--------------|
| **Classic Saliency (1998)** | Itti & Koch — purely bottom-up. |
| **Learning Saliency (2009)** | Judd et al. — adds semantic features + supervised learning. |
| **Deep Saliency (2015+)** | DeepGaze I/II, SAM — CNNs learn features end-to-end. |

---

# 📚 References
- Itti, Koch & Niebur, *IEEE TPAMI*, 1998  
- Harel et al., *GBVS*, NIPS 2006  
- Judd et al., *ICCV* 2009 (main paper)  
- Kümmerer et al., *DeepGaze I*, ICLR 2015  
- Xu et al., *Beyond Pixels*, JoV 2014  

---

# 🧭 Takeaway
> **Judd et al. (2009)** reframed saliency as a learnable problem:  
> humans don’t just notice contrast — they seek meaning.  
> This paper opened the path to modern deep attention models.

---
