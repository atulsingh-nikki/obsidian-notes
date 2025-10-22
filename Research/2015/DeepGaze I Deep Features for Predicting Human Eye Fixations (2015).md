---
title: "DeepGaze I: Deep Features for Predicting Human Eye Fixations (2015)"
aliases:
  - DeepGaze I
  - DeepGaze 1
authors:
  - Matthias Kümmerer
  - Lucas Theis
  - Matthias Bethge
year: 2015
venue: ICLR (Workshop Track)
tags:
  - paper
  - deep-learning
  - saliency
  - eye-tracking
  - human-attention
fields:
  - computer-vision
  - computational-neuroscience
  - cognitive-vision
impact: ⭐⭐⭐⭐⭐
status: read
related:
  - "[[Learning to Predict Where Humans Look (ICCV 2009)]]"
  - "[[DeepGaze II: Predicting Fixations with Deeper Features and Probabilistic Modeling (2017)]]"
  - "[[Foraging with the Eyes Dynamics in Human Visual Gaze and Deep Predictive Modeling|Foraging with the Eyes]]"
---

# 🧠 Summary
**DeepGaze I** was the first to use **pre-trained deep neural features** (from AlexNet) to model human visual attention.  
It demonstrated that high-level semantic features learned for object recognition naturally align with human fixation patterns — even without explicit supervision for saliency.

> “Pretrained CNNs already carry rich spatial priors for where humans look.”

---

# 🧩 Key Insights

| Aspect | Contribution |
|--------|---------------|
| **Input features** | AlexNet (ImageNet-trained) convolutional feature maps. |
| **Learning** | A linear readout layer trained to predict fixation density maps. |
| **Output** | Probabilistic saliency map (log-likelihood based). |
| **Evaluation** | MIT300 saliency benchmark. |
| **Main Result** | Outperformed all prior classical models (Itti, Judd, GBVS, etc.). |

---

# 🔬 Methodology

### 1️⃣ Deep Feature Extraction
- Take **conv5** layer activations from **AlexNet**.  
- Apply a **logistic regression** or **1×1 convolutional readout** to predict fixation probability.

### 2️⃣ Training Objective
- Train on MIT1003 dataset (same base as Judd et al.).  
- Optimize **log-likelihood of human fixations** under predicted saliency distribution.

$$
\mathcal{L} = -\sum_{(x,y) \in \text{fixations}} \log P(x,y \mid I)
$$

### 3️⃣ Center Bias
- Learned separately as an additive prior $ P_c(x,y) $.  
- Combined: $ P(x,y|I) \propto e^{S(x,y) + P_c(x,y)} $.

---

# 📈 Results
- Top performer on MIT300 (2015).  
- Validated that **object-centric CNNs predict attention implicitly**.  
- Showed strong correlation between high-level semantics and human gaze.

---

# 🌍 Impact and Legacy
- Shifted saliency research from **hand-crafted cues → deep features**.  
- Introduced probabilistic framing for saliency (not just “heatmaps”).  
- Direct ancestor of **DeepGaze II**, **SAM**, **SalGAN**, and transformer-based gaze models.  

---

# ⚙️ Architecture Sketch

---

Image → AlexNet conv layers → linear readout → saliency map → softmax → fixation probability

---

# 💬 Discussion Highlights
- CNN features pre-trained for object recognition generalize surprisingly well to human fixation.  
- Learning from deep representations reduces the need for explicit “face” or “text” detectors.  
- The model still lacks temporal and top-down cognitive context.  

---

# 🧩 Educational Connections

| Level | Concepts |
|-------|-----------|
| **Undergraduate** | CNN feature extraction; image classification vs. saliency. |
| **Graduate** | Transfer learning for attention; probabilistic modeling of fixations. |

---

# 📚 References
- Judd et al. (2009) Learning to Predict Where Humans Look  
- Krizhevsky et al. (2012) AlexNet  
- Kümmerer et al. (2015) DeepGaze I (ICLR Workshop)  
- Kümmerer et al. (2017) DeepGaze II (CVPR)  

---

# 🧭 Takeaway
> **DeepGaze I** proved that semantic knowledge embedded in CNNs mirrors the way humans allocate visual attention — a milestone linking perception, cognition, and learned representation.

