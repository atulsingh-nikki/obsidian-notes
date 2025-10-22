---
title: "DeepGaze III: Modeling Human Visual Attention with Transformers (2022)"
aliases:
  - DeepGaze 3
  - DeepGaze III
authors:
  - Matthias Kümmerer
  - Thomas S. A. Wallis
  - Matthias Bethge
year: 2022
venue: Neural Information Processing Systems (NeurIPS)
tags:
  - paper
  - deep-learning
  - transformers
  - saliency
  - human-vision
  - probabilistic-modeling
fields:
  - computer-vision
  - computational-neuroscience
  - cognitive-science
impact: ⭐⭐⭐⭐⭐⭐
status: read
related:
  - "[[Learning to Predict Where Humans Look (ICCV 2009)]]"
  - "[[Foraging with the Eyes Dynamics in Human Visual Gaze and Deep Predictive Modeling|Foraging with the Eyes]]"
  - "[[DeepGaze II Predicting Fixations with Deeper Features and Probabilistic Modeling (2017)|DeepGaze 2]]"
---

# 🧠 Summary
**DeepGaze III** extended the DeepGaze lineage into the **transformer era**, incorporating **Vision Transformers (ViT)** for feature extraction and a **probabilistic attention decoder** for fixation prediction.  
It achieved **human-level performance** on the MIT300 and CAT2000 benchmarks — the first model to do so.

> “For the first time, we can quantitatively match human gaze prediction — not by mimicking heuristics, but by modeling the underlying perceptual representation.”

---

# 🧩 Key Insights

| Aspect | Description |
|--------|--------------|
| **Feature Backbone** | Vision Transformer (ViT-B/16 pretrained on ImageNet-21K). |
| **Decoder** | Lightweight CNN + learned spatial prior for fixation density. |
| **Learning Objective** | Probabilistic log-likelihood + information gain (as in DeepGaze II). |
| **Result** | Achieved human inter-observer-level agreement on major benchmarks. |
| **Significance** | Unified transformer-level perception with probabilistic neuroscience modeling. |

---

# 🔬 Methodology

### 1️⃣ Feature Extraction
- Extract multi-scale features from **ViT** encoder layers (tokens reshaped to spatial grids).
- Each token encodes global and contextual information, improving scene-level reasoning.

### 2️⃣ Fixation Decoder
- Multi-layer convolutional projection maps transformer embeddings → saliency scores.
- Additive **learned center bias** (as in DeepGaze II):
  \[
  P(x,y|I) = \frac{e^{S(x,y) + C(x,y)}}{\sum_{x',y'} e^{S(x',y') + C(x',y')}}
  \]

### 3️⃣ Training
- Optimize **log-likelihood of fixation points**:
  \[
  \mathcal{L} = -\sum_{(x,y)} \log P(x,y|I)
  \]
- Information-theoretic metrics (IG) for model comparison.

### 4️⃣ Evaluation
- Datasets: MIT1003, MIT300, CAT2000, COCO-Freeview.
- Metrics: Information Gain (IG), NSS, KL divergence.

---

# 📊 Results

| Benchmark | IG (bits/fixation) | Status |
|------------|--------------------|---------|
| **MIT300** | 1.19 ± 0.02 | Matches human IOB baseline |
| **CAT2000** | 1.15 | State of the art |
| **COCO-Freeview** | 1.11 | Best reported |
| **AUC / NSS** | Comparable to or exceeding human fixations |

---

# 🌍 Impact and Legacy
- **First model** to achieve **human-level attention prediction** under benchmark evaluation.  
- Reframed saliency as a **representation-learning** problem rather than explicit feature engineering.  
- Demonstrated that **transformer attention** can mirror **biological visual attention**.  
- Serves as the foundation for **neural saliency maps in multimodal models** (e.g., CLIP, Flamingo, GPT-4V attention studies).

---

# ⚙️ Architecture Overview

---
Image → Vision Transformer (ViT) → spatial feature tokens → CNN decoder → saliency map S(x,y)  
+ learned center prior → softmax → fixation probability P(x,y|I)
---

# 💬 Discussion Highlights
- ViT attention heads mirror biological foveation and scene parsing.  
- Learned global context explains why DeepGaze III captures **object-level saliency** without explicit face/text priors.  
- Probabilistic framing continues to outperform purely pixel-wise regression.  
- Empirical finding: fixation entropy correlates with semantic richness of the image.  

---

# 🧩 Educational Connections

| Level | Concepts |
|--------|----------|
| **Undergraduate** | Vision Transformers, attention maps, saliency prediction. |
| **Graduate** | Probabilistic modeling of gaze, information gain metrics, human–AI perceptual alignment. |

---

# 📚 References
- Judd et al., *Learning to Predict Where Humans Look*, ICCV 2009  
- Kümmerer et al., *DeepGaze II*, CVPR 2017  
- Dosovitskiy et al., *ViT*, ICLR 2021  
- Kümmerer et al., *DeepGaze III*, NeurIPS 2022  

---

# 🧭 Takeaway
> **DeepGaze III** closes the human–machine gap in saliency prediction —  
> using transformer features and probabilistic modeling to match the way humans allocate visual attention.
