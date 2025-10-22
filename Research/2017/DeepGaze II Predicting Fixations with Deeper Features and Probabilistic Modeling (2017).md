---
title: "DeepGaze II: Predicting Fixations with Deeper Features and Probabilistic Modeling (2017)"
aliases:
  - DeepGaze II
  - DeepGaze 2
authors:
  - Matthias Kümmerer
  - Thomas S. A. Wallis
  - Matthias Bethge
year: 2017
venue: IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
tags:
  - paper
  - deep-learning
  - saliency
  - human-vision
  - probabilistic-modeling
fields:
  - computer-vision
  - computational-neuroscience
  - cognitive-vision
impact: ⭐⭐⭐⭐⭐
status: read
related:
  - "[[Learning to Predict Where Humans Look (ICCV 2009)]]"
  - "[[Foraging with the Eyes Dynamics in Human Visual Gaze and Deep Predictive Modeling|Foraging with the Eyes]]"
  - "[[DeepGaze I Deep Features for Predicting Human Eye Fixations (2015)|DeepGaze I]]"
---

# 🧠 Summary
**DeepGaze II** advanced saliency modeling by combining **VGG-19 features** with a **fully probabilistic framework**, treating human fixations as samples from a learned probability distribution over pixels.  
It refined **DeepGaze I** with deeper features, non-linear readouts, and stronger statistical interpretation.

> “Saliency maps are not heatmaps—they’re probability distributions of human fixation.”

---

# 🧩 Key Insights

| Aspect | Description |
|--------|--------------|
| **Features** | Extracted from VGG-19 (pretrained on ImageNet). |
| **Readout** | Learned non-linear layer mapping VGG features → fixation probabilities. |
| **Learning Objective** | Maximize log-likelihood of human fixations (information gain over baseline). |
| **Metric** | Used **Information Gain (IG)** instead of AUC to quantify predictivity. |
| **Result** | First model approaching the explainable variance of human inter-observer agreement. |

---

# 🔬 Methodology

### 1️⃣ Representation
- Deep features from multiple VGG convolutional layers (conv5_1, conv5_2, etc.).
- Upsampled and concatenated into a multi-scale spatial tensor.

### 2️⃣ Readout Network
- 1×1 convolutions + ReLU nonlinearity → saliency score map.
- Combined with a **learned center bias prior**:
  \[
  P(x,y|I) = \frac{e^{S(x,y) + C(x,y)}}{\sum_{x',y'} e^{S(x',y') + C(x',y')}}
  \]

### 3️⃣ Training Objective
\[
\mathcal{L} = -\sum_{(x,y) \in \text{fixations}} \log P(x,y|I)
\]
→ equivalent to maximizing **log-likelihood** of observed fixations.

### 4️⃣ Evaluation
- Benchmarks: MIT1003, MIT300, CAT2000 datasets.
- Reported **Information Gain (bits/fixation)** over baseline center prior.

---

# 📊 Results

| Metric | Result |
|---------|--------|
| **Information Gain (IG)** | 60–70% of explainable human variance. |
| **Improvement over DeepGaze I** | +45% IG, +12% NSS. |
| **Visual Quality** | Sharper, semantically accurate fixation maps. |

---

# 🌍 Impact and Legacy
- Established **probabilistic saliency** as the standard formulation.  
- Inspired **SALICON**, **SAM-ResNet**, **SalGAN**, and transformer-based gaze models.  
- Provided a bridge between **computational neuroscience** and deep learning attention maps.  
- Its probabilistic framing was later adapted in **Bayesian saliency** and **temporal gaze modeling** papers.

---

# ⚙️ Architecture Diagram

---
Image → VGG feature pyramid → 1×1 conv layers → saliency map (S)  
+ learned center bias → softmax → fixation probability P(x,y|I)
---

# 💬 Discussion Highlights
- Deep CNNs implicitly encode attention priors from object recognition tasks.  
- Probabilistic modeling avoids arbitrary normalization or post-hoc scaling.  
- Center bias is not a nuisance — it’s an intrinsic part of visual exploration.  
- Human fixations are explainable as a mixture of **bottom-up saliency** and **top-down semantics**.

---

# 🧩 Educational Connections

| Level | Concepts |
|--------|----------|
| **Undergraduate** | CNNs, transfer learning, visual attention. |
| **Graduate** | Probabilistic modeling of fixation maps; log-likelihood as training objective. |

---

# 📚 References
- Kümmerer et al., *DeepGaze I*, ICLR Workshop 2015  
- Simonyan & Zisserman, *VGG-19*, ICLR 2015  
- Judd et al., *Learning to Predict Where Humans Look*, ICCV 2009  
- Borji & Itti, *CAT2000 Benchmark*, 2015  

---

# 🧭 Takeaway
> **DeepGaze II** turned saliency prediction into a statistically grounded, deep feature-driven discipline —  
> moving beyond “heatmaps” to **probabilistic gaze modeling**, linking neural vision and human attention.
