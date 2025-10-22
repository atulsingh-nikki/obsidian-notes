---
title: "Foraging with the Eyes: Dynamics in Human Visual Gaze and Deep Predictive Modeling"
aliases:
  - Foraging with the Eyes
  - Gaze as Lévy Walk
authors:
  - Tejaswi V. Panchagnula
affiliation: "Indian Institute of Technology Madras"
year: 2025
venue: "arXiv preprint arXiv:2510.09299"
citations: "New, July 2025"
tags:
  - paper
  - eye-tracking
  - gaze-modeling
  - human-vision
  - levy-walk
  - cnn
fields:
  - cognitive-vision
  - human-computer-interaction
  - computational-neuroscience
impact: ⭐⭐⭐⭐
status: "read"
related:
  - "[[Learning to Predict Where Humans Look (Judd et al., 2009)]]"
  - "[[DeepGaze I (Kümmerer et al., 2015)]]"
  - "[[Beyond Pixels: Predicting Human Gaze (Xu et al., 2014)]]"

---

# 🧠 Summary

This study proposes that **human visual gaze behaves like a Lévy walk**, similar to **animal foraging** in sparse environments.  
It bridges **visual attention modeling** with **movement ecology**, showing that human eyes “forage” for information optimally.

> The gaze trajectory is not random noise — it follows heavy-tailed step length distributions, balancing short fixations with long exploratory saccades.

The author further builds a **CNN-based fixation prediction model** that learns attention heatmaps from images, demonstrating the *learnability of spatial fixation behavior* — though not its full temporal stochasticity.

---

# 🧩 Key Insights

| Aspect | Description |
|--------|--------------|
| **Core Hypothesis** | Human gaze follows a Lévy walk — a heavy-tailed stochastic process optimized for information foraging. |
| **Experiment** | 40 participants, 50 diverse images, 4 million gaze points via 120 Hz Aurora Smart Eye Tracker. |
| **Key Finding** | Eye movements exhibit power-law distributed step lengths (1 < μ ≤ 3), indicating Lévy-like exploration. |
| **Entropy Correlation** | Higher image entropy → longer average gaze jumps. |
| **Turning Angle** | Bimodal distribution — vertical preference (±90°) and straight-line persistence (0 radians). |
| **CNN Model** | Predicts fixation heatmaps from images using MobileNetV2 + U-Net decoder. |
| **Loss** | BCE + MSE + KL divergence composite loss. |
| **Outcome** | Accurately predicts fixation clusters but fails to model long-range Lévy-like jumps. |

---

# 🔬 Methodology

### 🧪 Data Collection
- 40 participants, free-viewing 50 images for 30 s each.
- Two groups (25 images each) balanced by image entropy.
- ~110k gaze points per subject; ~4 million total.
- Entropy computed via Shannon measure \(H = -\sum p(i)\log_2 p(i)\).

### 📊 Statistical Analysis
- Step length = distance between successive gaze points.
- Log–log slope of −2.2 → −2.4 per image ⇒ Lévy regime.
- Cumulative slope ≈ −3.5 → Gaussian aggregation (Central Limit Theorem).
- Positive correlation between image entropy and Lévy slope (μ).
- Turning angles show structured exploration (not random).

### 🧠 Modeling
- **Architecture:** MobileNetV2 encoder + transposed conv decoder → fixation map.
- **Loss:** \(L = 0.4·BCE + 0.3·MSE + 0.3·DKL(H‖\hat{H})\)
- **Training:** AdamW, cosine annealing, 10 epochs.
- **Output:** \( \hat{H} = D(E(I)) \), where \(\hat{H}\) = predicted fixation heatmap.

### 📈 Results
- High-quality fixation maps with strong spatial correlation.
- Validation ≈ training loss (good generalization).
- Captures multimodal saliency clusters, not Lévy-style long saccades.

---

# 🌍 Conceptual Link — “Foraging with the Eyes”

- Eye movement = **information foraging**.
- Short fixations ↔ local inspection; long saccades ↔ exploratory search.
- Analogous to **albatross flight patterns** and **animal foraging** under sparse-resource conditions.
- Visual system exhibits *optimal efficiency within biological energy constraints (~15 W)*.

---

# 🧩 Key Equations

1. **Entropy of an Image:**
   \[
   H = -\sum_{i=0}^{L-1} p(i)\log_2 p(i)
   \]

2. **Lévy Walk Step Distribution:**
   \[
   P(l) \sim |l|^{-\mu}, \quad 1 < \mu \le 3
   \]

3. **CNN Loss Function:**
   \[
   L = \alpha \, BCE + \beta \, MSE + \gamma \, D_{KL}(H \parallel \hat{H})
   \]

---

# ⚙️ Model Evaluation

| Metric | Observation |
|--------|--------------|
| **Qualitative Fit** | Accurate clustering of salient zones |
| **Quantitative** | BCE, MSE, KL losses converge smoothly |
| **Failure Mode** | Misses heavy-tailed long jumps |
| **Interpretation** | Static heatmaps ≠ dynamic gaze trajectories |

---

# 🧭 Discussion Highlights

- Gaze is **stochastic but structured**, shaped by both image entropy and human cognitive patterns.
- **Static saliency models** ignore the sequential nature of attention.
- **Autoregressive RNN/Transformer models** failed — gaze lacks consistent temporal order.
- Predictive modeling benefits from *distributional rather than sequential* representations.
- Applications:
  - Attention prediction for AR/VR
  - Adaptive HCI interfaces
  - Early diagnosis (autism, ADHD, neurodegeneration)

---

# 📚 Connections

| Domain | Insight |
|--------|----------|
| **Movement Ecology** | Visual scanpaths = Lévy-like foraging trajectories |
| **Cognitive Science** | Eye movements reveal information optimization |
| **Computer Vision** | CNNs can learn fixation probability but not dynamic stochasticity |
| **Statistical Physics** | Heavy-tailed distributions in human movement patterns |

---

# 🧩 Future Work

- Temporal modeling (e.g., RNN + transformer extensions).
- Gaze-conditioned generative models.
- Personalized gaze modeling across demographics.
- Coupling entropy-based image priors with dynamic foraging theory.

---

# 📖 References
Key works cited include:
- Judd et al. *Learning to Predict Where Humans Look* (ICCV 2009)
- Xu et al. *Predicting Human Gaze Beyond Pixels* (JoV 2014)
- Kümmerer et al. *DeepGaze I* (ICLR 2015)
- Viswanathan et al. *Lévy Flights in Random Searches* (Physica A 2000)
- Martinez-Conde et al. *Microsaccades and Visual Stability* (Nat Rev Neuro 2013)
- Brockmann et al. *Scaling Laws of Human Travel* (Nature 2006):contentReference[oaicite:1]{index=1}

---

# 🧭 Takeaway

> The human visual system "forages" for information with Lévy-like efficiency —  
> blending local fixation with global exploration —  
> a bridge between **ecological movement**, **statistical physics**, and **cognitive vision**.

---
