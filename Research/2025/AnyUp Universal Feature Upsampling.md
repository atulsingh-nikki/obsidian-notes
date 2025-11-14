---
title: "AnyUp: Universal Feature Upsampling"
aliases:
  - AnyUp
  - Universal Feature Upsampling
  - Wimmer et al. (2025)
authors:
  - Thomas Wimmer
  - Prune Truong
  - Marie-Julie Rakotosaona
  - Michael Oechsle
  - Federico Tombari
  - Bernt Schiele
  - Jan Eric Lenssen
institutions:
  - Max Planck Institute for Informatics
  - ETH Zurich
  - Google
  - TU Munich
year: 2025
venue: "arXiv preprint arXiv:2510.12764"
tags:
  - feature-upsampling
  - vision-foundation-models
  - computer-vision
  - transformer
  - representation-learning
  - encoder-agnostic
fields:
  - multi-scale feature modeling
  - dense prediction
impact: ⭐⭐⭐⭐⭐
status: "read"
code: https://github.com/wimmerth/anyup
website: https://wimmerth.github.io/anyup

---

# 🧠 Summary

**AnyUp** introduces a **universal, feature-agnostic upsampling model** capable of reconstructing high-resolution feature maps from any visual encoder (DINO, CLIP, SigLIP, MAE, etc.) **without retraining**.  
Unlike prior upsamplers (FeatUp, LoftUp, JAFAR), which are tied to specific encoders or resolutions, AnyUp generalizes across:
- **Any encoder**,  
- **Any resolution**, and  
- **Any downstream task**.

It achieves this through a **feature-agnostic convolutional layer**, **windowed attention**, and a **crop-based training strategy**, outperforming all prior learned and heuristic approaches in both **accuracy** and **semantic fidelity**.

---

# 🎯 Core Idea

> Upsampling is reframed as a **feature-space reconstruction problem** independent of encoder type or feature dimensionality.

Given a low-resolution feature map $p \in \mathbb{R}^{h \times w \times c}$ and a guidance RGB image $I_{hr} \in \mathbb{R}^{H \times W \times 3}$,  
AnyUp learns a mapping  
$$
f(p, I_{hr}) \to q \in \mathbb{R}^{H \times W \times c}
$$  
that preserves semantic structure and feature-space consistency.

---

# 🧩 Key Contributions

1. **Feature-Agnostic Layer**
   - Convolves each input channel independently with a learned filter basis.
   - Averages activations across channels, producing **outputs invariant to input dimensionality**.
   - Enables direct inference on unseen feature types (e.g., DINOv3 trained model applied to SigLIP features).

2. **Local Window Attention**
   - Restricts attention to local windows, improving spatial coherence and reducing irrelevant cross-image dependencies.
   - Boosts computational efficiency and prevents feature “cross-talk.”

3. **Crop-Based Training Strategy**
   - Trains on **image parts** rather than full-resolution images.
   - Reduces training cost and avoids moving models out-of-distribution.

4. **Consistency Regularization**
   - Includes self-consistency and input-consistency losses for robust locality and feature preservation:
     $$
     L_{total} = L_{cos-mse} + L_{self-consistency} + L_{input-consistency}
     $$

---

# ⚙️ Architecture Overview

- Base backbone: attention-based (JAFAR-inspired)
- Inputs: RGB + feature map
- Outputs: high-resolution features
- Canonical dimensionality achieved via **feature-agnostic convolution**
- Localized attention windows replace global attention
- Lightweight (trainable on 1 GPU in ~5 hrs)

---

# 📊 Results

| Task | Dataset | Metric | AnyUp | Best Prior |
|------|----------|--------|--------|-------------|
| **Semantic Segmentation** | ADE20k | mIoU ↑ | **42.43** | LoftUp: 42.02 |
| | COCO-Stuff | mIoU ↑ | **62.16** | JAFAR: 61.82 |
| | PASCAL-VOC | mIoU ↑ | **84.00** | JAFAR: 84.36 |
| **Depth Estimation** | NYUv2 | RMSE ↓ | **0.4755** | FeatUp: 0.4816 |
| **Surface Normals** | NYUv2 | RMSE ↓ | **31.17°** | JAFAR: 31.54° |

✅ State-of-the-art results across all tasks  
✅ Preserves feature semantics better than LoftUp and FeatUp  
✅ Generalizes to **unseen encoders** (SigLIP, DINOv3)

---

# 🌐 Feature-Agnostic Generalization

| Train Encoder | Test Encoder | Result |
|---------------|---------------|---------|
| DINOv2 (ViT-S) | SigLIP-2 | Matches retrained LoftUp performance |
| DINOv2 | DINOv3 | Performs identically to retrained DINOv3 model |
| DINOv2 | ResNet / CLIP | Maintains high spatial and semantic fidelity |

> The model trained on **DINOv2 ViT-S** features successfully generalizes to **SigLIP-2 and DINOv3** without retraining.

---

# 🧪 Ablation Insights

| Component Removed | Effect |
|--------------------|---------|
| No window attention | +2% RMSE error |
| No crop training | Slight degradation in mIoU |
| No consistency loss | Higher variance across local features |
| No feature path in key computation | Still outperforms all prior works |

All proposed modules significantly contribute to AnyUp’s generalization and performance stability.

---

# ⚠️ Limitations
- Relies on **linear combination assumption** of low-res features → may miss sub-patch detail.
- Not explicitly trained for **denoising** positional encoding artifacts (though compatible with FeatSharp pre-denoisers).
- Future work: integrate multi-scale non-linear attention and diffusion-based refinement.

---

# 🧭 Educational Connections

| Level | Concepts |
|--------|-----------|
| **Undergraduate** | Upsampling, attention mechanisms, encoder-decoder structure |
| **Graduate** | Vision transformer feature-space alignment, domain generalization, attention-window regularization |
| **Research** | Encoder-agnostic modeling, feature-space consistency objectives, universal representation learning |

---

# 📚 References
- Couairon et al., *JAFAR: Jack Up Any Feature at Any Resolution*, 2025  
- Huang et al., *LoftUp: Learning a Coordinate-based Feature Upsampler*, 2025  
- Fu et al., *FeatUp: Model-Agnostic Feature Upsampling*, ICLR 2024  
- Ranzinger et al., *FeatSharp*, ICML 2025  
- Wimmer et al., *AnyUp: Universal Feature Upsampling*, arXiv:2510.12764  

---

# 🧩 Takeaway
> **AnyUp** represents a shift from *encoder-tied feature refinement* to a **universal, plug-and-play feature upsampler**, preserving semantics and structure across models, tasks, and resolutions —  
> bringing foundation-level interoperability to the vision ecosystem.

---
