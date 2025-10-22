---
title: "Color Transfer Using Probabilistic Moving Least Squares (PMLS)"
aliases:
  - Probabilistic Moving Least Squares
  - PMLS
  - Hwang et al. (CVPR 2014)
authors:
  - Youngbae Hwang
  - Joon-Young Lee
  - In So Kweon
  - Seon Joo Kim
year: 2014
venue: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"
tags:
  - paper
  - color-transfer
  - nonparametric-modeling
  - computer-vision
  - computational-photography
fields:
  - image-processing
  - color-consistency
impact: ⭐⭐⭐⭐⭐
status: "read"
related:
  - "[[Reinhard Color Transfer (2001)]]"
  - "[[Polynomial Color Mapping (Ilie & Welch, ICCV 2005)]]"
  - "[[Unsupervised Local Color Correction (Oliveira et al., CVPR 2011)]]"

---

# 🧠 Summary
This paper proposes a **Probabilistic Moving Least Squares (PMLS)** framework for **color transfer** between images, handling nonlinear, nonparametric mappings in the 3D RGB space.  
By combining **moving least squares interpolation** with **probabilistic modeling**, it corrects color inconsistencies across illumination changes, camera differences, or stylistic edits more robustly than previous methods:contentReference[oaicite:0]{index=0}.

> “We solve for a full nonlinear and nonparametric color mapping in 3D RGB space using scattered point interpolation and probabilistic modeling to handle noise and misalignments.” — Hwang et al., CVPR 2014:contentReference[oaicite:1]{index=1}

---

# 🎯 Core Idea
Color transfer is treated as a **scattered interpolation problem**:
$$
x' = T_x(x) = A_x(x - \mu_u) + \mu_v
$$
where $A_x$ and $\mu_v$ are locally fitted affine transforms computed via **moving least squares (MLS)**.  
A **probabilistic weighting term** ensures robustness to noisy or mismatched color correspondences between source and reference images.

---

# 🔬 Methodology

### 1️⃣ Moving Least Squares
For a color \(x\) and corresponding color pairs $(u_k, v_k)$:
$$
\min_{T_x} \sum_k w_k \|T_x(u_k) - v_k\|^2, \quad w_k = \frac{1}{\|u_k - x\|^{2\alpha}}
$$
produces a local affine mapping $T_x(x) = A_x(x - u) + v$.

### 2️⃣ Probabilistic Weighting
Define the **reliability** of each color match using bidirectional probabilities:
$$
p(M\{I(i), J(j)\}) = 
\frac{p(I(i),J(j))^2}
{\sum_k p(I(i),J(k)) \sum_k p(I(k),J(j))}
$$
Then the final PMLS weight:
$$
w_k = \frac{1}{\|u_k - x\|^{2\alpha} + \epsilon} \times p(M\{I(i), J(j)\})
$$
→ blends geometric proximity and probabilistic reliability.

### 3️⃣ Extrapolation
When color overlap between source and target is sparse, synthetic control points are added using a **2nd-order polynomial color model**, ensuring smooth mapping for unseen colors:contentReference[oaicite:2]{index=2}.

---

# ⚙️ Implementation
- Control points from image registration (e.g., SIFT flow, planar homography).  
- Parallelized computation on GPU (CUDA), ~4.5s for 1MP images.  
- Typical parameters: α = 2, ε = 1, 20×20×20 RGB bins.

---

# 📊 Results

| Dataset | Competing Methods | Metrics | Outcome |
|----------|-------------------|----------|----------|
| Tonal Adjustment (Bychkovsky et al. 2011) | Reinhard [16], Polynomial [7], CIM [14], Tai [19] | PSNR, SSIM | PMLS achieves highest PSNR and SSIM |
| Cross-camera (iPhone ↔ Canon) | Linear, Polynomial, CIM | PSNR | PMLS outperforms by 1–2 dB |
| Retouching / Style Transfer | Global tone-mapping | Visual quality | Seamless transitions, no haloing |

**Quantitative:** PMLS outperforms all baselines in PSNR and SSIM across datasets (tonal, illumination, cross-camera, and photo-retouching):contentReference[oaicite:3]{index=3}.  
**Qualitative:** Visual mosaics show seamless transitions between reference and transferred colors (Fig. 4 in paper).

---

# 🧩 Applications
- **Cross-camera color consistency:** iPhone ↔ DSLR calibration.  
- **Photometric alignment for panoramas:** color-consistent stitching without blending.  
- **Video color transfer:** user edits one frame; learned transfer applied to the entire sequence (see Fig. 7).  

---

# 💬 Discussion & Limitations
✅ Handles nonlinear mappings robustly.  
✅ Works across lighting, device, and stylistic variation.  
⚠️ Assumes **one-to-one global mapping**; fails on **local effects** (e.g., shadows, specularities).  
⚙️ Future work suggested: piecewise-consistent mappings and multi-view optimization.

---

# 🧩 Educational Connections

| Level | Concepts |
|--------|----------|
| **Undergraduate** | Least squares, affine transforms, color spaces. |
| **Graduate** | Scattered data interpolation, probabilistic weighting, in-camera color pipeline modeling. |

---

# 📚 References
- Reinhard et al. (2001), *Color Transfer Between Images*  
- Ilie & Welch (2005), *Polynomial Color Mapping*  
- Oliveira et al. (2011), *Unsupervised Local Color Correction*  
- Hwang et al. (2014), *Color Transfer Using Probabilistic Moving Least Squares*, CVPR  

---

# 🧭 Takeaway
> **Probabilistic Moving Least Squares (PMLS)** reframes color transfer as a robust, data-driven interpolation in color space —  
> blending geometric locality with probabilistic reliability to deliver precise, artifact-free photometric alignment across cameras, lighting, and styles.

---
