---
layout: post
title: "Measuring Contrast Between SDR and HDR Images: Bridging Dynamic Range Domains"
description: "Learn how to compare contrast between Standard Dynamic Range (SDR) and High Dynamic Range (HDR) images, whether they share the same content or are completely different scenes. Covers encoding differences, tone mapping, and practical comparison methods for both scenarios."
tags: [computer-vision, image-processing, contrast, hdr, sdr, tone-mapping, quality-assessment]
---

> **Note**: This post builds on [Understanding Contrast in Images: From Perception to Computation]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}), [Understanding Contrast in Color Images: Beyond Luminance]({{ "/2025/12/27/understanding-color-contrast.html" | relative_url }}), [Measuring Contrast Between Two Color Images: Comparison Metrics and Methods]({{ "/2025/12/28/measuring-contrast-between-images.html" | relative_url }}), and [Comparing Contrast Across Different Images: Content-Independent Metrics]({{ "/2025/12/29/comparing-contrast-across-different-images.html" | relative_url }}). This post addresses the unique challenges of comparing images with fundamentally different dynamic ranges.

## Table of Contents
- [1. The Fundamental Challenge](#1-the-fundamental-challenge)
  - [1.1. What Makes SDR-HDR Comparison Different?](#11-what-makes-sdr-hdr-comparison-different)
  - [1.2. Two Comparison Scenarios](#12-two-comparison-scenarios)
- [2. Understanding SDR and HDR Representations](#2-understanding-sdr-and-hdr-representations)
  - [2.1. SDR (Standard Dynamic Range)](#21-sdr-standard-dynamic-range)
  - [2.2. HDR (High Dynamic Range)](#22-hdr-high-dynamic-range)
  - [2.3. Key Differences](#23-key-differences)
- [3. Encoding and Color Space Considerations](#3-encoding-and-color-space-considerations)
  - [3.1. SDR Encoding](#31-sdr-encoding)
  - [3.2. HDR Encoding](#32-hdr-encoding)
  - [3.3. Color Gamut Differences](#33-color-gamut-differences)
- [4. Tone Mapping: The Bridge Between HDR and SDR](#4-tone-mapping-the-bridge-between-hdr-and-sdr)
  - [4.1. Global Tone Mapping Operators (TMOs)](#41-global-tone-mapping-operators-tmos)
  - [4.2. Local Tone Mapping Operators](#42-local-tone-mapping-operators)
  - [4.3. Impact on Contrast](#43-impact-on-contrast)
- [5. Scenario 1: Same Content, Different Dynamic Range](#5-scenario-1-same-content-different-dynamic-range)
  - [5.1. Preparation: Bringing Images to Common Space](#51-preparation-bringing-images-to-common-space)
  - [5.2. Direct Comparison Approaches](#52-direct-comparison-approaches)
  - [5.3. Contrast Preservation Metrics](#53-contrast-preservation-metrics)
  - [5.4. Perceptual Quality Metrics](#54-perceptual-quality-metrics)
- [6. Scenario 2: Different Content, Different Dynamic Range](#6-scenario-2-different-content-different-dynamic-range)
  - [6.1. Dynamic Range Normalization](#61-dynamic-range-normalization)
  - [6.2. Scale-Invariant Contrast Metrics](#62-scale-invariant-contrast-metrics)
  - [6.3. Distribution-Based Comparison](#63-distribution-based-comparison)
- [7. Practical Metrics for SDR-HDR Contrast Comparison](#7-practical-metrics-for-sdr-hdr-contrast-comparison)
  - [7.1. HDR-VDP (Visual Difference Predictor)](#71-hdr-vdp-visual-difference-predictor)
  - [7.2. PU-SSIM and HDR-SSIM](#72-pu-ssim-and-hdr-ssim)
  - [7.3. Custom Contrast Comparison Framework](#73-custom-contrast-comparison-framework)
- [8. Implementation Guide](#8-implementation-guide)
  - [8.1. HDR Image Loading and Linearization](#81-hdr-image-loading-and-linearization)
  - [8.2. Tone Mapping for Comparison](#82-tone-mapping-for-comparison)
  - [8.3. Contrast Comparison Pipeline](#83-contrast-comparison-pipeline)
- [9. Common Pitfalls and How to Avoid Them](#9-common-pitfalls-and-how-to-avoid-them)
- [10. Applications](#10-applications)
  - [10.1. HDR Tone Mapping Evaluation](#101-hdr-tone-mapping-evaluation)
  - [10.2. Content Mastering for Different Displays](#102-content-mastering-for-different-displays)
  - [10.3. HDR Dataset Quality Control](#103-hdr-dataset-quality-control)
- [11. Conclusion](#11-conclusion)
- [12. Further Reading](#12-further-reading)

## 1. The Fundamental Challenge

### 1.1. What Makes SDR-HDR Comparison Different?

Comparing contrast between SDR (Standard Dynamic Range) and HDR (High Dynamic Range) images is fundamentally more challenging than comparing two SDR or two HDR images because:

**Different Physical Domains**:
- **SDR**: Typically 0-100 cd/m² (nits), designed for CRT/LCD displays
- **HDR**: 0.0001-10,000+ cd/m² (or higher), representing real-world luminance

**Different Encoding**:
- **SDR**: Gamma-encoded (sRGB, Rec.709), 8-bit per channel
- **HDR**: Linear, PQ (Perceptual Quantizer), or HLG (Hybrid Log-Gamma), 10-bit or 16-bit

**Different Perceptual Intent**:
- **SDR**: Display-referred (optimized for specific display characteristics)
- **HDR**: Scene-referred (aims to represent actual scene luminance)

*Note: In practice, SDR pipelines often mix scene-referred processing with display-referred encoding, whereas HDR explicitly preserves scene luminance over a wider range. However, for most consumer SDR content, display-referred framing is accurate.*

**Incomparable Pixel Values**:
- A pixel value of 128 in SDR ≠ 128 in linear HDR ≠ 128 in PQ-encoded HDR
- Direct numerical comparison is meaningless without proper transformation

### 1.2. Two Comparison Scenarios

Just as we distinguished in the previous post, SDR-HDR comparison has two fundamental scenarios:

**Scenario 1: Same Content**
- Example: Comparing an HDR capture to its SDR tone-mapped version
- **Goal**: Evaluate how well the SDR rendition preserves the contrast characteristics of the HDR original
- **Approach**: Pixel-wise and local comparisons after bringing to common space
- **Use case**: Tone mapping quality assessment, display adaptation evaluation

**Scenario 2: Different Content**
- Example: Comparing contrast "richness" of an HDR sunset vs. SDR portrait
- **Goal**: Compare statistical contrast properties across dynamic range domains
- **Approach**: Content-independent metrics after normalization
- **Use case**: Dataset analysis, cross-domain style transfer, quality benchmarking

## 2. Understanding SDR and HDR Representations

### 2.1. SDR (Standard Dynamic Range)

**Luminance Range**: Typically 0.1 - 100 cd/m² (about 1000:1 contrast ratio)

**Encoding**: 
- **sRGB/Rec.709**: Gamma 2.2-2.4 encoding
- **8-bit**: 0-255 per channel
- **Display-referred**: Values optimized for typical displays

**Color Gamut**: Rec.709 / sRGB (narrower than human vision)

**Representation**:

$$ V_{SDR} = \begin{cases} 
12.92 \cdot L & \text{if } L \leq 0.0031308 \\
1.055 \cdot L^{1/2.4} - 0.055 & \text{if } L > 0.0031308
\end{cases} $$

where $L$ is linear luminance normalized to [0, 1].

### 2.2. HDR (High Dynamic Range)

**Luminance Range**: 0.0001 - 10,000+ cd/m² (up to 100,000,000:1 contrast ratio)

**Encoding**:
- **Linear**: Direct representation of scene luminance (e.g., OpenEXR)
- **PQ (ST 2084)**: Perceptual quantizer, optimized for 10,000 cd/m² peak
- **HLG (Hybrid Log-Gamma)**: Scene-referred, backward compatible with SDR

**Color Gamut**: Rec.2020 / DCI-P3 (wider than Rec.709)

**PQ Encoding**:

$$ V_{PQ} = \left( \frac{c_1 + c_2 \cdot L^{m_1}}{1 + c_3 \cdot L^{m_1}} \right)^{m_2} $$

where:
- $L$ = normalized linear luminance (0-1, representing 0-10,000 cd/m²)
- $m_1 = 0.1593$, $m_2 = 78.8438$
- $c_1 = 0.8359$, $c_2 = 18.8516$, $c_3 = 18.6875$

### 2.3. Key Differences

| Aspect | SDR | HDR |
|--------|-----|-----|
| **Peak Luminance** | ~100 cd/m² | 1,000 - 10,000+ cd/m² |
| **Dynamic Range** | ~6-7 stops | 14-20+ stops |
| **Bit Depth** | 8-bit | 10-bit, 16-bit, or float |
| **Encoding** | Gamma (sRGB/Rec.709) | Linear, PQ, or HLG |
| **Color Space** | Rec.709/sRGB | Rec.2020/DCI-P3 |
| **Representation** | Display-referred | Scene-referred |
| **Clipping** | Highlights and shadows clipped | Preserves full scene range |

**Critical Insight**: Because of these differences, **direct pixel value comparison is meaningless**. We must transform images to a common space.

## 3. Encoding and Color Space Considerations

### 3.1. SDR Encoding

**sRGB Transfer Function** (display-referred):
- Converts linear RGB to gamma-encoded values
- Designed for CRT displays (approximately gamma 2.2)
- Perceptually uniform in the SDR range

**Luminance Calculation** (for grayscale/contrast analysis):

$$ Y_{709} = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B $$

where R, G, B are **linear** values (after inverse gamma).

**Note on Color Space**: For simplicity and widespread practice, we use Rec.709 luminance coefficients throughout this post. When working with Rec.2020 HDR content, you may substitute Rec.2020 coefficients ($Y_{2020} = 0.2627 \cdot R + 0.6780 \cdot G + 0.0593 \cdot B$) for stricter colorimetric accuracy. The difference is typically small for contrast analysis but matters for precise photometry.

### 3.2. HDR Encoding

**PQ (Perceptual Quantizer) - ST 2084**:
- Perceptually uniform across full HDR range
- 10-bit PQ can represent 0.0001 to 10,000 cd/m²
- More bits allocated to perceptually important mid-tones

**HLG (Hybrid Log-Gamma) - ITU-R BT.2100**:
- Combines gamma and logarithmic curves
- Backward compatible with SDR displays
- Scene-referred (relative luminance)

**Linear (OpenEXR, RGBE)**:
- Direct representation of scene luminance
- Floating-point values (no quantization beyond float precision)
- Ideal for image processing, but requires tone mapping for display

### 3.3. Color Gamut Differences

**Rec.709 (SDR)** covers ~35% of CIE 1931 color space
**Rec.2020 (HDR)** covers ~75% of CIE 1931 color space

**Important**: Wide-gamut HDR colors may be **out of gamut** for SDR. This affects:
- Chromatic contrast (HDR may have higher color saturation)
- Gamut mapping introduces additional contrast changes
- Some HDR details may be lost in SDR conversion

## 4. Tone Mapping: The Bridge Between HDR and SDR

To compare contrast between HDR and SDR, we often need to **tone map** the HDR image to SDR range. The choice of tone mapping operator (TMO) dramatically affects measured contrast.

### 4.1. Global Tone Mapping Operators (TMOs)

Global TMOs apply the same transformation to all pixels based on global statistics.

**Reinhard Global**:

$$ L_d = \frac{L}{1 + L} $$

where $L$ is normalized linear luminance.

**Properties**:
- Compresses dynamic range uniformly
- Reduces global contrast
- Preserves local contrast relatively well for moderate compression
- Simple and fast

**Exponential**:

$$ L_d = 1 - e^{-\alpha \cdot L} $$

where $\alpha$ controls compression strength.

**Drago Logarithmic**:

$$ L_d = \frac{\log(1 + L \cdot L_{max}^p)}{\log(1 + L_{max}^p) \cdot \log(b)} $$

where $L_{max}$ is max luminance, $p$ and $b$ are parameters.

### 4.2. Local Tone Mapping Operators

Local TMOs adapt compression based on local image content.

**Bilateral Filtering-Based** (Durand & Dorsey):
- Separates base layer (low-frequency) and detail layer (high-frequency)
- Compresses base, preserves detail
- **Preserves local contrast** well
- Can introduce halos

**Gradient Domain** (Fattal et al.):
- Operates in gradient domain to preserve local gradients
- Excellent local contrast preservation
- Computationally expensive

**Properties**:
- **Better local contrast preservation** than global TMOs
- Can introduce artifacts (halos, gradient reversals)
- Computationally more expensive

### 4.3. Impact on Contrast

**Global TMOs**:
- ✅ Preserve relative luminance relationships
- ✅ No spatial artifacts
- ❌ Reduce global contrast significantly
- ❌ May compress local contrast in high-contrast regions

**Local TMOs**:
- ✅ Preserve local contrast better
- ✅ Maintain detail in shadows and highlights
- ❌ Can introduce halos (false local contrast)
- ❌ May violate global luminance relationships

**Critical for Comparison**: 
- State which TMO was used
- Different TMOs yield different contrast characteristics
- For fair comparison, use multiple TMOs or a standardized one

## 5. Scenario 1: Same Content, Different Dynamic Range

**Example**: Comparing an HDR photo to its SDR version, or evaluating a tone mapping algorithm.

### 5.1. Preparation: Bringing Images to Common Space

**Option A: HDR → SDR (Tone Mapping)**

```python
# Apply tone mapping to HDR image
sdr_from_hdr = tone_map(hdr_linear, method='reinhard')
# Now compare: sdr_from_hdr vs sdr_original
```

**Pros**: 
- Works with standard SDR image processing tools
- Easier to visualize
- Aligns with human perception of displayed SDR image

**Cons**:
- Choice of TMO affects all subsequent metrics
- May lose information present in HDR

**Option B: SDR → HDR (Inverse Tone Mapping)**

```python
# Linearize SDR and expand to HDR range
hdr_from_sdr = expand_to_hdr(linearize_sdr(sdr_image))
# Now compare: hdr_original vs hdr_from_sdr
```

**Pros**:
- Preserves HDR information in original HDR image
- Can analyze contrast in linear space

**Cons**:
- Inverse tone mapping is ill-posed (information loss)
- Cannot recover clipped highlights/shadows from SDR
- Requires assumptions about SDR creation process

**Recommended Approach**: **Option A (HDR → SDR)** for most use cases, especially when evaluating tone mapping quality.

### 5.2. Direct Comparison Approaches

Once in common space (typically SDR after tone mapping):

**Important Caveat**: RMS contrast and similar global metrics should be treated as **coarse statistical descriptors**, not perceptual contrast measures. They provide useful quantitative comparisons but do not capture human contrast perception. For perceptual assessment, use HDR-VDP, PU-SSIM, or other HVS-based metrics (Section 7).

**Luminance Contrast Difference** (global):

$$ \Delta C_{RMS} = \left| \frac{\sigma_{HDR \to SDR}}{\mu_{HDR \to SDR}} - \frac{\sigma_{SDR}}{\mu_{SDR}} \right| $$

**Local Contrast Preservation**:

Compute local contrast for both images, then measure difference:

$$ C_{local}(x,y) = \frac{\sigma_w(x,y)}{\mu_w(x,y)} $$

$$ \Delta C_{local} = \frac{1}{N} \sum_{x,y} \left| C_{local}^{HDR}(x,y) - C_{local}^{SDR}(x,y) \right| $$

**Spatial Frequency Preservation**:

Compare power spectra to see if high-frequency detail (local contrast) is preserved:

$$ \text{Frequency Preservation} = \frac{P_{high}^{HDR \to SDR}}{P_{high}^{SDR}} $$

**Note**: Results depend on frequency band definitions (cutoff frequency, linear vs. log spectrum, pre- or post-gamma) and should be interpreted comparatively, not absolutely. Standardize the analysis pipeline when comparing across different tone mappings.

### 5.3. Contrast Preservation Metrics

**Contrast Preservation Index** (custom metric):

$$ CPI = 1 - \frac{\Delta C_{local}}{\max(C_{local}^{HDR}, C_{local}^{SDR})} $$

Range: [0, 1], where 1 = perfect preservation

**Note**: CPI is a heuristic engineering metric and should not be interpreted as perceptual fidelity. It is sensitive to window size and undefined when both contrasts are near zero. Use it for relative comparison within a controlled experimental setup, not as an absolute quality indicator.

**Multi-Scale Contrast Fidelity**:

Compute RMS contrast at multiple scales (using Gaussian pyramids), then aggregate:

$$ MCF = \prod_{s=1}^{S} \left( 1 - \frac{|C_s^{HDR} - C_s^{SDR}|}{C_s^{HDR} + C_s^{SDR}} \right) $$

### 5.4. Perceptual Quality Metrics

**SSIM in Luminance Channel**:

After tone mapping, compute SSIM on luminance:

$$ SSIM(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} $$

The **contrast component** of SSIM specifically measures contrast similarity:

$$ C(x, y) = \frac{2\sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2} $$

**Caveat**: SSIM's contrast component measures local variance similarity, not perceptual contrast preservation. It does not fully capture HVS contrast fidelity, especially across large luminance differences.

**HDR-VDP** (Visual Difference Predictor):
- Perceptually-based metric designed for HDR
- Accounts for viewing conditions and display characteristics
- Predicts visible differences between HDR and SDR
- Provides spatial map of contrast differences

## 6. Scenario 2: Different Content, Different Dynamic Range

**Example**: Comparing contrast characteristics of an HDR sunset vs. SDR cityscape for dataset balancing.

### 6.1. Dynamic Range Normalization

**Challenge**: HDR images can have values spanning 0.0001 to 10,000 cd/m², while SDR is 0-255 (or 0-1 normalized). We need scale-invariant metrics.

**Approach 1: Percentile-Based Normalization**

```python
# For both HDR and SDR, normalize to [0, 1] using percentiles
L_hdr_norm = (L_hdr - L_hdr_p01) / (L_hdr_p99 - L_hdr_p01)
L_sdr_norm = (L_sdr - L_sdr_p01) / (L_sdr_p99 - L_sdr_p01)

# Now compute contrast metrics on normalized images
C_RMS_hdr = std(L_hdr_norm) / mean(L_hdr_norm)
C_RMS_sdr = std(L_sdr_norm) / mean(L_sdr_norm)
```

**Approach 2: Log-Domain Comparison**

HDR images are naturally suited to log-domain representation (this is why HLG uses log encoding):

```python
# Convert to log domain (handles HDR range gracefully)
L_hdr_log = np.log10(L_hdr + epsilon)
L_sdr_log = np.log10(linearize_sdr(sdr_image) + epsilon)

# Compute contrast in log domain
# Note: In log-luminance space, absolute dispersion (std) is often more
# interpretable than relative dispersion (std/mean), as mean can approach
# zero or change sign in log space
C_RMS_hdr_log = std(L_hdr_log) / mean(L_hdr_log)
C_RMS_sdr_log = std(L_sdr_log) / mean(L_sdr_log)

# Alternative: Use absolute dispersion
contrast_hdr_log = std(L_hdr_log)  # More robust in log space
contrast_sdr_log = std(L_sdr_log)
```

### 6.2. Scale-Invariant Contrast Metrics

**Coefficient of Variation** (already scale-invariant):

$$ C_{RMS} = \frac{\sigma_L}{\mu_L} $$

This is naturally scale-invariant and works for both HDR and SDR after proper linearization.

**Dynamic Range Utilization** (normalized):

**Physical DR Utilization** (for HDR in absolute luminance units):

$$ U_{DR}^{physical} = \frac{L_{p99} - L_{p01}}{L_{max\_possible} - L_{min\_possible}} $$

For HDR: $L_{max\_possible} = 10000$ cd/m², $L_{min\_possible} = 0.0001$ cd/m²

**Statistical DR Utilization** (after normalization):

$$ U_{DR}^{statistical} = \frac{L_{p99} - L_{p01}}{L_{max\_image} - L_{min\_image}} $$

For normalized images: Both HDR and SDR are scaled to [0, 1]

**Important**: These measure different things:
- **Physical**: "This HDR image uses 80% of the 10,000 cd/m² range" (absolute meaning)
- **Statistical**: "This image uses 80% of its own value range" (relative meaning)

For cross-domain comparison, use statistical DR utilization after proper normalization.

### 6.3. Distribution-Based Comparison

**Histogram Comparison in Log-Luminance Space**:

```python
# Convert both to log-luminance
log_hdr = np.log10(hdr_linear + 1e-4)
log_sdr = np.log10(linearize_sdr(sdr_image) + 1e-4)

# CRITICAL: Use shared bin range for meaningful comparison
# Option 1: Use absolute log-luminance range (requires similar capture conditions)
log_min = min(log_hdr.min(), log_sdr.min())
log_max = max(log_hdr.max(), log_sdr.max())

hist_hdr, _ = np.histogram(log_hdr, bins=256, range=(log_min, log_max), density=True)
hist_sdr, _ = np.histogram(log_sdr, bins=256, range=(log_min, log_max), density=True)

# Option 2: Normalize to same range (for statistical comparison)
log_hdr_norm = (log_hdr - log_hdr.min()) / (log_hdr.max() - log_hdr.min())
log_sdr_norm = (log_sdr - log_sdr.min()) / (log_sdr.max() - log_sdr.min())

# Use the SAME normalized range [0, 1] for both
hist_hdr_norm, _ = np.histogram(log_hdr_norm, bins=256, range=(0, 1), density=True)
hist_sdr_norm, _ = np.histogram(log_sdr_norm, bins=256, range=(0, 1), density=True)

similarity = compare_histograms(hist_hdr, hist_sdr)
```

**Entropy Comparison**:

Entropy in log-luminance space:

$$ H = -\sum_{i} p_i \log_2(p_i) $$

**Note**: Log-luminance reduces scale sensitivity, but entropy remains dependent on binning and normalization. It is not strictly scale-invariant—histogram bin edges and quantization affect the entropy value. Use consistent binning across images for meaningful comparison.

**Multi-Scale Contrast Features**:

Compute contrast at multiple scales (Laplacian pyramid), normalized by local mean:

$$ \mathbf{f}_{contrast} = [C_{scale1}, C_{scale2}, ..., C_{scaleN}] $$

Then compare feature vectors using cosine similarity or Euclidean distance.

## 7. Practical Metrics for SDR-HDR Contrast Comparison

### 7.1. HDR-VDP (Visual Difference Predictor)

**Description**: State-of-the-art perceptual metric for HDR images, accounts for viewing conditions and HVS characteristics.

**How it works**:
- Models Contrast Sensitivity Function (CSF) across luminance range
- Decomposes images into spatial frequency bands
- Predicts visibility of differences at each frequency and luminance level
- Outputs probability of detection (Q-score)

**For contrast comparison**:
- HDR-VDP-2 provides per-pixel contrast difference maps
- Can isolate contrast distortions from other artifacts
- Requires specification of viewing conditions (distance, display peak luminance)

**Usage**:
```python
# Pseudo-code (requires HDR-VDP implementation)
q_score, difference_map = hdr_vdp(hdr_image, sdr_tone_mapped, 
                                   pixels_per_degree=30,
                                   peak_luminance_sdr=100,
                                   peak_luminance_hdr=1000)
```

**Perceptual Grounding: Contrast Sensitivity Function**

A key reason HDR-VDP and perceptual metrics outperform simple numerical comparisons is their grounding in the **Contrast Sensitivity Function (CSF)**:

**The CSF Principle**: Humans do not perceive contrast uniformly across spatial frequencies and luminance levels. Specifically:

- **Spatial frequency dependence**: We're most sensitive to mid-frequencies (~4-8 cycles/degree), less to very low or very high frequencies
- **Luminance dependence**: Contrast sensitivity changes with adaptation luminance—higher in mesopic/photopic ranges, lower in scotopic
- **Masking effects**: High-contrast regions can mask nearby low-contrast details

**Implications for HDR-SDR Comparison**:

1. **PQ Encoding** is designed to match perceptual uniformity across the HDR luminance range, mimicking CSF behavior
2. **HDR-VDP** explicitly models CSF, weighting contrast differences by their perceptual visibility
3. **Simple metrics like RMS contrast** treat all frequencies equally—a 10% contrast difference in high frequencies may be imperceptible, while the same difference at 4 cpd is obvious

This is why HDR-VDP correlates better with human judgment than PSNR or raw contrast differences—it asks "can a human *see* this contrast difference?" rather than just "is it numerically different?"

**Display Dependency Critical Note**:

HDR perception fundamentally depends on:
- **Peak luminance** of the display (100 cd/m² vs 1000 cd/m² changes appearance dramatically)
- **Viewing distance** (affects spatial frequency as seen by the eye)
- **Ambient lighting** (affects adaptation state)

All perceptual metrics (HDR-VDP, PU-SSIM) require these parameters. Reporting results without specifying viewing conditions renders them non-reproducible.

### 7.2. PU-SSIM and HDR-SSIM

**PU-SSIM (Perceptually Uniform SSIM)**:
- Converts images to perceptually uniform space (PU encoding)
- Applies SSIM in PU space
- Better correlation with human perception for HDR content

**HDR-SSIM**:
- Extension of SSIM for HDR images
- Uses log-luminance domain
- Weights by local adaptation luminance

**Formula** (simplified):

$$ L_{PU} = L^{0.2} $$

Then apply standard SSIM on $L_{PU}$.

### 7.3. Custom Contrast Comparison Framework

**Multi-Domain Contrast Analysis**:

```python
def compare_sdr_hdr_contrast(hdr_linear, sdr_image, method='comprehensive'):
    """
    Compare contrast between HDR and SDR images.
    
    Returns:
        dict: Comprehensive contrast comparison metrics
    """
    results = {}
    
    # 1. Tone map HDR to SDR range
    sdr_from_hdr = tone_map(hdr_linear, method='reinhard')
    
    # 2. Global contrast comparison (after tone mapping)
    results['global_contrast'] = {
        'C_RMS_hdr_tm': compute_rms_contrast(sdr_from_hdr),
        'C_RMS_sdr': compute_rms_contrast(sdr_image),
        'delta': abs(compute_rms_contrast(sdr_from_hdr) - compute_rms_contrast(sdr_image))
    }
    
    # 3. Local contrast preservation
    local_hdr = compute_local_contrast(sdr_from_hdr)
    local_sdr = compute_local_contrast(sdr_image)
    results['local_contrast_preservation'] = np.mean(np.abs(local_hdr - local_sdr))
    
    # 4. Multi-scale contrast
    results['multiscale'] = compare_multiscale_contrast(sdr_from_hdr, sdr_image)
    
    # 5. Dynamic range utilization
    results['DR_utilization'] = {
        'hdr_original': compute_DR_utilization(hdr_linear, hdr_range=True),
        'sdr': compute_DR_utilization(sdr_image, hdr_range=False)
    }
    
    # 6. Perceptual quality (SSIM contrast component)
    ssim_val, ssim_components = compute_ssim(sdr_from_hdr, sdr_image, full=True)
    results['ssim_contrast'] = ssim_components['contrast']
    
    return results
```

## 8. Implementation Guide

### 8.1. HDR Image Loading and Linearization

```python
import numpy as np
import cv2
import imageio

def load_hdr_image(filepath):
    """
    Load HDR image and return linear luminance values.
    Supports: .hdr, .exr formats
    """
    if filepath.endswith('.exr'):
        # OpenEXR format (linear, float)
        import OpenEXR
        import Imath
        file = OpenEXR.InputFile(filepath)
        header = file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        # Read RGB channels
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ['R', 'G', 'B']
        rgb = [np.frombuffer(file.channel(c, FLOAT), dtype=np.float32) 
               for c in channels]
        rgb = [c.reshape(size[1], size[0]) for c in rgb]
        hdr_linear = np.stack(rgb, axis=-1)
        
    elif filepath.endswith('.hdr'):
        # Radiance HDR format (RGBE)
        hdr_linear = imageio.imread(filepath, format='HDR-FI')
        
    else:
        raise ValueError(f"Unsupported HDR format: {filepath}")
    
    # Convert to luminance (Rec.709)
    luminance = 0.2126 * hdr_linear[..., 0] + \
                0.7152 * hdr_linear[..., 1] + \
                0.0722 * hdr_linear[..., 2]
    
    return hdr_linear, luminance

def linearize_sdr(sdr_image):
    """
    Convert sRGB/Rec.709 SDR image to linear RGB.
    
    Args:
        sdr_image: numpy array, shape (H, W, 3), dtype uint8 or float [0, 1]
    
    Returns:
        linear_rgb: numpy array, shape (H, W, 3), dtype float
    """
    # Normalize to [0, 1] if uint8
    if sdr_image.dtype == np.uint8:
        sdr_norm = sdr_image.astype(np.float32) / 255.0
    else:
        sdr_norm = sdr_image.astype(np.float32)
    
    # Apply inverse sRGB gamma
    linear = np.where(
        sdr_norm <= 0.04045,
        sdr_norm / 12.92,
        np.power((sdr_norm + 0.055) / 1.055, 2.4)
    )
    
    return linear
```

### 8.2. Tone Mapping for Comparison

```python
def tone_map_reinhard(hdr_linear, key=0.18, saturation=1.0):
    """
    Reinhard global tone mapping operator.
    
    Args:
        hdr_linear: Linear HDR image, shape (H, W, 3)
        key: Controls overall brightness (default 0.18 for middle gray)
        saturation: Color saturation (1.0 = preserve, <1.0 = desaturate)
    
    Returns:
        sdr_gamma: Gamma-encoded SDR image [0, 1]
    """
    # Convert to luminance
    lum = 0.2126 * hdr_linear[..., 0] + \
          0.7152 * hdr_linear[..., 1] + \
          0.0722 * hdr_linear[..., 2]
    
    # Avoid log(0)
    lum = np.maximum(lum, 1e-8)
    
    # Compute log-average luminance
    log_lum_avg = np.exp(np.mean(np.log(lum + 1e-8)))
    
    # Scale luminance
    lum_scaled = (key / log_lum_avg) * lum
    
    # Tone map
    lum_display = lum_scaled / (1.0 + lum_scaled)
    
    # Apply to RGB channels
    rgb_display = hdr_linear * (lum_display / (lum + 1e-8))[..., np.newaxis]
    
    # Saturation control
    lum_display_rgb = lum_display[..., np.newaxis]
    rgb_display = lum_display_rgb + saturation * (rgb_display - lum_display_rgb)
    
    # Clamp and apply gamma
    rgb_display = np.clip(rgb_display, 0, 1)
    sdr_gamma = np.where(
        rgb_display <= 0.0031308,
        12.92 * rgb_display,
        1.055 * np.power(rgb_display, 1.0/2.4) - 0.055
    )
    
    return sdr_gamma

def tone_map_drago(hdr_linear, ldmax=100, bias=0.85):
    """
    Drago logarithmic tone mapping (adaptive logarithmic mapping).
    
    Args:
        hdr_linear: Linear HDR image
        ldmax: Maximum display luminance
        bias: Controls contrast (0.5-1.0, higher = more contrast)
    """
    lum = 0.2126 * hdr_linear[..., 0] + \
          0.7152 * hdr_linear[..., 1] + \
          0.0722 * hdr_linear[..., 2]
    
    lum = np.maximum(lum, 1e-8)
    lw_max = np.max(lum)
    lw_avg = np.exp(np.mean(np.log(lum)))
    
    # Adaptive log base
    log_base = np.log10(bias)
    
    # Drago formula
    lum_display = (np.log10(1 + lum / lw_avg) / np.log10(1 + lw_max / lw_avg)) / log_base
    
    # Apply to RGB
    rgb_display = hdr_linear * (lum_display / (lum + 1e-8))[..., np.newaxis]
    rgb_display = np.clip(rgb_display, 0, 1)
    
    # Gamma encode
    sdr_gamma = np.power(rgb_display, 1.0/2.2)
    
    return sdr_gamma
```

### 8.3. Contrast Comparison Pipeline

```python
def compare_sdr_hdr_contrast(hdr_path, sdr_path, tone_map_method='reinhard'):
    """
    Complete pipeline for comparing contrast between HDR and SDR images.
    """
    # 1. Load images
    hdr_linear, hdr_lum = load_hdr_image(hdr_path)
    sdr_image = cv2.imread(sdr_path)
    sdr_image = cv2.cvtColor(sdr_image, cv2.COLOR_BGR2RGB) / 255.0
    
    # 2. Linearize SDR
    sdr_linear = linearize_sdr(sdr_image)
    sdr_lum = 0.2126 * sdr_linear[..., 0] + \
              0.7152 * sdr_linear[..., 1] + \
              0.0722 * sdr_linear[..., 2]
    
    # 3. Tone map HDR to SDR
    if tone_map_method == 'reinhard':
        sdr_from_hdr = tone_map_reinhard(hdr_linear)
    elif tone_map_method == 'drago':
        sdr_from_hdr = tone_map_drago(hdr_linear)
    else:
        raise ValueError(f"Unknown tone map method: {tone_map_method}")
    
    # Linearize tone-mapped result for fair comparison
    sdr_from_hdr_linear = linearize_sdr(sdr_from_hdr)
    sdr_from_hdr_lum = 0.2126 * sdr_from_hdr_linear[..., 0] + \
                       0.7152 * sdr_from_hdr_linear[..., 1] + \
                       0.0722 * sdr_from_hdr_linear[..., 2]
    
    # 4. Compute contrast metrics
    results = {}
    
    # Global RMS contrast
    results['C_RMS_hdr_tm'] = np.std(sdr_from_hdr_lum) / (np.mean(sdr_from_hdr_lum) + 1e-8)
    results['C_RMS_sdr'] = np.std(sdr_lum) / (np.mean(sdr_lum) + 1e-8)
    results['delta_C_RMS'] = abs(results['C_RMS_hdr_tm'] - results['C_RMS_sdr'])
    
    # Dynamic range utilization
    # Statistical DR utilization: how much of the image's own range is used
    results['DR_hdr_original'] = (np.percentile(hdr_lum, 99) - np.percentile(hdr_lum, 1)) / \
                                 (np.max(hdr_lum) - np.min(hdr_lum) + 1e-8)
    results['DR_sdr'] = (np.percentile(sdr_lum, 99) - np.percentile(sdr_lum, 1)) / \
                        (np.max(sdr_lum) - np.min(sdr_lum) + 1e-8)
    
    # Local contrast comparison
    from scipy.ndimage import uniform_filter, generic_filter
    
    window_size = 11
    local_mean_hdr = uniform_filter(sdr_from_hdr_lum, size=window_size)
    local_std_hdr = generic_filter(sdr_from_hdr_lum, np.std, size=window_size)
    local_contrast_hdr = local_std_hdr / (local_mean_hdr + 1e-8)
    
    local_mean_sdr = uniform_filter(sdr_lum, size=window_size)
    local_std_sdr = generic_filter(sdr_lum, np.std, size=window_size)
    local_contrast_sdr = local_std_sdr / (local_mean_sdr + 1e-8)
    
    results['local_contrast_diff'] = np.mean(np.abs(local_contrast_hdr - local_contrast_sdr))
    results['local_contrast_corr'] = np.corrcoef(local_contrast_hdr.ravel(), 
                                                  local_contrast_sdr.ravel())[0, 1]
    
    # SSIM (on gamma-encoded for display simulation)
    from skimage.metrics import structural_similarity as ssim
    ssim_val = ssim(sdr_from_hdr, sdr_image, channel_axis=-1)
    results['SSIM'] = ssim_val
    
    # Histogram entropy comparison (in log-luminance)
    log_hdr = np.log10(hdr_lum + 1e-4)
    log_sdr = np.log10(sdr_lum + 1e-4)
    
    # CRITICAL: Use shared bin range for meaningful comparison
    log_min = min(log_hdr.min(), log_sdr.min())
    log_max = max(log_hdr.max(), log_sdr.max())
    
    hist_hdr, _ = np.histogram(log_hdr, bins=256, range=(log_min, log_max), density=True)
    hist_sdr, _ = np.histogram(log_sdr, bins=256, range=(log_min, log_max), density=True)
    
    hist_hdr = hist_hdr[hist_hdr > 0]
    hist_sdr = hist_sdr[hist_sdr > 0]
    
    results['entropy_hdr'] = -np.sum(hist_hdr * np.log2(hist_hdr))
    results['entropy_sdr'] = -np.sum(hist_sdr * np.log2(hist_sdr))
    
    return results

# Example usage:
# results = compare_sdr_hdr_contrast('scene.exr', 'scene_sdr.jpg', 'reinhard')
# print(f"RMS Contrast difference: {results['delta_C_RMS']:.3f}")
# print(f"Local contrast preservation: {results['local_contrast_corr']:.3f}")
# print(f"SSIM: {results['SSIM']:.3f}")
```

## 9. Common Pitfalls and How to Avoid Them

### Quick Reference: What NOT to Use for SDR-HDR Contrast Comparison

| Metric                | Use for SDR-HDR contrast? | Why / Why not                                                      |
|-----------------------|---------------------------|--------------------------------------------------------------------|
| **PSNR**              | ❌                        | Absolute error meaningless across different dynamic ranges         |
| **Raw pixel diff**    | ❌                        | Pixel values in different encodings are incomparable               |
| **Raw histogram**     | ❌                        | Encoding-dependent; bin ranges differ                              |
| **RMS (gamma-encoded)** | ❌                      | Must linearize first; gamma space distorts photometric relationships |
| **RMS (linear)**      | ⚠️                        | Needs normalization; OK after tone mapping to common space         |
| **SSIM (RGB)**        | ⚠️                        | Better than PSNR, but not HDR-aware; use on tone-mapped images    |
| **SSIM contrast term** | ⚠️                       | Local variance similarity ≠ perceptual contrast fidelity           |
| **Histogram entropy** | ⚠️                        | Sensitive to binning; requires shared bin edges and log-luminance  |
| **HDR-VDP**           | ✅                        | Perceptual, CSF-based, designed for HDR                            |
| **PU-SSIM / HDR-SSIM** | ✅                       | Perceptually uniform encoding before comparison                    |
| **Local contrast (linear)** | ✅                  | Valid after tone mapping to common space                           |
| **Log-luminance metrics** | ✅                    | Scale-invariant, suitable for cross-domain comparison              |

**Key Principle**: Always ask "Are these values in the same perceptual/photometric space?" before comparing.

### Pitfall 1: Comparing Raw Pixel Values

**❌ Wrong**:
```python
diff = hdr_image - sdr_image  # Meaningless!
```

**✅ Correct**:
```python
# Bring to common space first
sdr_from_hdr = tone_map(hdr_image)
diff = sdr_from_hdr - sdr_image  # Now meaningful
```

### Pitfall 2: Ignoring Encoding

**❌ Wrong**:
```python
C_RMS_sdr = np.std(sdr_gamma) / np.mean(sdr_gamma)  # Wrong! Gamma-encoded
```

**✅ Correct**:
```python
sdr_linear = linearize_sdr(sdr_gamma)
C_RMS_sdr = np.std(sdr_linear) / np.mean(sdr_linear)  # Correct!
```

### Pitfall 3: Not Specifying Tone Mapping Operator

**❌ Wrong**:
"The HDR image has higher contrast than SDR."

**✅ Correct**:
"After Reinhard tone mapping, the HDR image exhibits 15% higher local contrast than the native SDR image."

### Pitfall 4: Comparing Different Color Gamuts Directly

**❌ Wrong**:
```python
chromatic_contrast_hdr = compute_chromatic_contrast(hdr_rec2020)
chromatic_contrast_sdr = compute_chromatic_contrast(sdr_rec709)
diff = chromatic_contrast_hdr - chromatic_contrast_sdr  # Not comparable!
```

**✅ Correct**:
```python
# Convert to same color space (e.g., Rec.709) first
hdr_rec709 = convert_color_space(hdr_rec2020, 'Rec.2020', 'Rec.709')
# Now compare
```

### Pitfall 5: Assuming Linear Relationship

**❌ Wrong Assumption**:
"If HDR luminance is 10x higher, contrast should be 10x higher."

**✅ Correct Understanding**:
Contrast is **relative** (standard deviation / mean). The absolute luminance range doesn't determine contrast. A low-key image (dark overall) can have high contrast.

## 10. Applications

### 10.1. HDR Tone Mapping Evaluation

**Goal**: Determine which tone mapping operator best preserves contrast from HDR original.

**Approach**:
```python
tmos = ['reinhard', 'drago', 'durand', 'mantiuk']
results = {}

for tmo in tmos:
    sdr = tone_map(hdr_original, method=tmo)
    results[tmo] = {
        'local_contrast_preservation': compute_local_contrast_correlation(hdr_original, sdr),
        'global_contrast': compute_rms_contrast(sdr),
        'perceptual_quality': compute_hdr_vdp(hdr_original, sdr)
    }

# Select TMO with highest contrast preservation
best_tmo = max(results, key=lambda x: results[x]['local_contrast_preservation'])
```

### 10.2. Content Mastering for Different Displays

**Goal**: Create SDR master from HDR source that maintains contrast appearance.

**Approach**:
1. Analyze contrast distribution in HDR original
2. Apply tone mapping
3. Measure contrast preservation across different luminance regions
4. Adjust TMO parameters to maximize contrast preservation in perceptually important regions

```python
def adaptive_tone_mapping_for_contrast(hdr, regions_of_interest=None):
    # Identify important contrast regions
    if regions_of_interest is None:
        regions_of_interest = detect_high_contrast_regions(hdr)
    
    # Optimize TMO parameters to preserve contrast in ROIs
    best_params = optimize_tmo_for_contrast_preservation(hdr, regions_of_interest)
    
    sdr_master = tone_map(hdr, **best_params)
    return sdr_master
```

### 10.3. HDR Dataset Quality Control

**Goal**: Ensure consistent contrast characteristics across mixed HDR/SDR dataset.

**Approach**:
```python
dataset_stats = []

for image_path in dataset:
    if is_hdr(image_path):
        hdr = load_hdr(image_path)
        # Compute contrast in log-luminance (scale-invariant)
        log_lum = np.log10(get_luminance(hdr) + 1e-4)
        contrast = np.std(log_lum) / np.mean(log_lum)
        stats = {'path': image_path, 'type': 'HDR', 'contrast': contrast}
    else:
        sdr = load_sdr(image_path)
        sdr_linear = linearize_sdr(sdr)
        log_lum = np.log10(get_luminance(sdr_linear) + 1e-4)
        contrast = np.std(log_lum) / np.mean(log_lum)
        stats = {'path': image_path, 'type': 'SDR', 'contrast': contrast}
    
    dataset_stats.append(stats)

# Identify outliers
import pandas as pd
df = pd.DataFrame(dataset_stats)
outliers = df[np.abs(df['contrast'] - df['contrast'].mean()) > 2 * df['contrast'].std()]
```

## 11. Conclusion

Measuring contrast between SDR and HDR images requires careful attention to:

**Fundamental Principles**:
1. **Never compare raw pixel values** directly—always transform to common space
2. **Specify the tone mapping operator** used (or use multiple for robustness)
3. **Work in linear luminance space** for photometrically accurate metrics
4. **Use scale-invariant metrics** for cross-domain comparison
5. **Account for perceptual nonlinearities** (use log-luminance or PU encoding)

**Key Distinctions**:

| Aspect | Same Content | Different Content |
|--------|--------------|-------------------|
| **Goal** | Evaluate tone mapping quality | Compare statistical properties |
| **Common Space** | Tone map HDR → SDR | Normalize both to same range |
| **Metrics** | Local contrast preservation, SSIM | RMS contrast, entropy, distribution |
| **Pixel Correspondence** | Meaningful (same scene) | Meaningless (different scenes) |
| **Applications** | TMO evaluation, quality assessment | Dataset balancing, retrieval |

**Recommended Workflow**:

1. **Load and linearize** both images properly
2. **Choose comparison scenario** (same content vs. different content)
3. **Transform to common space** (usually HDR → SDR via tone mapping)
4. **Compute metrics** in linear luminance domain
5. **Report methodology** (TMO, color space, normalization)

By following these principles, you can perform meaningful contrast comparisons between SDR and HDR images, enabling applications from tone mapping evaluation to cross-domain dataset analysis.

## 12. Further Reading

### Related Posts in This Series:
- [Understanding Contrast in Images: From Perception to Computation]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }})
- [Understanding Contrast in Color Images: Beyond Luminance]({{ "/2025/12/27/understanding-color-contrast.html" | relative_url }})
- [Measuring Contrast Between Two Color Images: Comparison Metrics and Methods]({{ "/2025/12/28/measuring-contrast-between-images.html" | relative_url }})
- [Comparing Contrast Across Different Images: Content-Independent Metrics]({{ "/2025/12/29/comparing-contrast-across-different-images.html" | relative_url }})

### HDR and Tone Mapping:
- Reinhard, E., et al. (2010). *High Dynamic Range Imaging: Acquisition, Display, and Image-Based Lighting* (2nd ed.). Morgan Kaufmann.
- Mantiuk, R., et al. (2015). "HDR-VDP-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions." *ACM TOG*, 30(4).
- Durand, F., & Dorsey, J. (2002). "Fast bilateral filtering for the display of high-dynamic-range images." *ACM TOG*, 21(3), 257-266.

### Perceptual Metrics:
- Narwaria, M., et al. (2015). "HDR-VDP-2.2: A calibrated method for objective quality prediction of high-dynamic range and standard images." *Journal of Electronic Imaging*, 24(1).
- Aydin, T. O., et al. (2008). "Dynamic range independent image quality assessment." *ACM TOG*, 27(3).

### Standards:
- ITU-R BT.2100: "Image parameter values for high dynamic range television for use in production and international programme exchange"
- SMPTE ST 2084: "High Dynamic Range Electro-Optical Transfer Function of Mastering Reference Displays"

