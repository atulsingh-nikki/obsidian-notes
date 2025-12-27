---
layout: post
title: "Comparing Contrast Across Different Images: Content-Independent Metrics"
description: "Learn how to compare contrast characteristics between completely different images with unrelated content, using content-independent metrics for dataset analysis, image retrieval, quality normalization, and photography workflows."
tags: [computer-vision, image-processing, contrast, metrics, dataset-analysis, image-retrieval]
---

When comparing contrast between two images, there's a fundamental distinction that's often overlooked: **Are we comparing different versions of the same scene, or are we comparing two completely different images with unrelated content?**

Most contrast comparison metrics (like those used in image enhancement evaluation or compression assessment) assume we're comparing variations of the **same image**—the pixel correspondence is meaningful, and we can compute pixel-wise or local differences. But what if we want to compare the contrast characteristics of a sunset photo against a portrait, or a cityscape against a forest scene?

> **Note**: This post builds on [Understanding Contrast in Images: From Perception to Computation]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}), [Understanding Contrast in Color Images: Beyond Luminance]({{ "/2025/12/27/understanding-color-contrast.html" | relative_url }}), and [Measuring Contrast Between Two Color Images: Comparison Metrics and Methods]({{ "/2025/12/28/measuring-contrast-between-images.html" | relative_url }}). The key difference here is that we're dealing with **content-independent** contrast comparison.

## Table of Contents
- [1. The Fundamental Difference](#1-the-fundamental-difference)
  - [1.1. Same-Content Comparison](#11-same-content-comparison)
  - [1.2. Different-Content Comparison](#12-different-content-comparison)
- [2. Why Compare Contrast Across Different Images?](#2-why-compare-contrast-across-different-images)
- [3. Content-Independent Contrast Metrics](#3-content-independent-contrast-metrics)
  - [3.1. Global Statistical Measures](#31-global-statistical-measures)
  - [3.2. Histogram Shape Descriptors](#32-histogram-shape-descriptors)
  - [3.3. Dynamic Range Metrics](#33-dynamic-range-metrics)
  - [3.4. Spatial Frequency Content](#34-spatial-frequency-content)
- [4. Distribution-Based Comparison Methods](#4-distribution-based-comparison-methods)
  - [4.1. Histogram Similarity Metrics](#41-histogram-similarity-metrics)
  - [4.2. Statistical Distribution Tests](#42-statistical-distribution-tests)
  - [4.3. Quantile-Based Comparison](#43-quantile-based-comparison)
- [5. Contrast Characterization Features](#5-contrast-characterization-features)
  - [5.1. Global Contrast Index](#51-global-contrast-index)
  - [5.2. Local Contrast Statistics](#52-local-contrast-statistics)
  - [5.3. Edge Density and Strength](#53-edge-density-and-strength)
  - [5.4. Texture Contrast Measures](#54-texture-contrast-measures)
- [6. Multi-Dimensional Contrast Descriptors](#6-multi-dimensional-contrast-descriptors)
  - [6.1. Contrast Feature Vectors](#61-contrast-feature-vectors)
  - [6.2. Distance Metrics in Feature Space](#62-distance-metrics-in-feature-space)
- [7. Scale-Invariant Comparison](#7-scale-invariant-comparison)
  - [7.1. Normalized Metrics](#71-normalized-metrics)
  - [7.2. Relative Contrast Measures](#72-relative-contrast-measures)
- [8. Practical Examples and Interpretation](#8-practical-examples-and-interpretation)
- [9. Applications](#9-applications)
  - [9.1. Dataset Analysis and Quality Control](#91-dataset-analysis-and-quality-control)
  - [9.2. Image Retrieval by Contrast Similarity](#92-image-retrieval-by-contrast-similarity)
  - [9.3. Photography Workflow Optimization](#93-photography-workflow-optimization)
  - [9.4. Content-Aware Processing](#94-content-aware-processing)
  - [9.5. Machine Learning Dataset Curation](#95-machine-learning-dataset-curation)
- [10. Limitations and Considerations](#10-limitations-and-considerations)
- [11. Implementation Guide](#11-implementation-guide)
  - [11.1. Advanced Topics and Missing Pieces](#111-advanced-topics-and-missing-pieces)
- [12. Conclusion](#12-conclusion)
- [13. Further Reading](#13-further-reading)

## 1. The Fundamental Difference

### 1.1. Same-Content Comparison

When comparing contrast between two versions of the same image:
- **Pixel correspondence is meaningful**: We can compare $(x,y)$ in image 1 with $(x,y)$ in image 2
- **Spatial structure is preserved**: Objects are in the same locations
- **Local differences are interpretable**: A difference at pixel $(x,y)$ relates to a specific scene element
- **Example metrics**: SSIM, local contrast difference maps, pixel-wise comparisons

**Use cases**: Enhancement evaluation, compression assessment, change detection

### 1.2. Different-Content Comparison

When comparing contrast between completely different images:
- **Pixel correspondence is meaningless**: $(x,y)$ in image 1 has no relation to $(x,y)$ in image 2
- **Spatial structure is unrelated**: Completely different scenes, objects, compositions
- **Only aggregate statistics are comparable**: Global properties, distributions, statistical descriptors
- **Example metrics**: Histogram similarity, RMS contrast, statistical moments, frequency content

**Use cases**: Dataset analysis, image retrieval, quality normalization, style matching

**Critical Distinction:**

> **These metrics are statistically content-independent (same value regardless of spatial arrangement), but not perceptually invariant to scene semantics.** A cityscape and a forest may have identical edge density but radically different perceived contrast. These metrics capture statistical properties that *correlate with* contrast, not contrast perception itself.

## 2. Why Compare Contrast Across Different Images?

Understanding contrast characteristics across diverse images is crucial for:

- **Dataset Quality Control**: Ensuring consistent contrast distribution in training datasets for machine learning
- **Image Retrieval**: Finding images with similar "look and feel" based on contrast properties
- **Photography Workflows**: Matching contrast styles across different photos in a collection
- **Content Recommendation**: Suggesting images with similar visual characteristics
- **Automated Grading**: Applying consistent contrast adjustments across diverse content
- **Quality Normalization**: Standardizing image quality metrics across varied scenes
- **Style Transfer**: Identifying source images with desired contrast properties
- **Display Optimization**: Adapting content for different viewing conditions

## 3. Content-Independent Contrast Metrics

### 3.1. Global Statistical Measures

These metrics summarize contrast properties without relying on spatial correspondence:

**RMS Contrast (Coefficient of Variation):**

$$ C_{RMS} = \frac{\sigma_L}{\mu_L} $$

where $\sigma_L$ is the standard deviation and $\mu_L$ is the mean luminance.

**Comparison:**

$$ \Delta C_{RMS} = \mid C_{RMS}^{(2)} - C_{RMS}^{(1)} \mid $$

**Properties:**
- Content-independent (same value regardless of spatial arrangement)
- Normalized by mean (accounts for different exposure levels)
- Single scalar value per image
- Range: [0, ∞), typical values for natural images: 0.3-0.7

**Important**: RMS contrast assumes **linear luminance space**, not gamma-encoded RGB. Apply to linear luminance values for meaningful results, not directly to sRGB pixel values.

**Michelson Contrast:**

$$ C_M = \frac{L_{max} - L_{min}}{L_{max} + L_{min}} $$

**Properties:**
- Measures dynamic range utilization
- Range: [0, 1]
- 1 indicates full dynamic range usage
- Sensitive to outliers (single bright/dark pixel affects the metric)

**Important**: Michelson contrast is designed for periodic or two-level patterns. For complex natural scenes, interpret it as a **dynamic-range indicator**, not as perceptual contrast per se.

**Weber Contrast (for peak detection):**

$$ C_W = \frac{L_{peak} - L_{background}}{L_{background}} $$

Useful for images with dominant bright objects on darker backgrounds (or vice versa).

### 3.2. Histogram Shape Descriptors

Histogram-based metrics capture luminance distribution without spatial information:

**Histogram Moments:**

- **Mean** ($\mu$): Average luminance
- **Variance** ($\sigma^2$): Spread of luminance values
- **Skewness**: 

$$ \gamma_1 = \frac{E[(L - \mu)^3]}{\sigma^3} $$

  Positive skew: predominantly dark image with bright highlights  
  Negative skew: predominantly bright image with dark shadows

- **Kurtosis**:

$$ \gamma_2 = \frac{E[(L - \mu)^4]}{\sigma^4} - 3 $$

  High kurtosis: sharp peaks (high contrast with narrow midtone range)  
  Low kurtosis: flat distribution (low contrast, spread across range)

**Comparison:**

$$ D_{moments} = \sqrt{(\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2 + w_s(\gamma_{1,1} - \gamma_{1,2})^2 + w_k(\gamma_{2,1} - \gamma_{2,2})^2} $$

where $w_s$ and $w_k$ are weights for skewness and kurtosis.

**Histogram Entropy:**

$$ H = -\sum_{i=0}^{255} p_i \log_2(p_i) $$

where $p_i$ is the normalized histogram value at bin $i$.

**Properties:**
- Entropy reflects **tonal diversity**, not contrast magnitude
- High entropy: wide distribution of tones (can be high-contrast with even distribution, or tone-mapped HDR)
- Low entropy: concentrated distribution (can be low-contrast uniform scene, or high-contrast bimodal)
- Content-independent
- Range: [0, 8] for 8-bit images

**Important**: Entropy measures distribution spread, not contrast strength. A uniform histogram can represent both flat low-contrast images and evenly-distributed high-contrast images.

### 3.3. Dynamic Range Metrics

**Occupied Dynamic Range:**

$$ R_{occupied} = L_{max} - L_{min} $$

For 8-bit images, range is [0, 255].

**Effective Dynamic Range (excluding outliers):**

$$ R_{effective} = L_{99th} - L_{1st} $$

where $L_{99th}$ and $L_{1st}$ are the 99th and 1st percentile luminance values.

**Dynamic Range Utilization:**

$$ U_{DR} = \frac{R_{effective}}{R_{total}} $$

where $R_{total}$ is the full available range (e.g., 255 for 8-bit).

**Properties:**
- $U_{DR} \approx 1$: Full dynamic range used (high contrast potential)
- $U_{DR} < 0.5$: Limited range (low contrast, washed out or compressed)
- Robust to individual outlier pixels

### 3.4. Spatial Frequency Content

**Average Gradient Magnitude:**

$$ \overline{G} = \frac{1}{MN} \sum_{x,y} \sqrt{G_x(x,y)^2 + G_y(x,y)^2} $$

where $G_x$ and $G_y$ are image gradients (e.g., using Sobel operators).

**Properties:**
- Measures edge density and strength
- High values: many sharp transitions (high local contrast)
- Low values: smooth image (low local contrast)
- Content-independent (same metric for any spatial arrangement of edges)

**Spatial Frequency Ratio (High/Low):**

Compute power spectrum and compare high-frequency vs. low-frequency energy:

$$ R_{freq} = \frac{\sum_{r > r_{cutoff}} P(r)}{\sum_{r \leq r_{cutoff}} P(r)} $$

where $P(r)$ is the radial power spectrum and $r_{cutoff}$ separates low and high frequencies.

**Properties:**
- High ratio: often indicates detail-rich content with high local contrast
- Low ratio: often indicates smooth content with low local contrast
- Independent of specific content

**Important caveats:**
- **Noise** increases high-frequency energy without increasing contrast
- **Fine textures** inflate gradients without necessarily increasing perceived contrast
- **Blur** reduces frequency content but may not always reduce perceived contrast
- These are contrast *correlates*, not direct measures of perceptual contrast

## 4. Distribution-Based Comparison Methods

### 4.1. Histogram Similarity Metrics

When comparing histograms of two different images, we assess the **shape** of the distributions, not pixel-wise correspondence.

**Correlation:**

$$ d_{corr} = \frac{\sum_i (h_1(i) - \bar{h}_1)(h_2(i) - \bar{h}_2)}{\sqrt{\sum_i (h_1(i) - \bar{h}_1)^2 \sum_i (h_2(i) - \bar{h}_2)^2}} $$

Range: [-1, 1], where 1 indicates identical distributions.

**Chi-Square:**

$$ \chi^2 = \sum_{i=0}^{255} \frac{(h_1(i) - h_2(i))^2}{h_1(i) + h_2(i)} $$

Lower values indicate more similar distributions.

**Bhattacharyya Distance:**

$$ d_B = \sqrt{1 - \sum_{i=0}^{255} \sqrt{h_1(i) \cdot h_2(i)}} $$

Range: [0, 1], where 0 indicates identical distributions.

**Earth Mover's Distance (EMD):**

Minimum "work" to transform one histogram into another, considering bin proximity.

### 4.2. Statistical Distribution Tests

**Kolmogorov-Smirnov Test:**

Compare cumulative distribution functions (CDFs):

$$ D_{KS} = \max_k \mid CDF_1(k) - CDF_2(k) \mid $$

**Two-Sample t-Test (for means):**

Test if the mean luminance of two images is significantly different.

**F-Test (for variances):**

Test if the variance (contrast) of two images is significantly different.

**Properties:**
- Provides statistical significance
- Useful for dataset-level analysis
- Can identify if differences are due to chance

**Critical limitation**: These statistical tests assume independent, identically distributed samples. Image pixels are **spatially correlated** and violate i.i.d. assumptions. Additionally, p-values become meaningless for large images due to sample size effects.

**Correct usage**: Apply these tests on **populations of images** (e.g., comparing two datasets), **not on individual image pairs**. For pairwise image comparison, use distance metrics from Section 4.1 or feature-based methods from Section 6.

### 4.3. Quantile-Based Comparison

**Interquartile Range (IQR):**

$$ IQR = Q_3 - Q_1 $$

where $Q_3$ is the 75th percentile and $Q_1$ is the 25th percentile.

**Comparison:**

$$ \Delta IQR = \mid IQR_2 - IQR_1 \mid $$

**Quantile-Quantile (Q-Q) Plot Analysis:**

Plot quantiles of image 1 against quantiles of image 2. If distributions are similar, points lie on a straight line.

**Quantile Distance:**

$$ D_Q = \sum_{q=0.1, 0.2, \ldots, 0.9} \mid L_1(q) - L_2(q) \mid $$

where $L_i(q)$ is the luminance value at quantile $q$ in image $i$.

## 5. Contrast Characterization Features

### 5.1. Global Contrast Index

A composite metric combining multiple aspects:

$$ GCI = w_1 \cdot C_{RMS} + w_2 \cdot U_{DR} + w_3 \cdot H + w_4 \cdot \overline{G} $$

where weights $w_i$ can be tuned for specific applications.

**Important**: GCI is an **engineering heuristic**, not a perceptually validated metric. It is:
- Not invariant to feature scaling
- Highly sensitive to weight choices
- Useful for application-specific tuning, but lacks theoretical foundation

For perceptually grounded comparisons, consider perceptual models like multi-scale contrast (Peli) or validated image quality metrics (SSIM, MS-SSIM).

### 5.2. Local Contrast Statistics

**Local Contrast Distribution:**

Compute local contrast in sliding windows:

$$ C_{local}(x, y) = \frac{\sigma_w(x, y)}{\mu_w(x, y)} $$

Then characterize the **distribution** of local contrast values:
- Mean local contrast: $\overline{C_{local}}$
- Std dev of local contrast: $\sigma_{C_{local}}$
- Max local contrast: $C_{local}^{max}$

**Comparison:**

$$ \Delta \overline{C_{local}} = \mid \overline{C_{local}^{(2)}} - \overline{C_{local}^{(1)}} \mid $$

**Properties:**
- Captures spatial variation in contrast
- Higher $\sigma_{C_{local}}$: non-uniform contrast (some regions high, some low)
- Lower $\sigma_{C_{local}}$: uniform contrast throughout image

### 5.3. Edge Density and Strength

**Edge Density:**

$$ \rho_{edge} = \frac{\text{Number of edge pixels}}{\text{Total pixels}} $$

where edge pixels are determined by thresholding gradient magnitude.

**Average Edge Strength:**

$$ \overline{E} = \frac{1}{N_{edge}} \sum_{\text{edge pixels}} G(x, y) $$

**Comparison:**

$$ D_{edge} = \sqrt{(\rho_{edge}^{(1)} - \rho_{edge}^{(2)})^2 + w \cdot (\overline{E}^{(1)} - \overline{E}^{(2)})^2} $$

### 5.4. Texture Contrast Measures

**Gray Level Co-occurrence Matrix (GLCM) Contrast:**

$$ C_{GLCM} = \sum_{i,j} (i-j)^2 \cdot P(i,j) $$

where $P(i,j)$ is the GLCM, representing the probability of adjacent pixels having intensities $i$ and $j$.

**Properties:**
- Captures texture-level contrast
- High values: rough texture with large intensity variations
- Low values: smooth texture
- Content-independent (aggregated over entire image)

## 6. Multi-Dimensional Contrast Descriptors

### 6.1. Contrast Feature Vectors

Represent each image as a feature vector:

$$ \mathbf{f} = [C_{RMS}, U_{DR}, H, \overline{G}, \gamma_1, \gamma_2, \overline{C_{local}}, \sigma_{C_{local}}, \rho_{edge}, \overline{E}, C_{GLCM}]^T $$

### 6.2. Distance Metrics in Feature Space

**Euclidean Distance:**

$$ D_{euclidean} = \|\mathbf{f}_1 - \mathbf{f}_2\| = \sqrt{\sum_i (f_{1,i} - f_{2,i})^2} $$

**Mahalanobis Distance:**

$$ D_{mahalanobis} = \sqrt{(\mathbf{f}_1 - \mathbf{f}_2)^T \Sigma^{-1} (\mathbf{f}_1 - \mathbf{f}_2)} $$

where $\Sigma$ is the covariance matrix of features (computed from a reference dataset).

**Cosine Similarity:**

$$ \text{sim}_{cosine} = \frac{\mathbf{f}_1 \cdot \mathbf{f}_2}{\|\mathbf{f}_1\| \|\mathbf{f}_2\|} $$

**Properties:**
- Multi-dimensional representation captures various aspects of contrast
- Distance metric choice depends on application
- Can be used for clustering, retrieval, classification

## 7. Scale-Invariant Comparison

### 7.1. Normalized Metrics

To compare images of different sizes or resolutions:

**Normalized RMS Contrast:** Already normalized by mean luminance

**Normalized Histogram:** Divide histogram by total pixel count

**Normalized Edge Density:** Already a ratio (edge pixels / total pixels)

### 7.2. Relative Contrast Measures

**Contrast Ratio:**

$$ R_C = \frac{C_{RMS}^{(2)}}{C_{RMS}^{(1)}} $$

Interpretation:
- $R_C > 1$: Image 2 has higher contrast
- $R_C < 1$: Image 1 has higher contrast
- $R_C \approx 1$: Similar contrast

**Log Contrast Difference (dB):**

$$ \Delta C_{dB} = 20 \log_{10}\left(\frac{C_{RMS}^{(2)}}{C_{RMS}^{(1)}}\right) $$

Symmetric around 0 dB (no difference).

## 8. Practical Examples and Interpretation

Let's compare contrast characteristics across four completely different images.

**Note**: The numeric values below are **illustrative examples** for demonstration purposes, not measured from actual images. They represent plausible ranges to demonstrate how metrics distinguish between image types.

| Image | Content Description | $C_{RMS}$ | $U_{DR}$ | $H$ (bits) | $\overline{G}$ | Interpretation |
|-------|---------------------|-----------|----------|------------|----------------|----------------|
| **A: Sunset** | Golden hour landscape with vibrant sky | 0.52 | 0.85 | 7.2 | 12.5 | High contrast, full dynamic range, rich detail |
| **B: Portrait** | Studio portrait with soft lighting | 0.28 | 0.45 | 6.8 | 6.3 | Low contrast, limited range, smooth gradients |
| **C: Cityscape** | Urban scene with bright signs and dark buildings | 0.68 | 0.95 | 7.5 | 18.2 | Very high contrast, full range, many edges |
| **D: Foggy Forest** | Misty forest with low visibility | 0.15 | 0.30 | 5.8 | 3.1 | Very low contrast, compressed range, few edges |

**Pairwise Comparisons:**

| Pair | $\Delta C_{RMS}$ | Interpretation |
|------|------------------|----------------|
| A vs. B | 0.24 | Sunset has significantly higher contrast than portrait |
| A vs. C | 0.16 | Both high contrast, cityscape slightly higher |
| A vs. D | 0.37 | Dramatic difference: sunset vs. foggy forest |
| B vs. C | 0.40 | Portrait is low contrast, cityscape is very high |
| B vs. D | 0.13 | Both low contrast, portrait slightly higher |
| C vs. D | 0.53 | Maximum difference: high-contrast cityscape vs. low-contrast fog |

**Similarity Ranking (by contrast characteristics):**
1. A (Sunset) and C (Cityscape) - both high contrast
2. B (Portrait) and D (Forest) - both low contrast
3. Most dissimilar: C (Cityscape) and D (Forest)

## 9. Applications

### 9.1. Dataset Analysis and Quality Control

**Use case:** Analyzing contrast distribution in a large image dataset (e.g., for machine learning)

**Approach:**
1. Compute contrast features for all images
2. Plot histogram of $C_{RMS}$ values
3. Identify outliers (extremely high or low contrast)
4. Filter or normalize dataset

**Example:** Training a neural network on images with widely varying contrast can hurt performance. Identify and normalize images outside the 5th-95th percentile range.

### 9.2. Image Retrieval by Contrast Similarity

**Use case:** "Find me images that look like this one" (in terms of contrast, not content)

**Approach:**
1. Extract contrast feature vector $\mathbf{f}_{query}$ from query image
2. Compute distance to all images in database
3. Retrieve top-K images with smallest distance

**Example:** A photographer wants to find all images in their portfolio with similar "moody, low-contrast" aesthetic, regardless of subject.

### 9.3. Photography Workflow Optimization

**Use case:** Applying consistent contrast adjustments across diverse photos

**Approach:**
1. Identify reference image with desired contrast characteristics
2. Compute target contrast metrics
3. For each image, compute current metrics
4. Apply adaptive enhancement to match target metrics

**Example:** Preparing a photo series for exhibition—ensure all images have similar contrast properties for visual coherence.

### 9.4. Content-Aware Processing

**Use case:** Automatic contrast adjustment based on image type

**Approach:**
1. Classify image into categories (portrait, landscape, architecture, etc.)
2. For each category, define typical contrast ranges
3. If image contrast deviates significantly, apply correction

**Example:** Low-contrast portraits are acceptable (soft lighting), but low-contrast landscapes may indicate haze—apply dehazing.

### 9.5. Machine Learning Dataset Curation

**Use case:** Creating balanced training datasets

**Approach:**
1. Compute contrast features for candidate images
2. Cluster images by contrast characteristics
3. Sample uniformly from clusters to ensure diversity

**Example:** Ensuring a facial recognition dataset includes faces in various contrast conditions (backlit, well-lit, high-key, low-key).

## 10. Limitations and Considerations

### When Content Matters

Even though these metrics are content-independent, **context matters**:
- A low-contrast portrait may be perfectly fine (soft, flattering light)
- A low-contrast landscape may indicate technical problems (haze, poor exposure)

### Perceptual vs. Statistical

Statistical similarity doesn't always mean perceptual similarity:
- Two images with identical $C_{RMS}$ may look very different
- Spatial distribution matters for perception (not captured by global metrics)

### Histogram Clipping

Images with clipped highlights or shadows can have misleading metrics:
- High $C_{RMS}$ might result from clipping, not true scene contrast
- Always check for clipping when interpreting contrast metrics

### Resolution Dependence

Some metrics (like edge density) can vary with image resolution:
- Downsampling can reduce perceived edge density
- Normalize by image size or resample to standard resolution

### Color vs. Luminance

Most metrics here focus on luminance contrast:
- Color images have chromatic contrast as well
- Consider converting to CIELAB and analyzing L*, a*, b* channels separately

## 11. Implementation Guide

### Python Example

```python
import numpy as np
from scipy import ndimage
from skimage import feature, measure

def compute_contrast_features(image_gray):
    """
    Compute comprehensive contrast features for a grayscale image.
    
    Args:
        image_gray: 2D numpy array (grayscale image, range [0, 255])
    
    Returns:
        dict: Dictionary of contrast features
    """
    features = {}
    
    # 1. RMS Contrast
    mean_L = np.mean(image_gray)
    std_L = np.std(image_gray)
    features['C_RMS'] = std_L / (mean_L + 1e-8)
    
    # 2. Michelson Contrast
    L_max = np.max(image_gray)
    L_min = np.min(image_gray)
    features['C_Michelson'] = (L_max - L_min) / (L_max + L_min + 1e-8)
    
    # 3. Dynamic Range Utilization
    L_99 = np.percentile(image_gray, 99)
    L_01 = np.percentile(image_gray, 1)
    R_effective = L_99 - L_01
    features['U_DR'] = R_effective / 255.0
    
    # 4. Histogram Entropy
    hist, _ = np.histogram(image_gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]  # Remove zero bins
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    features['H'] = entropy
    
    # 5. Histogram Moments
    features['mean'] = mean_L
    features['std'] = std_L
    features['skewness'] = measure.moments_central(image_gray.ravel(), center=mean_L, order=3)[0] / (std_L**3 + 1e-8)
    features['kurtosis'] = measure.moments_central(image_gray.ravel(), center=mean_L, order=4)[0] / (std_L**4 + 1e-8) - 3
    
    # 6. Average Gradient Magnitude
    grad_x = ndimage.sobel(image_gray, axis=1)
    grad_y = ndimage.sobel(image_gray, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    features['avg_gradient'] = np.mean(grad_mag)
    
    # 7. Edge Density
    edges = feature.canny(image_gray, sigma=1.0)
    features['edge_density'] = np.sum(edges) / edges.size
    features['avg_edge_strength'] = np.mean(grad_mag[edges])
    
    # 8. Local Contrast Statistics
    window_size = 11
    local_mean = ndimage.uniform_filter(image_gray, size=window_size)
    local_std = ndimage.generic_filter(image_gray, np.std, size=window_size)
    local_contrast = local_std / (local_mean + 1e-8)
    features['mean_local_contrast'] = np.mean(local_contrast)
    features['std_local_contrast'] = np.std(local_contrast)
    
    return features

def compare_contrast(features1, features2):
    """
    Compare two images based on their contrast features.
    
    Args:
        features1: dict from compute_contrast_features
        features2: dict from compute_contrast_features
    
    Returns:
        dict: Comparison metrics
    """
    comparison = {}
    
    # Absolute differences for key metrics
    comparison['delta_C_RMS'] = abs(features1['C_RMS'] - features2['C_RMS'])
    comparison['delta_U_DR'] = abs(features1['U_DR'] - features2['U_DR'])
    comparison['delta_H'] = abs(features1['H'] - features2['H'])
    comparison['delta_avg_gradient'] = abs(features1['avg_gradient'] - features2['avg_gradient'])
    
    # Contrast ratio
    comparison['contrast_ratio'] = features2['C_RMS'] / (features1['C_RMS'] + 1e-8)
    comparison['contrast_ratio_dB'] = 20 * np.log10(comparison['contrast_ratio'])
    
    # Feature vector distance
    feature_keys = ['C_RMS', 'U_DR', 'H', 'avg_gradient', 'mean_local_contrast']
    vec1 = np.array([features1[k] for k in feature_keys])
    vec2 = np.array([features2[k] for k in feature_keys])
    
    # IMPORTANT: For proper comparison, normalize using dataset-wide statistics
    # This example uses fixed bounds for each feature based on typical ranges
    # In production, compute these from your dataset
    feature_ranges = {
        'C_RMS': (0.0, 1.0),           # Typical range for natural images
        'U_DR': (0.0, 1.0),             # Already normalized
        'H': (0.0, 8.0),                # Max entropy for 8-bit
        'avg_gradient': (0.0, 50.0),    # Typical range (tune for your data)
        'mean_local_contrast': (0.0, 1.0)  # Typical range
    }
    
    # Normalize each feature using fixed bounds
    vec1_norm = np.array([(features1[k] - feature_ranges[k][0]) / 
                          (feature_ranges[k][1] - feature_ranges[k][0] + 1e-8) 
                          for k in feature_keys])
    vec2_norm = np.array([(features2[k] - feature_ranges[k][0]) / 
                          (feature_ranges[k][1] - feature_ranges[k][0] + 1e-8) 
                          for k in feature_keys])
    
    # Clip to [0, 1] in case values exceed expected ranges
    vec1_norm = np.clip(vec1_norm, 0, 1)
    vec2_norm = np.clip(vec2_norm, 0, 1)
    
    comparison['euclidean_distance'] = np.linalg.norm(vec1_norm - vec2_norm)
    comparison['cosine_similarity'] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    
    return comparison

# Example usage:
# image1 = cv2.imread('sunset.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('portrait.jpg', cv2.IMREAD_GRAYSCALE)
# 
# features1 = compute_contrast_features(image1)
# features2 = compute_contrast_features(image2)
# comparison = compare_contrast(features1, features2)
# 
# print(f"RMS Contrast difference: {comparison['delta_C_RMS']:.3f}")
# print(f"Contrast ratio (dB): {comparison['contrast_ratio_dB']:.2f} dB")
```

### Key Implementation Notes

1. **Normalization (CRITICAL)**: When normalizing feature vectors for distance computation:
   - **DO NOT** normalize each vector independently (destroys comparability)
   - **DO** use dataset-wide statistics (min/max or z-score computed from reference dataset)
   - **OR** use fixed bounds based on theoretical/typical ranges for each feature
   - The example code uses fixed bounds; adapt to your specific application
2. **Robustness**: Use percentiles (e.g., 1st, 99th) instead of min/max to avoid outliers
3. **Window sizes**: Adjust local contrast window size based on image resolution
4. **Color images**: Apply to luminance channel or process RGB channels separately
5. **Linear space**: Apply RMS contrast and gradient computations in linear luminance space, not gamma-encoded RGB

## 11.1. Advanced Topics and Missing Pieces

### Perceptual Contrast Models

The metrics presented in this post are **statistical contrast correlates**, not perceptually grounded contrast measures. For applications requiring perceptual accuracy, consider:

**Multi-Scale Contrast (Peli, 1990)**:
- Decomposes images into multiple spatial frequency bands
- Computes local contrast at each scale
- Aggregates across scales weighted by human contrast sensitivity
- More closely matches human perception than single-scale metrics

**Contrast Sensitivity Function (CSF)**:
- Human sensitivity to contrast varies with spatial frequency
- Peak sensitivity around 4-8 cycles/degree
- Filtering images by CSF provides perceptually weighted contrast
- Used in advanced image quality metrics (HDR-VDP, PU-SSIM)

**Why LPIPS/Deep Features Are Not Contrast-Pure**:
- Modern learned perceptual metrics (LPIPS, DISTS) capture overall perceptual similarity
- They conflate contrast with texture, color, and semantic content
- Useful for general image comparison but not for isolating contrast properties
- If you need contrast-specific comparison, stick to the statistical metrics in this post

### HDR and Tone Mapping Considerations

Modern imaging pipelines introduce significant complexity:

**Display-Referred vs Scene-Referred**:
- Metrics here assume **display-referred** images (SDR, 0-255 range)
- HDR images (scene-referred, linear, high dynamic range) require different treatment
- Applying these metrics to HDR requires tone mapping first, which itself affects contrast

**Tone Mapping Impact**:
- Different tone mapping operators (TMOs) produce vastly different contrast
- Global TMOs compress dynamic range uniformly (low local contrast)
- Local TMOs preserve local contrast but may introduce halos
- Comparing "contrast" across TMOs is ill-defined without specifying the operator

**Recommendations**:
- For SDR images: apply metrics as presented
- For HDR images: specify the tone mapping operator and apply to tone-mapped result
- For cross-HDR/SDR comparison: compute metrics in a common space (e.g., both tone-mapped with same operator)

### When These Metrics Break Down

These content-independent metrics are **not universally applicable**:

**Failure cases**:
- **Semantic content matters**: A blurry portrait may have acceptable contrast, but a blurry document is unusable
- **Spatial distribution ignored**: Two images with identical RMS contrast can look completely different
- **Color-luminance interactions**: Chromatic contrast can compensate for low luminance contrast (opponent color effects)
- **Perceptual adaptation**: Contrast perception depends on viewing conditions (ambient light, display, adaptation state)

**Better approaches for these cases**:
- For semantic tasks: use task-specific quality metrics (OCR quality for documents, face quality scores for portraits)
- For perceptual tasks: use multi-scale, CSF-weighted, or learned perceptual metrics
- For color images: analyze luminance and chromatic contrast separately (CIELAB L*, a*, b*)

## 12. Conclusion

Comparing contrast across different images with unrelated content requires a fundamentally different approach than comparing versions of the same image. The key principles are:

1. **Use content-independent metrics**: Global statistics, histogram properties, frequency content
2. **Avoid spatial correspondence**: Don't compare pixel-by-pixel or local regions directly
3. **Characterize distributions**: Focus on statistical properties and shape descriptors
4. **Multi-dimensional representation**: Combine multiple metrics into feature vectors
5. **Context matters**: Interpret metrics in the context of image type and application

**When to use which approach:**

| Scenario | Approach | Key Metrics |
|----------|----------|-------------|
| Same image, different processing | Pixel-wise comparison | SSIM, local contrast maps, pixel differences |
| Different images, contrast similarity | Distribution comparison | RMS contrast, histogram metrics, feature vectors |
| Dataset quality control | Aggregate statistics | Mean RMS contrast, contrast distribution, outlier detection |
| Image retrieval | Feature distance | Multi-dimensional feature vectors, distance metrics |

By understanding these distinctions and applying appropriate metrics, we can effectively analyze, compare, and manipulate contrast characteristics across diverse image collections, enabling applications from photography workflows to machine learning dataset curation.

## 13. Further Reading

- [Understanding Contrast in Images: From Perception to Computation]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }})
- [Understanding Contrast in Color Images: Beyond Luminance]({{ "/2025/12/27/understanding-color-contrast.html" | relative_url }})
- [Measuring Contrast Between Two Color Images: Comparison Metrics and Methods]({{ "/2025/12/28/measuring-contrast-between-images.html" | relative_url }})
- Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
- Peli, E. (1990). "Contrast in complex images." *Journal of the Optical Society of America A*, 7(10), 2032-2040.
- Matkovic, K., et al. (2005). "Global contrast factor—a new approach to image contrast." *Computational Aesthetics*, 159-168.
- Haralick, R. M., et al. (1973). "Textural features for image classification." *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), 610-621.

