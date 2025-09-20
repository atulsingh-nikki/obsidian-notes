---
title: "HDRSDR-VQA: A Subjective Video Quality Dataset for HDR and SDR Comparative Evaluation (2025)"
aliases:
  - HDRSDR VQA
authors:
  - Bowen Chen
  - Cheng-han Lee
  - Yixu Chen
  - Zaixi Shang
  - Hai Wei
  - Alan C. Bovik
year: 2025
venue: "arXiv preprint"
doi: "10.48550/arXiv.2505.21831"
dataset:
  - name: HDRSDR-VQA
  - size: 960 videos
  - source sequences: 54 clean sequences
  - formats: both HDR10 and SDR
  - distortion levels: 9 per format (8 distortions + reference) :contentReference[oaicite:1]{index=1}
tags:
  - video-quality
  - HDR
  - SDR
  - subjective study
  - dataset
fields:
  - video-quality-assessment
  - perceptual-modeling
  - streaming / encoding
related:
  - "[[LIVE-HDR VQA]]" (other HDR video quality datasets)
  - "[[HDR vs SDR studies]]"
predecessors:
  - LIVE-HDR
  - LIVE HDRvsSDR (etc.) :contentReference[oaicite:2]{index=2}
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary  
HDRSDR-VQA provides a large video quality dataset designed to compare High Dynamic Range (HDR) vs Standard Dynamic Range (SDR) video content under realistic viewing conditions. It supports analysis of when HDR is perceptually preferred and quantifies how distortions affect HDR/SDR formats. :contentReference[oaicite:3]{index=3}

# Key Details

- **Content & sources**: 54 diverse source sequences, including open-source 8K HDR content, VoD content, live sports, and anchor videos. :contentReference[oaicite:4]{index=4}  
- **Video formats**: Each source rendered in both HDR10 and SDR, then encoded with 9 quality levels (8 distorted + reference) for each format → total of **960 video clips** (480 HDR + 480 SDR). :contentReference[oaicite:5]{index=5}  
- **Display conditions**: Tested on 6 different consumer HDR-capable televisions (varied technologies and price tiers) to capture variability across real devices. :contentReference[oaicite:6]{index=6}  
- **Participants**: 145 human subjects. :contentReference[oaicite:7]{index=7}  
- **Method**: Pairwise comparisons (PC) with active sampling to gather subjective judgments; data scaled into Just-Objectionable-Difference (JOD) scores. :contentReference[oaicite:8]{index=8}  

# Results / Findings

- HDR generally preferred in content with high brightness, colorfulness, and rich texture. :contentReference[oaicite:9]{index=9}  
- In low brightness, high motion, or low dynamic scenes, SDR sometimes outperforms or matches HDR (due to distortions, bitrate or display limitations). :contentReference[oaicite:10]{index=10}  
- The margins of preference vary with bitrate, device capability, and scene attributes.  

# Why It Matters

- Fills a gap in existing VQA datasets by **pairing HDR & SDR** directly, under matched distortion levels and content, enabling direct comparisons. :contentReference[oaicite:11]{index=11}  
- Useful for improving objective VQA metrics to better model HDR vs SDR perceptual differences.  
- Impacts streaming/adaptive encoding decisions: which format to use depending on content, display, and network conditions.  

# Applications

- Training or benchmarking perceptual quality models that handle HDR & SDR.  
- Adaptive streaming services that switch between HDR/SDR formats.  
- Display manufacturers optimizing tone mapping / display pipelines.  
- Research into perceptual thresholds (when HDR is worth the extra cost / bitrate).  

# Implementation Notes & Caveats

- The dataset uses pairwise comparisons, which may be more reliable for small perceptual differences but are more labor-intensive.  
- Display devices vary significantly; results may depend on device brightness, color gamut.  
- Distortion types (resolution, bitrate) are specific; results generalize best under similar encoding distortions.  

# Questions / Open Directions

- How well do current objective VQA metrics correlate with the HDRSDR-VQA JOD scores?  
- Could learned VQA models be adapted to use display-metadata (peak brightness, color gamut) to better predict HDR/SDR differences?  
- How do different tone mapping algorithms affect perceived quality in this dataset?  
- Is there room for viewer adaptation (e.g. viewing ambient light) to shift preference between HDR & SDR?  

# My Notes

- Dataset looks well-designed: large and balanced HDR vs SDR, multiple devices, paired content.  
- Might become a standard benchmark for HDR vs SDR perceptual work.  
- Might test or expose weaknesses of streaming pipelines that assume HDR always superior.  
- If working on VQA or encoding / streaming, this is likely essential reference.  

---
