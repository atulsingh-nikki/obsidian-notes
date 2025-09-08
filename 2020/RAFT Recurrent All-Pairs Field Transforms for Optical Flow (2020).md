---
title: "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow (2020)"
aliases:
  - RAFT
  - Recurrent All-Pairs Field Transforms
authors:
  - Zachary Teed
  - Jia Deng
year: 2020
venue: "ECCV (Best Paper)"
doi: "10.1007/978-3-030-58536-5_7"
arxiv: "https://arxiv.org/abs/2003.12039"
code: "https://github.com/princeton-vl/RAFT"
citations: 6000+
dataset:
  - FlyingChairs
  - FlyingThings3D
  - Sintel
  - KITTI
tags:
  - paper
  - optical-flow
  - architecture
  - iterative-refinement
fields:
  - vision
  - motion-estimation
related:
  - "[[FlowNet (2015)]]"
  - "[[PWC-Net (2018)]]"
  - "[[GMA (2021)]]"
predecessors:
  - "[[PWC-Net (2018)]]"
successors:
  - "[[GMA (Global Motion Aggregation, 2021)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**RAFT** introduced a new architecture for **optical flow estimation** that set a new state of the art. It leveraged a **4D all-pairs correlation volume** (every pixel in image1 matched with every pixel in image2) and refined flow estimates via a **recurrent update operator**, enabling precise, globally-aware motion estimation.

# Key Idea
> Compute correlations between all pixel pairs once, then refine flow estimates iteratively using a recurrent unit over the correlation volume.

# Method
- **Feature extraction**: CNN encoder for both images.  
- **All-pairs correlation volume**: Compute correlations between every pixel in image1 and image2 (dense 4D volume).  
- **Recurrent update operator**:  
  - A convolutional GRU refines flow estimates over multiple iterations.  
  - At each step, the network samples from the correlation volume and updates the flow field.  
- **Iterative refinement**: 12–32 iterations gradually improve accuracy.  

# Results
- Achieved **SOTA performance** on Sintel and KITTI benchmarks at publication.  
- Robust to large displacements thanks to global correlation volume.  
- Outperformed prior methods like PWC-Net and LiteFlowNet by large margins.  

# Why it Mattered
- Represented an **architectural revolution** for optical flow.  
- Showed that iterative refinement + all-pairs correlation is more effective than pyramid warping.  
- Inspired follow-ups like GMA, CRAFT, and transformer-based flow models.  

# Architectural Pattern
- CNN feature extractor.  
- 4D correlation volume.  
- Recurrent GRU update operator for iterative flow refinement.  

# Connections
- Improved upon pyramid/warping designs (FlowNet2, PWC-Net).  
- Inspired transformer-based optical flow networks (e.g., FlowFormer).  
- Useful in video editing and motion analysis pipelines.  

# Implementation Notes
- Correlation volume computed once, reused in all iterations.  
- Training uses synthetic datasets (FlyingChairs, FlyingThings3D) + fine-tuning on Sintel/KITTI.  
- Inference relatively efficient despite dense correlation, thanks to GPU optimization.  

# Critiques / Limitations
- High memory usage for very high-resolution inputs.  
- Iterative refinement increases latency compared to lightweight models.  
- Still requires fine-tuning for real-world deployment.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Optical flow: estimating motion between consecutive frames.  
- Correlation volumes as similarity maps.  
- Iterative refinement with recurrent units.  

## Postgraduate-Level Concepts
- 4D all-pairs correlation for global motion reasoning.  
- Recurrent GRU update operators in dense prediction.  
- Trade-offs: memory vs accuracy in flow estimation.  

---

# My Notes
- RAFT is the **ResNet moment for optical flow**: simple, elegant, and dominating benchmarks.  
- Open question: Can RAFT’s recurrent refinement be adapted for **video diffusion consistency**?  
- Possible extension: Replace GRU with transformer modules (as seen in FlowFormer).  
- Still my go-to baseline when thinking about **dense correspondence in video pipelines**.  

---
