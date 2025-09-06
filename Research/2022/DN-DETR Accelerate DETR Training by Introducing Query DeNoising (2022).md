---
title: "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising (2022)"
aliases: 
  - DN-DETR
  - DeNoising DETR
authors:
  - Feng Li
  - Hao Zhang
  - Huaizhe Xu
  - Shilong Liu
  - Lei Zhang
  - Lionel M. Ni
  - Heung-Yeung Shum
year: 2022
venue: "CVPR"
doi: "10.1109/CVPR52688.2022.00957"
arxiv: "https://arxiv.org/abs/2203.01305"
code: "https://github.com/IDEA-Research/DN-DETR"
citations: 1500+
dataset:
  - COCO
tags:
  - paper
  - object-detection
  - transformer
  - detr-variants
fields:
  - vision
  - detection
related:
  - "[[DETR (2020)]]"
  - "[[Deformable DETR (2021)]]"
  - "[[DINO (2023)]]"
predecessors:
  - "[[Deformable DETR (2021)]]"
successors:
  - "[[DINO (2023)]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
DN-DETR introduced **denoising training for queries** to improve DETR’s convergence and recall. By injecting noisy versions of ground truth boxes into training queries, the model learns to **denoise them** and stabilize bipartite matching. This significantly **speeds up training** and improves detection quality.

# Key Idea
> Add **denoising queries** during training, teaching DETR to refine noisy boxes and handle matching more effectively, thereby accelerating convergence.

# Method
- **Denoising Queries**:  
  - Perturb ground truth boxes (random noise on position, size, labels).  
  - Inject them as extra queries alongside normal object queries.  
  - Model learns to map noisy queries back to true boxes/labels.  
- **Training**:  
  - Retains Hungarian matching for regular queries.  
  - Denoising loss applied directly to noisy queries.  
- **Inference**:  
  - No denoising queries used → same inference cost as DETR.  

# Results
- Achieved **2–6× faster convergence** on COCO compared to DETR/Deformable DETR.  
- Improved **recall and precision**, especially in early training epochs.  
- Outperformed Faster R-CNN baselines with fewer epochs.  

# Why it Mattered
- Solved DETR’s training inefficiency while keeping end-to-end nature.  
- Made DETR variants competitive for practical use.  
- Provided a simple yet powerful idea reused in later models (DINO, diffusion detectors).  

# Architectural Pattern
- Transformer-based DETR backbone.  
- Noisy query augmentation during training.  
- Dual loss: denoising + matching.  

# Connections
- **Contemporaries**: Conditional DETR, Efficient DETR.  
- **Influence**: DINO (denoising + contrastive matching).  

# Implementation Notes
- Number of denoising queries is a tunable parameter.  
- No extra cost at inference.  
- Compatible with Deformable DETR backbone.  

# Critiques / Limitations
- Requires careful tuning of noise distribution.  
- Still slower than YOLO-style detectors.  
- Hungarian matching remains a bottleneck.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/2203.01305)  
- [Official PyTorch implementation](https://github.com/IDEA-Research/DN-DETR)  
- [COCO pre-trained models available]  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Probability**: Noise injection for robustness.  
- **Geometry**: Bounding box perturbations.  
- **Optimization**: Multi-task loss with denoising.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Query denoising in transformers.  
- **Computer Vision**: Accelerating training via curriculum-like noise.  
- **Research Methodology**: Stable bipartite matching under noise.  
- **Advanced Optimization**: Joint denoising + detection losses.  

---

# My Notes
- DN-DETR is elegant: **teach the model to denoise its own noisy inputs**.  
- Connects naturally to **diffusion-like priors** (denoising training).  
- Open question: Can DN-DETR’s denoising be extended to **video queries** for tracking?  
- Possible extension: Merge DN-DETR with **video diffusion detectors** for temporally stable object editing.  

---
