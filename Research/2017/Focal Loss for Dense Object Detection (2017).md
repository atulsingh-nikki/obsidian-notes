---
title: Focal Loss for Dense Object Detection (2017)
aliases:
  - RetinaNet
  - Focal Loss
authors:
  - Tsung-Yi Lin
  - Priya Goyal
  - Ross Girshick
  - Kaiming He
  - Piotr Dollár
year: 2017
venue: ICCV
doi: 10.1109/ICCV.2017.324
arxiv: https://arxiv.org/abs/1708.02002
code: https://github.com/facebookresearch/Detectron
citations: 20,000+
dataset:
  - MS COCO
tags:
  - paper
  - object-detection
  - loss-function
  - dense-detection
fields:
  - vision
  - detection
related:
  - "[[YOLO You Only Look Once — Unified, Real-Time Object Detection|YOLO (2016)]]"
  - "[[Feature Pyramid Networks for Object Detection (2017)|Feature Pyramid Networks]]"
predecessors:
  - "[[SSD (Single Shot MultiBox Detector)]]"
  - "[[Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks|Faster R-CNN]]"
successors:
  - "[[DETR End-to-End Object Detection with Transformers (2020)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
This paper introduced **RetinaNet**, a one-stage object detector, and the **Focal Loss**, which addresses the extreme class imbalance problem in dense detection by reducing the loss contribution from well-classified examples. This allowed one-stage detectors to achieve accuracy competitive with two-stage detectors (e.g., [[Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks|Faster R-CNN (2015)]]).

# Key Idea
> Use **Focal Loss** to focus training on hard, misclassified examples, enabling dense one-stage detectors to match two-stage accuracy.

# Method
- **Architecture**: RetinaNet = one-stage detector with a Feature Pyramid Network (FPN) backbone.  
- **Focal Loss**:  
  $$
  FL(p_t) = -(1-p_t)^\gamma \log(p_t)
 $$  
  where $p_t$ is the predicted probability for the true class, and γ (gamma) is a focusing parameter.  
- Effect: down-weights loss for easy, high-confidence examples, prevents them from overwhelming training.  
- Trained on dense anchors, similar to SSD, but stabilized with focal loss.  

# Results
- Achieved **state-of-the-art on MS COCO** at the time.  
- Outperformed SSD and YOLO while maintaining real-time speed.  
- Matched [[Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks|Faster R-CNN]] accuracy with higher efficiency.  

# Why it Mattered
- Solved the **accuracy gap** between one-stage and two-stage detectors.  
- Established RetinaNet + Focal Loss as a **new standard for dense detection**.  
- Focal Loss has been reused in many tasks beyond detection (segmentation, keypoint detection, generative modeling).  

# Architectural Pattern
- One-stage dense detector.  
- FPN backbone + anchor-based classification/regression.  
- New loss function to handle class imbalance.  

# Connections
- **Contemporaries**: YOLOv2 (2017), SSD.  
- **Influence**: CenterNet, DETR (transformer detectors), diffusion-based detectors.  

# Implementation Notes
- Focal Loss parameter γ typically set to 2.  
- α-balancing term sometimes added to account for class imbalance.  
- Works best with strong multi-scale features (FPN).  

# Critiques / Limitations
- Anchor-based (later anchor-free detectors simplified pipelines).  
- Requires careful tuning of γ and α.  
- Training still sensitive compared to two-stage methods.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1708.02002)  
- [Detectron implementation](https://github.com/facebookresearch/Detectron)  
- [PyTorch implementation](https://github.com/yhenon/pytorch-retinanet)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Matrix multiplications in CNN backbone.  
- **Probability & Statistics**: Cross-entropy vs focal loss.  
- **Optimization Basics**: Loss weighting, handling imbalance.  

## Postgraduate-Level Concepts
- **Neural Network Design**: One-stage vs two-stage detectors.  
- **Computer Vision**: Dense object detection benchmarks.  
- **Research Methodology**: Loss design as a lever for performance.  
- **Advanced Optimization**: Stability in dense prediction training.  

---

# My Notes
- Directly relevant to **object selection/tracking in video**, where imbalance is also severe.  
- Open question: Can **focal-like reweighting** improve mask training for soft edges?  
- Possible extension: Combine **focal loss with diffusion-based objectives** for controllable, balanced generation.  
