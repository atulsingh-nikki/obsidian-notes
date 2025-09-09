---
title: "DeepCap: Monocular Human Performance Capture Using Weak Supervision (2020)"
aliases:
  - DeepCap
  - Monocular Performance Capture
authors:
  - Marc Habermann
  - Weipeng Xu
  - Michael Zollhöfer
  - Gerard Pons-Moll
  - Christian Theobalt
year: 2020
venue: "CVPR (Best Student Paper Honorable Mention)"
doi: "10.1109/CVPR42600.2020.00561"
arxiv: "https://arxiv.org/abs/2004.04675"
code: "https://graphics.tu-bs.de/publications/deepcap"
citations: ~400+
dataset:
  - Captured monocular human performance datasets
  - Public benchmark sequences
tags:
  - paper
  - 3d-reconstruction
  - human-performance-capture
  - weak-supervision
  - non-rigid
fields:
  - vision
  - graphics
  - human-capture
related:
  - "[[MonoPerfCap (2018)]]"
  - "[[Neural Body (2021)]]"
  - "[[Animatable NeRF (2021)]]"
predecessors:
  - "[[MonoPerfCap (2018)]]"
successors:
  - "[[Neural Body (2021)]]"
  - "[[Animatable NeRF (2021)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**DeepCap** introduced a **weakly-supervised framework** for **dense human performance capture from monocular RGB video**. Unlike previous methods requiring multi-view setups or heavy supervision, DeepCap jointly estimated **human pose** and **non-rigid surface deformation** from single-view input.

# Key Idea
> Achieve dense monocular human performance capture using **weak supervision** by combining parametric body models (pose priors) with learned non-rigid deformation fields.

# Method
- **Input**: Monocular RGB video of a moving person.  
- **Base model**: SMPL parametric human body model for skeletal pose.  
- **Non-rigid deformation field**: Neural network predicts dense surface displacements.  
- **Weak supervision**:  
  - 2D keypoints from pose detectors.  
  - Silhouette constraints.  
  - Photometric consistency.  
- **Output**: Dense 3D mesh reconstruction per frame (pose + deformations).  

# Results
- Achieved high-quality monocular human capture comparable to multi-view systems.  
- Reconstructed dynamic clothing and surface detail.  
- Robust to challenging motions and loose clothing.  

# Why it Mattered
- Pushed human performance capture toward **practical monocular setups**.  
- Reduced reliance on motion-capture studios and markers.  
- Influenced later neural implicit body models (Neural Body, Animatable NeRF).  

# Architectural Pattern
- Parametric skeleton-based body model (SMPL).  
- Neural deformation layer for non-rigid detail.  
- Weakly supervised training losses (keypoints, silhouettes, appearance).  

# Connections
- Builds on **MonoPerfCap (2018)**.  
- Predecessor to implicit neural human models (Neural Body, Animatable NeRF).  
- Related to body modeling + non-rigid deformation.  

# Implementation Notes
- Relies on accurate 2D keypoint detection.  
- Weak supervision avoids need for 3D ground-truth.  
- Inference real-time-ish on GPUs.  

# Critiques / Limitations
- Struggles with severe occlusions.  
- Quality depends on silhouette/pose detector accuracy.  
- Clothing detail limited compared to later NeRF-based methods.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Basics of pose estimation.  
- Parametric models (SMPL).  
- Weak vs strong supervision.  

## Postgraduate-Level Concepts
- Joint optimization of pose + non-rigid deformation.  
- Differentiable silhouette and photometric losses.  
- Trade-offs in monocular 3D reconstruction.  

---

# My Notes
- DeepCap was a **key step bridging classical body models and neural implicit methods**.  
- Showed how weak supervision can scale to in-the-wild monocular performance capture.  
- Open question: Can weak supervision be combined with NeRF-style implicit models for **generalizable human capture**?  
- Possible extension: Use DeepCap priors to initialize **dynamic NeRF or Gaussian splatting** for humans.  

---
