---
title: "Total Capture: A 3D Deformation Model for Tracking Faces, Hands, and Bodies (2018)"
aliases: 
  - Total Capture
  - 3D Deformation Model for Human Tracking
authors:
  - Hanbyul Joo
  - Tomas Simon
  - Yaser Sheikh
year: 2018
venue: "CVPR"
doi: "10.1109/CVPR.2018.00187"
arxiv: "https://arxiv.org/abs/1801.01615"
code: "http://www.cs.cmu.edu/~hanbyulj/totalcapture/"  
citations: 1500+
dataset:
  - Panoptic Studio Dataset
tags:
  - paper
  - human-pose
  - motion-capture
  - 3d-tracking
fields:
  - vision
  - graphics
  - human-pose-estimation
related:
  - "[[SMPL Model (2015)]]"
  - "[[DensePose (2018)]]"
predecessors:
  - "[[SMPL (2015)]]"
  - "[[OpenPose (2016)]]"
successors:
  - "[[FrankMocap (2021)]]"
  - "[[MetaHuman Tracking Models]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
This paper introduced **Total Capture**, a unified **3D deformation model** capable of tracking the **face, hands, and body simultaneously** from multi-view video data. It leveraged a **large-scale Panoptic Studio dataset** to learn realistic human motions and interactions.

# Key Idea
> Extend parametric body models (like SMPL) into a **holistic model** that jointly represents and tracks the **body, hands, and face** in 3D.

# Method
- Built a **3D deformation model** combining body, hand, and face representations into a single framework.  
- Trained using the **Panoptic Studio dataset**, which provides rich multi-view recordings of social interactions.  
- Optimized pose and shape by fitting the model to multi-view detections.  
- Incorporated facial expressions and hand articulations for full-body performance capture.  

# Results
- First method to capture **full human pose (body + face + hands)** in 3D from video.  
- Achieved robust tracking across multiple subjects and interactions.  
- Demonstrated applications in motion capture, VR/AR, and social signal analysis.  

# Why it Mattered
- Marked a milestone in **holistic human performance capture**.  
- Went beyond body-only models (SMPL) by integrating **fine details of hands and face**.  
- Enabled research in human communication, behavior analysis, and immersive technologies.  

# Architectural Pattern
- Parametric deformable model (extension of SMPL).  
- Multi-view optimization and fitting.  
- Unified representation for body, hands, and face.  

# Connections
- **Contemporaries**: DensePose (2018), Human3.6M-based pose estimators.  
- **Influence**: FrankMocap, MetaHuman technologies, full-body VR avatars.  

# Implementation Notes
- Requires multi-view input; not monocular.  
- Computationally heavy due to optimization.  
- Panoptic Studio dataset central to training and validation.  

# Critiques / Limitations
- Not real-time; slower optimization-based approach.  
- Multi-camera setup required; monocular generalization is limited.  
- Hand and face detail not as refined as specialized models.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1801.01615)  
- [Project page](http://www.cs.cmu.edu/~hanbyulj/totalcapture/)  
- [Panoptic Studio dataset](http://domedb.perception.cs.cmu.edu/)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Deformation models, PCA for shape representation.  
- **Probability & Statistics**: Optimization under uncertainty.  
- **Geometry**: 3D rigid and non-rigid transformations.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Parametric models integrated with detection.  
- **Computer Vision**: Multi-view geometry, human pose estimation.  
- **Research Methodology**: Large-scale dataset-driven model design.  
- **Advanced Optimization**: Joint fitting of high-dimensional deformable models.  

---

# My Notes
- Connects strongly to **multi-person editing workflows** in video.  
- Open question: How can **monocular video-based methods** reach Panoptic Studio quality?  
- Possible extension: Use diffusion priors for **face/hand detail refinement in video mocap**.  

---
