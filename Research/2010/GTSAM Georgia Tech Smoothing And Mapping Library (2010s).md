---
title: "GTSAM: Georgia Tech Smoothing And Mapping Library (2010s)"
aliases:
  - GTSAM
  - Georgia Tech Smoothing and Mapping
authors:
  - Frank Dellaert
  - Michael Kaess
  - et al.
year: 2010s (first public release ~2012)
venue: "Open-Source Library / IJRR companion papers"
doi: ""
citations: 7000+
tags:
  - library
  - slam
  - factor-graph
  - optimization
  - robotics
fields:
  - robotics
  - computer-vision
  - probabilistic-inference
related:
  - "[[GraphSLAM (2004)]]"
  - "[[iSAM (2007)]]"
  - "[[iSAM2 (2012)]]"
  - "[[Factor Graph SLAM (2000s)]]"
predecessors:
  - "[[iSAM2 (2012)]]"
successors:
  - "[[Kimera (2019)]]"
  - "[[Modern VSLAM Systems (ORB-SLAM, VINS-Mono)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**GTSAM (Georgia Tech Smoothing And Mapping)** is an open-source C++ library for **factor graph–based optimization in SLAM, structure-from-motion (SfM), and robotics applications**. It unified GraphSLAM, iSAM, and iSAM2 into a practical software toolkit, enabling large-scale real-time inference.

# Key Idea
> Provide a general-purpose **factor graph framework** with efficient solvers (batch + incremental) for robotics and vision applications, exposing both probabilistic and optimization perspectives.

# Features
- **Factor Graph Abstraction**: Variables (poses, landmarks) + factors (constraints).  
- **Solvers**:  
  - Batch optimization (Gauss–Newton, Levenberg–Marquardt).  
  - Incremental optimization via **iSAM / iSAM2**.  
- **Data Structures**: Bayes nets, Bayes trees for efficient inference.  
- **Applications**: SLAM, bundle adjustment, calibration, visual-inertial odometry.  

# Results
- Widely used in academia and industry as a **SLAM back-end library**.  
- Supported benchmarks in robotics and SfM.  
- Enabled research reproducibility and rapid prototyping.  

# Why it Mattered
- Made factor graph methods accessible beyond experts.  
- Served as the backbone for many SLAM systems (Kimera, ORB-SLAM variants, VINS-Mono back-ends).  
- Provided a bridge between **theory (probabilistic inference)** and **practice (efficient robotics systems)**.  

# Architectural Pattern
- Factor graph representation.  
- Gaussian factorization (sparse linear algebra).  
- Incremental Bayes tree updates (iSAM2).  

# Connections
- Built on **GraphSLAM (2004)**, **iSAM (2007)**, and **iSAM2 (2012)**.  
- Successor frameworks: **Kimera (2019)** (semantic + 3D SLAM).  
- Related to SfM libraries (Ceres, COLMAP).  

# Implementation Notes
- Written in C++ with Python bindings.  
- Actively maintained by Georgia Tech + open-source contributors.  
- Used in robotics, AR/VR, and autonomous systems.  

# Critiques / Limitations
- Assumes Gaussian noise (like most factor graph SLAM).  
- Requires robust **front-end** (feature extraction, loop closure).  
- Steep learning curve compared to higher-level SLAM frameworks.  

---

# Educational Connections

## Undergraduate-Level Concepts
- SLAM can be written as a **graph problem**.  
- GTSAM = library that solves these graphs efficiently.  
- Example: robot mapping a house → constraints = walls, doors, odometry.  

## Postgraduate-Level Concepts
- Factor graph formulations in probabilistic inference.  
- Sparse linear algebra for large-scale optimization.  
- Incremental vs batch solvers in SLAM.  
- Using GTSAM for bundle adjustment in structure-from-motion.  

---

# My Notes
- GTSAM = **the PyTorch of SLAM back-ends**.  
- Standardized research; almost eveGTSAM: Georgia Tech Smoothing And Mapping Library (2010s)ry modern SLAM system cites or uses it.  
- Open question: Can GTSAM evolve to support **learned factors** and hybrid deep-learning + optimization pipelines?  
- Possible extension: Differentiable GTSAM for end-to-end visual-inertial learning.  

---
