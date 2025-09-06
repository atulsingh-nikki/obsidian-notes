---
title: "Learning to Solve Hard Minimal Problems (2022)"
aliases:
  - Hard Minimal Problems Solver
  - Pick & Solve Minimal Problems
authors:
  - Viktor Larsson
  - Zuzana Kukelova
  - Tomas Pajdla
  - Marc Pollefeys
year: 2022
venue: "CVPR (Best Paper Award)"
doi: "10.1109/CVPR52688.2022.00536"
arxiv: "https://arxiv.org/abs/2205.00362"
code: "https://github.com/vlarsson/CVPR22-HardMinimal"
citations: 250+
dataset:
  - Synthetic benchmarks for geometric solvers
  - Real geometric vision tasks (camera pose, multi-view geometry)
tags:
  - paper
  - geometry
  - minimal-problems
  - homotopy
  - machine-learning
fields:
  - vision
  - 3d-reconstruction
  - robotics
  - geometry
related:
  - "[[Pixel-Perfect SfM with Featuremetric Refinement (2021)]]"
  - "[[Viewing Graph Solvability (2021)]]"
predecessors:
  - "[[Classical Minimal Solvers in Geometric Vision]]"
successors:
  - "[[Learning-Accelerated Geometry Solvers (2023+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
This paper addressed the challenge of **solving hard minimal problems** in 3D geometric vision, such as estimating camera pose from minimal correspondences. Traditional algebraic solvers for such problems are slow and brittle. The authors proposed a **"pick & solve" method** combining machine learning and **homotopy continuation**, dramatically accelerating solver performance.

# Key Idea
> Use a **learned policy** to pick promising solution candidates, then refine them with **homotopy continuation**, reducing the complexity of solving minimal problems in 3D geometry.

# Method
- **Minimal problems**: Hard polynomial systems (e.g., pose estimation, camera geometry).  
- **Traditional approach**: Gröbner basis solvers (exact but slow).  
- **Proposed method**:  
  1. Use ML to **predict promising solution paths**.  
  2. Apply **homotopy continuation** to track solutions efficiently.  
- **Pick & Solve**: Not all candidate paths are needed — only the most likely.  

# Results
- Achieved **order-of-magnitude speedup** compared to classical solvers.  
- Maintained or improved accuracy on geometric benchmarks.  
- Demonstrated scalability to real SfM and camera pose problems.  

# Why it Mattered
- Brought **machine learning into the core of geometric vision solvers**.  
- Made previously impractical minimal solvers usable in practice.  
- Advanced both theoretical and applied 3D reconstruction.  

# Architectural Pattern
- ML predictor + homotopy continuation.  
- "Pick & solve" strategy for polynomial root finding.  

# Connections
- Related to **SfM, multi-view geometry, camera calibration**.  
- Complements **Pixel-Perfect SfM (2021)** and **Viewing Graph Solvability (2021)**.  
- Successor to classical minimal solvers in vision.  

# Implementation Notes
- Integrates into existing SfM pipelines.  
- Requires training ML predictor for each problem type.  
- Code available open-source.  

# Critiques / Limitations
- Needs retraining for each new minimal problem.  
- Still sensitive to noise in some configurations.  
- ML predictor adds extra complexity vs purely algebraic methods.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What are **minimal problems** in geometry (e.g., 5-point essential matrix).  
- Why solving polynomial equations is needed in 3D reconstruction.  
- Basics of camera pose estimation and structure-from-motion.  
- Difference between brute-force solvers vs learned guidance.  

## Postgraduate-Level Concepts
- Gröbner basis methods vs homotopy continuation.  
- How ML can guide numerical solvers for polynomial systems.  
- Trade-offs between exactness and efficiency in geometry solvers.  
- Implications for real-time SfM, SLAM, and robotics.  

---

# My Notes
- A **landmark paper**: merged ML with hardcore geometry in a principled way.  
- Homotopy continuation is powerful but was underused in vision until this.  
- Open question: Can we design **generalizable pick & solve models** that adapt across many minimal problems?  
- Possible extension: Combine solver acceleration with **differentiable geometry pipelines** for end-to-end learning.  

---
