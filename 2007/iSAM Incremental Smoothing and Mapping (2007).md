---
title: "iSAM: Incremental Smoothing and Mapping (2007)"
aliases:
  - iSAM
  - Incremental SLAM
authors:
  - Michael Kaess
  - Ananth Ranganathan
  - Frank Dellaert
year: 2007
venue: "IEEE Transactions on Robotics"
doi: "10.1109/TRO.2007.907017"
citations: 10000+
tags:
  - paper
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
  - "[[Factor Graph SLAM (2000s)]]"
  - "[[GTSAM (2010s)]]"
predecessors:
  - "[[GraphSLAM (2004)]]"
successors:
  - "[[iSAM2 (2012)]]"
  - "[[GTSAM (2010s)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**iSAM (Kaess, Ranganathan, Dellaert, 2007)** introduced **Incremental Smoothing and Mapping**, enabling **real-time SLAM** by updating the solution incrementally rather than re-solving from scratch when new data arrives. It revolutionized practical SLAM back-ends.

# Key Idea
> Represent SLAM as a **factor graph** and solve it with **incremental updates** to the sparse matrix system, avoiding expensive batch re-optimization.

# Method
- **Factor graph formulation**:  
  - Variables = robot poses, landmarks.  
  - Factors = odometry and measurement constraints.  
- **Incremental updates**:  
  - New measurements → add new factors.  
  - Update solution incrementally via sparse matrix factorization.  
- **Smoothing perspective**: Keeps entire trajectory in graph (not just filtering).  

# Results
- Achieved **real-time SLAM** in large-scale environments.  
- Orders of magnitude faster than batch GraphSLAM.  
- Widely adopted in robotics (mobile robots, visual SLAM).  

# Why it Mattered
- Brought **factor graph optimization** into real-time robotics.  
- Practical bridge between probabilistic theory and robot implementation.  
- Foundation for later tools like **iSAM2** and **GTSAM**.  

# Architectural Pattern
- Factor graph + incremental sparse optimization.  
- Smoothing-based SLAM (trajectory + map jointly estimated).  

# Connections
- Successor to **GraphSLAM (2004)**.  
- Predecessor to **iSAM2 (2012)** and **GTSAM (2010s)**.  
- Used in **ORB-SLAM**, **LSD-SLAM**, and many modern VSLAM systems.  

# Implementation Notes
- Relies on sparse matrix techniques (QR factorization).  
- Incremental updates keep computation bounded.  
- Memory efficient relative to batch GraphSLAM.  

# Critiques / Limitations
- Sensitive to linearization errors (common in nonlinear optimization).  
- Still assumes Gaussian noise.  
- iSAM2 later improved consistency and efficiency with Bayes trees.  

---

# Educational Connections

## Undergraduate-Level Concepts
- GraphSLAM = solving all constraints at once; iSAM = solving incrementally.  
- Analogy: instead of rewriting an essay every time, just edit the new sentence.  
- Example: robot exploring a building → update map with each new corridor scan.  

## Postgraduate-Level Concepts
- Factor graph optimization with QR factorization.  
- Smoothing vs filtering trade-offs in SLAM.  
- Incremental sparse matrix updates.  
- Relation to modern back-ends (Ceres, GTSAM, iSAM2).  

---

# My Notes
- iSAM = **GraphSLAM in real-time**.  
- Made factor graph SLAM usable outside labs.  
- Open question: How to merge iSAM-style incremental solvers with **learned front-ends** (deep features, learned loop closures)?  
- Possible extension: Differentiable iSAM inside end-to-end SLAM pipelines.  

---
