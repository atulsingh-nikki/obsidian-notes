---
title: "iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree (2012)"
aliases:
  - iSAM2
  - Bayes Tree SLAM
authors:
  - Michael Kaess
  - Hordur Johannsson
  - Richard Roberts
  - Viorela Ila
  - John J. Leonard
  - Frank Dellaert
year: 2012
venue: "International Journal of Robotics Research (IJRR)"
doi: "10.1177/0278364911430419"
citations: 12000+
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
  - "[[iSAM (2007)]]"
  - "[[GTSAM (2010s)]]"
predecessors:
  - "[[iSAM (2007)]]"
successors:
  - "[[GTSAM (2010s)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**iSAM2** (Kaess et al., 2012) extended **iSAM (2007)** by introducing the **Bayes tree** data structure, which enables more efficient incremental updates, relinearization, and variable reordering. It became the standard back-end for modern SLAM libraries such as **GTSAM**.

# Key Idea
> Use a **Bayes tree** (a tree-structured factorization of the factor graph) to efficiently update and relinearize variables incrementally, while preserving sparsity and scalability.

# Method
- **Bayes tree**: A tree structure representing conditional dependencies from factor graph elimination.  
- **Incremental updates**: Add new factors → update only affected subtree in Bayes tree.  
- **Relinearization**: Selectively relinearize only variables with significant changes.  
- **Variable reordering**: Dynamically reorder variables to maintain sparsity.  

# Results
- Achieved faster, more scalable incremental optimization than iSAM.  
- Supported **real-time SLAM** in large-scale environments.  
- Provided theoretical foundation for robust incremental inference.  

# Why it Mattered
- iSAM2 became the **standard SLAM back-end** in robotics research.  
- Enabled highly efficient factor graph optimization.  
- Widely adopted via **GTSAM**, influencing both academia and industry.  

# Architectural Pattern
- Factor graph → elimination → Bayes net → Bayes tree.  
- Incremental updates localized to affected subtree.  

# Connections
- Successor to **iSAM (2007)**.  
- Implemented in **GTSAM (Georgia Tech Smoothing And Mapping)**.  
- Basis of modern SLAM systems (ORB-SLAM, VINS-Mono, Kimera).  

# Implementation Notes
- Exploits sparse linear algebra.  
- Dynamic relinearization improves accuracy.  
- Bayes tree supports efficient marginalization queries.  

# Critiques / Limitations
- Still assumes Gaussian noise.  
- Requires robust loop closure detection in front-end.  
- Optimization can stall under poor initializations.  

---

# Educational Connections

## Undergraduate-Level Concepts
- SLAM graph = nodes (poses/landmarks), edges (measurements).  
- iSAM2 uses a **tree** to organize constraints and update efficiently.  
- Analogy: instead of re-solving an entire puzzle, just adjust the affected pieces.  

## Postgraduate-Level Concepts
- Bayes tree derivation from variable elimination.  
- Selective relinearization strategies.  
- Complexity trade-offs vs iSAM (QR factorization).  
- Use in factor graph SLAM libraries (GTSAM).  

---

# My Notes
- iSAM2 = **practical backbone of modern SLAM back-ends**.  
- Key insight: Bayes tree as a dynamic data structure for SLAM inference.  
- Open question: How to adapt iSAM2-style incremental solvers for **probabilistic deep learning pipelines**?  
- Possible extension: Differentiable Bayes trees + neural factors for learned SLAM.  

---
