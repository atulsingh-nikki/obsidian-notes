---
title: "GraphSLAM: Simultaneous Localization and Mapping via Graph Optimization (2004)"
aliases:
  - GraphSLAM
  - SLAM as Graph Optimization
authors:
  - Sebastian Thrun
  - Wolfram Burgard
  - Dieter Fox
year: 2004
venue: "Book Chapter: Probabilistic Robotics / Early Conference Papers"
doi: ""
citations: 20000+
tags:
  - paper
  - slam
  - robotics
  - factor-graph
  - optimization
fields:
  - robotics
  - computer-vision
  - probabilistic-inference
related:
  - "[[Particle Filter (1993)]]"
  - "[[Factor Graph SLAM (2000s)]]"
  - "[[A New Approach to Linear Filtering and Prediction Problems (1960)|Kalman Filter]]"
  - "[[Extended Kalman Filter (EKF, 1969)]]"
  - "[[FastSLAM A Factored Solution to the Simultaneous Localization and Mapping Problem (2002)|FastSLAM]]"
predecessors:
  - "[[FastSLAM (2002)]]"
successors:
  - "[[Factor Graph SLAM (2000s)]]"
  - "[[GTSAM (2010s)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
**GraphSLAM (Thrun et al., 2004)** introduced a new perspective on SLAM by formulating it as a **graph optimization problem**. Robot poses are nodes, constraints (odometry, landmark observations) are edges. SLAM is solved by finding the most likely configuration of the graph that satisfies all constraints.

# Key Idea
> Represent SLAM as a **graph of constraints** between robot poses and landmarks. Solve for the configuration that best satisfies these constraints, turning SLAM into a nonlinear least-squares optimization problem.

# Method
- **Graph representation**:  
  - Nodes: robot poses and landmarks.  
  - Edges: spatial constraints (odometry, observations).  
- **Optimization**:  
  - Formulate SLAM as a nonlinear least-squares problem.  
  - Solve with iterative optimization methods (Gauss–Newton, Levenberg–Marquardt).  
- **Back-end / Front-end split**:  
  - Front-end: builds the graph (data association, sensor processing).  
  - Back-end: optimizes the graph (pose graph optimization).  

# Results
- Unified SLAM under a flexible, general framework.  
- Scaled better than EKF-SLAM for large environments.  
- Improved robustness to loop closure constraints.  

# Why it Mattered
- Shifted SLAM from **filtering** (EKF, Particle Filters) to **optimization**.  
- Foundation of modern SLAM systems (pose graph optimization, factor graphs).  
- Still the backbone of libraries like **GTSAM**, **Ceres**, and modern visual SLAM.  

# Architectural Pattern
- Factor graph / constraint graph.  
- Nonlinear least-squares optimization.  

# Connections
- Predecessor: **FastSLAM (2002)** (particle + EKF factorization).  
- Successor: **Factor Graph SLAM (2000s)** and **GTSAM (2010s)**.  
- Influential in **Visual SLAM (ORB-SLAM, LSD-SLAM, DSO)**.  

# Implementation Notes
- Graph optimization can be expensive for very large maps (sparse matrix techniques help).  
- Loop closure detection critical for accuracy.  
- Requires robust data association.  

# Critiques / Limitations
- Assumes Gaussian measurement models.  
- Sensitive to incorrect loop closures.  
- Optimization may converge to local minima.  

---

# Educational Connections

## Undergraduate-Level Concepts
- SLAM = robot localizes itself while mapping environment.  
- Graph idea: robot poses = dots, constraints = springs; optimize spring tensions to find consistent map.  
- Example: robot drives in a square and returns → loop closure improves accuracy.  

## Postgraduate-Level Concepts
- Nonlinear least-squares optimization.  
- Factor graphs vs pose graphs.  
- Front-end (data association) vs back-end (optimization).  
- Extensions: robust optimization, incremental solvers (iSAM).  

---

# My Notes
- GraphSLAM = **the unifying language of SLAM**.  
- Represents the field’s shift from **filters → optimization**.  
- Open question: Can we integrate **deep features and learned priors** into graph-SLAM frameworks?  
- Possible extension: Hybrid **Neural Factor Graphs** for SLAM with semantic priors.  

---
