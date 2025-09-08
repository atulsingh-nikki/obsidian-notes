---
title: "Efficient Globally Optimal Consensus Maximisation with Tree Search (2016)"
aliases: 
  - Consensus Maximisation Tree Search
  - Tóth16 Tree Search
authors:
  - Dániel Baráth
  - Tat-Jun Chin
  - Frank C. Park
  - Anders Eriksson
year: 2016
venue: "ECCV"
doi: "10.1007/978-3-319-46475-6_10"
arxiv: "https://arxiv.org/abs/1606.04820"
code: "https://github.com/danini/Consensus-Maximization-TreeSearch" # (if available/unofficial)
citations: 300+
dataset:
  - Synthetic correspondences
  - Real-world geometric matching datasets
tags:
  - paper
  - optimization
  - robust-estimation
  - ransac
fields:
  - vision
  - optimization
  - geometry
related:
  - "[[RANSAC]]"
  - "[[Branch-and-Bound for Model Fitting]]"
predecessors:
  - "[[RANSAC (Fischler & Bolles, 1981)]]"
  - "[[LO-RANSAC]]"
successors:
  - "[[GORE (Globally-Optimal Rigid Estimation)]]"
  - "[[MAGSAC]]"
impact: ⭐⭐⭐⭐☆
status: "to-read"
---

# Summary
This work introduces a **tree search method for consensus maximisation** in robust model fitting problems. Unlike RANSAC, which provides approximate solutions, the proposed approach guarantees **globally optimal solutions** to consensus maximisation while maintaining efficiency through pruning strategies.

# Key Idea
> Solve consensus maximisation exactly using tree search with pruning, achieving global optimality more efficiently than naive branch-and-bound.

# Method
- Problem: **Consensus maximisation** — given correspondences with outliers, find model parameters that maximize the number of inliers.  
- Uses a **tree search strategy** to explore solution space.  
- Introduces **bounding and pruning functions** to cut branches of the search tree that cannot improve the solution.  
- Provides guarantees of **global optimality** while being more efficient than classical branch-and-bound.  

# Results
- Outperforms branch-and-bound baselines for geometric model fitting tasks.  
- Works on problems like fundamental matrix estimation, homography estimation, and rigid alignment.  
- Demonstrated **significant runtime improvements** over naive global solvers.  

# Why it Mattered
- Brought **global optimality** into practical reach for robust estimation tasks in computer vision.  
- Showed that exact consensus maximisation can be tractable with efficient search strategies.  
- Influenced later globally optimal methods (e.g., GORE, MAGSAC, deterministic solvers).  

# Architectural Pattern
- **Tree search + bounding functions**.  
- Hybrid between **branch-and-bound** and **consensus maximisation**.  
- Deterministic, globally optimal solver (contrast with stochastic RANSAC).  

# Connections
- **Contemporaries**: MLESAC, PROSAC, LO-RANSAC.  
- **Influence**: GORE (2019), deterministic robust solvers, differentiable RANSAC variants.  

# Implementation Notes
- Computational complexity higher than RANSAC for very high-dimensional problems.  
- Works well when number of parameters and data size are moderate.  
- Needs carefully designed bounding functions for pruning efficiency.  

# Critiques / Limitations
- Not as scalable as stochastic RANSAC for very large datasets.  
- Still slower than approximate methods in real-time settings.  
- Performance tied to quality of pruning heuristics.  

# Repro / Resources
- [Paper link (ECCV 2016)](https://arxiv.org/abs/1606.04820)  
- [Code repo (unofficial)](https://github.com/danini/Consensus-Maximization-TreeSearch)  
- Related tutorials: robust geometric estimation surveys.  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Solving systems of equations (e.g., fundamental matrix, homography).  
- **Probability & Statistics**: Outlier handling, consensus measures.  
- **Optimization Basics**: Search strategies, greedy vs global optima.  
- **Data Structures**: Trees for search exploration.  

## Postgraduate-Level Concepts
- **Numerical Methods**: Branch-and-bound, global optimisation.  
- **Machine Learning Theory**: Robustness and generalisation in model fitting.  
- **Computer Vision**: Geometric model estimation with outliers.  
- **Research Methodology**: Trade-offs between exactness and effici
