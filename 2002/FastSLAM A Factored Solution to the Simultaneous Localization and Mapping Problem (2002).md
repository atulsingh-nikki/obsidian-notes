---
title: "FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem (2002)"
aliases:
  - FastSLAM
  - Rao-Blackwellized Particle Filter for SLAM
authors:
  - Michael Montemerlo
  - Sebastian Thrun
  - Daphne Koller
  - Ben Wegbreit
year: 2002
venue: "AAAI / Robotics"
doi: ""
citations: 15000+
tags:
  - paper
  - slam
  - robotics
  - particle-filter
  - mapping
fields:
  - robotics
  - control-systems
  - computer-vision
  - probabilistic-inference
related:
  - "[[Particle Filter (1993)]]"
  - "[[Kalman Filter (1960)]]"
  - "[[Extended Kalman Filter (1969)]]"
  - "[[GraphSLAM (2004)]]"
predecessors:
  - "[[Particle Filter (1993)]]"
  - "[[Rao-Blackwellized Particle Filter (2000)]]"
successors:
  - "[[GraphSLAM (2004)]]"
  - "[[Factor Graph SLAM (2000s)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**FastSLAM** (Montemerlo et al., 2002) introduced a **Rao-Blackwellized Particle Filter (RBPF)** for **Simultaneous Localization and Mapping (SLAM)**. It factored the SLAM problem into **robot trajectory estimation via particles** and **landmark mapping via independent EKFs**, making SLAM scalable to large environments.

# Key Idea
> Factorize SLAM:  
> - Use **particles** to represent robot trajectories.  
> - Use **independent EKFs** to estimate landmark positions conditioned on each particle trajectory.  
This drastically reduces complexity compared to monolithic SLAM.  

# Method
- **State representation**:  
  - Robot trajectory: represented by particles.  
  - Landmarks: each particle maintains a set of independent Gaussian landmark estimates.  
- **Rao-Blackwellization**: Marginalizes landmark uncertainty analytically using EKFs, leaving only trajectory uncertainty to be sampled.  
- **Algorithm**:  
  1. Sample robot trajectories (particle filter).  
  2. Update landmarks with EKFs for each particle.  
  3. Resample particles based on measurement likelihood.  

# Results
- Reduced complexity from O(N²) to O(N log M) (N = #landmarks, M = #particles).  
- Demonstrated large-scale SLAM in real-world environments.  
- Became a cornerstone in **probabilistic robotics**.  

# Why it Mattered
- Solved scalability issues of EKF-SLAM.  
- Enabled practical SLAM in robots navigating large maps.  
- Inspired modern **graph-based SLAM** and factor graph methods.  

# Architectural Pattern
- Hybrid: particle filters (for trajectory) + EKFs (for landmarks).  
- Factored representation of uncertainty.  

# Connections
- Successor to **Particle Filters (1993)** and **RBPF (2000)**.  
- Predecessor to **GraphSLAM (2004)** and **Factor Graph SLAM**.  
- Influential in robotics, AR/VR tracking, autonomous navigation.  

# Implementation Notes
- Each particle maintains its own map (computationally expensive for many particles).  
- Requires good resampling to avoid particle depletion.  
- Later optimized with FastSLAM 2.0 (improved proposal distributions).  

# Critiques / Limitations
- Particle set can collapse if too few particles are used.  
- Map duplication across particles is memory-heavy.  
- Performance still degrades with extremely large environments.  

---

# Educational Connections

## Undergraduate-Level Concepts
- SLAM = robot builds a map while localizing itself.  
- Particle filters can track robot path uncertainty.  
- EKFs update landmark positions efficiently.  
- Example: robot exploring a maze, gradually mapping landmarks.  

## Postgraduate-Level Concepts
- Rao-Blackwellization for marginalizing conditional distributions.  
- Complexity reduction in probabilistic inference.  
- FastSLAM vs EKF-SLAM: scalability vs accuracy trade-offs.  
- Extensions: FastSLAM 2.0, GraphSLAM, factor graphs.  

---

# My Notes
- FastSLAM = **particle filters meet SLAM**.  
- Key contribution: factoring trajectory vs landmarks.  
- Open question: In the deep learning era, should SLAM still rely on PF/EKF, or hybridize with learned representations?  
- Possible extension: **Neural FastSLAM** → particle filters with learned map encoders.  

---
