---
title: "Particle Filter: Sequential Monte Carlo Methods (1993)"
aliases:
  - Particle Filter
  - Sequential Monte Carlo (SMC)
  - Bootstrap Filter
authors:
  - Neil Gordon
  - David J. Salmond
  - Adrian F. M. Smith
year: 1993
venue: "IEE Proceedings F (Radar and Signal Processing)"
doi: "10.1049/ip-f-2.1993.0015"
citations: 35000+
tags:
  - paper
  - filtering
  - estimation
  - monte-carlo
  - robotics
fields:
  - control-systems
  - signal-processing
  - robotics
  - computer-vision
related:
  - "[[Kalman Filter (1960)]]"
  - "[[Extended Kalman Filter (1969)]]"
  - "[[Unscented Kalman Filter (1997)]]"
  - "[[Bayesian Filtering]]"
predecessors:
  - "[[Kalman Filter (1960)]]"
  - "[[Extended Kalman Filter (1969)]]"
successors:
  - "[[Rao-Blackwellized Particle Filter (2000)]]"
  - "[[Factor Graph SLAM (2000s)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
The **Particle Filter (Gordon et al., 1993)** introduced **Sequential Monte Carlo (SMC)** methods for state estimation in **nonlinear, non-Gaussian systems**. Unlike Kalman-family filters, it represents probability distributions with **particles (samples)** instead of Gaussian assumptions.

# Key Idea
> Approximate the posterior distribution of states with a set of **weighted particles**, updating them recursively via sampling, importance weighting, and resampling.

# Method
- **State-space model**:  
  - State: $x_k = f(x_{k-1}, u_k) + w_k$  
  - Measurement: $z_k = h(x_k) + v_k$  
- **Algorithm**:  
  1. **Prediction**: Propagate particles via dynamics $f$.  
  2. **Update**: Weight particles by likelihood under measurement $h$Particle Filter: Sequential Monte Carlo Methods (1993).  
  3. **Resampling**: Replace low-weight particles with high-weight ones.  
- **Output**: Empirical posterior distribution of states.  

# Results
- Handles arbitrary nonlinearities and noise distributions.  
- Outperforms EKF/UKF in strongly nonlinear and multimodal cases.  
- Became the foundation of **Monte Carlo state estimation**.  

# Why it Mattered
- First practical Bayesian filter without Gaussian assumptions.  
- Enabled robust tracking, robotics localization, and SLAM.  
- Opened the door to **Monte Carlo methods in online estimation**.  

# Architectural Pattern
- Recursive Bayesian filter with Monte Carlo approximation.  
- Resampling to prevent particle degeneracy.  

# Connections
- Successor to **Kalman (1960)**, **EKF (1969)**, **UKF (1997)**.  
- Basis of **Rao-Blackwellized Particle Filter (RBPF, 2000)** and **FastSLAM (2002)**.  
- Related to modern probabilistic robotics and vision tracking.  

# Implementation Notes
- Complexity: O(N) per step (N = #particles).  
- Trade-off: more particles → better approximation, higher compute cost.  
- Sensitive to sample impoverishment (solved via advanced resampling).  

# Critiques / Limitations
- Computationally expensive for high-dimensional states.  
- Requires many particles to approximate distributions well.  
- Resampling may lose diversity in particle set.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Kalman filters assume Gaussian noise; particle filters don’t.  
- Uses many random “particles” to represent possible states.  
- Example: tracking a person walking through a building with noisy sensors.  

## Postgraduate-Level Concepts
- Sequential Monte Carlo theory.  
- Importance sampling and weight degeneracy.  
- High-dimensional curse of dimensionality.  
- Applications in SLAM, Bayesian tracking, probabilistic programming.  

---

# My Notes
- Particle Filter = **Bayesian filtering without Gaussian crutches**.  
- Hugely influential in robotics and tracking.  
- Open question: How to make particle methods scalable to high-dimensional problems (images, deep latent states)?  
- Possible extension: Neural Particle Filters → combining deep generative models with SMC.  

---
