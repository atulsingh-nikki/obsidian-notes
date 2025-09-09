---
title: "A New ApproacA New Approach to Linear Filtering and Prediction Problems (1960)h to Linear Filtering and Prediction Problems (1960)"
aliases:
  - Kalman Filter
  - Linear Quadratic Estimator (LQE)
authors:
  - Rudolf E. Kalman
year: 1960
venue: "Journal of Basic Engineering"
doi: "10.1115/1.3662552"
citations: 100000+
tags:
  - paper
  - filtering
  - estimation
  - control-theory
  - foundational
fields:
  - control-systems
  - signal-processing
  - computer-vision
  - robotics
related:
  - "[[Extended Kalman Filter (EKF, 1969)]]"
  - "[[Particle Filter (1993)]]"
  - "[[Bayesian Filtering]]"
predecessors:
  - "[[Wiener Filter (1949)]]"
successors:
  - "[[Extended Kalman Filter (1969)]]"
  - "[[Unscented Kalman Filter (1997)]]"
  - "[[Particle Filter (1993)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Rudolf E. Kalman (1960)** introduced the **Kalman Filter**, a recursive algorithm for **optimal linear filtering and prediction** of dynamic systems with Gaussian noise. It became a cornerstone in **control theory, signal processing, robotics, and computer vision**.

# Key Idea
> Maintain a recursive estimate of a system’s state by combining **model-based predictions** with **noisy measurements**, minimizing the mean squared error under linear-Gaussian assumptions.

# Method
- **System model**:  
  - State transition: `x_k = A x_{k-1} + B u_k + w_k`  
  - Measurement: `z_k = H x_k + v_k`  
  - Noise: Gaussian `w_k ~ N(0, Q)`, `v_k ~ N(0, R)`  
- **Two-step recursion**:  
  1. **Prediction**: project state and covariance forward.  
  2. **Update**: correct prediction using new measurement and **Kalman gain**.  
- **Optimality**: Minimizes mean squared error (linear + Gaussian).  

# Results
- Recursive, efficient (doesn’t require storing full history).  
- Provides uncertainty estimates (covariance).  
- Outperformed previous batch estimation (Wiener filter).  

# Why it Mattered
- Revolutionized estimation in engineering and control.  
- Powered aerospace navigation (Apollo, satellites, aircraft).  
- Still fundamental in robotics, tracking, and vision.  

# Architectural Pattern
- Recursive Bayesian estimator.  
- Prediction–correction loop.  

# Connections
- Predecessor: **Wiener filter** (non-recursive, frequency domain).  
- Successors: **Extended Kalman Filter (EKF, nonlinear)**, **Unscented Kalman Filter (UKF)**, **Particle Filter**.  
- Widely used in **object tracking, SLAM, AR/VR, robotics**.  

# Implementation Notes
- Efficient: O(n³) per step (matrix ops).  
- Requires system model matrices (A, B, H, Q, R).  
- Assumes linear-Gaussian; breaks down otherwise.  

# Critiques / Limitations
- Assumes **linearity** and **Gaussian noise**.  
- Model mismatch degrades performance.  
- Nonlinear / multimodal cases require EKF, UKF, or Particle Filters.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Tracking = combining predictions with measurements.  
- Why averaging isn’t enough → need uncertainty weighting.  
- Example: estimating car position with noisy GPS and motion model.  

## Postgraduate-Level Concepts
- Derivation from Bayesian filtering.  
- Matrix Riccati equations for covariance update.  
- Optimality proof under linear-Gaussian assumptions.  
- Extensions: EKF, UKF, Particle Filter, factor graphs (SLAM).  

---

# My Notes
- Kalman Filter = **the backbone of estimation theory**.  
- Elegant: recursive, uncertainty-aware, mathematically optimal.  
- Open question: How to integrate **Kalman-like uncertainty estimation** into modern deep learning pipelines (e.g., differentiable filters)?  
- Possible extension: Deep Kalman Filters (DKF), neural state-space models.  

---
