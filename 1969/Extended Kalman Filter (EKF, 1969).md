---
title: "Extended Kalman Filter (EKF, 1969)"
aliases:
  - EKF
  - Nonlinear Kalman Filter
authors:
  - Stanley F. Schmidt
  - Rudolf E. Kalman (influence)
year: 1969
venue: "NASA Technical Report / Control Theory Literature"
doi: ""
citations: 25000+
tags:
  - paper
  - filtering
  - estimation
  - control-theory
  - robotics
fields:
  - control-systems
  - signal-processing
  - robotics
  - computer-vision
related:
  - "[[Kalman Filter (1960)]]"
  - "[[Unscented Kalman Filter (UKF, 1997)]]"
  - "[[Particle Filter (1993)]]"
predecessors:
  - "[[Kalman Filter (1960)]]"
successors:
  - "[[Unscented Kalman Filter (1997)]]"
  - "[[Particle Filter (1993)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
The **Extended Kalman Filter (EKF)** generalized the **Kalman Filter** to **nonlinear systems** by applying **first-order Taylor expansions (linearization)** of nonlinear dynamics and measurement functions. It became the standard estimator in **robotics, navigation, and SLAM**.

# Key Idea
> Approximate nonlinear state transitions and measurements with **linearizations (Jacobians)**, then apply the Kalman filter equations using these approximations.

# Method
- **System model**:  
  - State: `x_k = f(x_{k-1}, u_k) + w_k`  
  - Measurement: `z_k = h(x_k) + v_k`  
- **Linearization**:  
  - Compute Jacobians of `f` and `h` around current estimate.  
  - Use Jacobians in place of A, H matrices of the standard Kalman filter.  
- **Two-step recursion**:  
  1. **Prediction**: propagate state via nonlinear `f`.  
  2. **Update**: correct with measurement via nonlinear `h`.  

# Results
- Enabled Kalman filtering in nonlinear domains (aerospace, robotics).  
- Widely used in spacecraft navigation, GPS/INS fusion, SLAM.  
- Remains a standard baseline for nonlinear estimation.  

# Why it Mattered
- Extended Kalman filter theory from linear to nonlinear real-world systems.  
- Enabled practical robotics and aerospace applications.  
- Provided the first scalable nonlinear Bayesian estimator.  

# Architectural Pattern
- Recursive Bayesian estimator with linearization.  
- Uses Jacobians at each step.  

# Connections
- Predecessor: **Kalman Filter (1960)**.  
- Successors: **Unscented Kalman Filter (UKF, 1997)**, **Particle Filter (1993)**.  
- Still used in modern robotics pipelines (e.g., SLAM).  

# Implementation Notes
- Requires differentiable `f` and `h`.  
- Jacobian computation can be costly.  
- Sensitive to poor linearization (large errors if dynamics highly nonlinear).  

# Critiques / Limitations
- Only first-order approximation; accuracy degrades for strongly nonlinear systems.  
- Can diverge if initial guess poor or noise large.  
- Less robust than sampling-based methods (Particle Filters).  

---

# Educational Connections

## Undergraduate-Level Concepts
- Kalman filter assumes linear systems; EKF extends to nonlinear.  
- Uses **linear approximation (Taylor expansion)**.  
- Example: estimating a robot’s position with nonlinear motion + noisy sensors.  

## Postgraduate-Level Concepts
- Jacobians in filtering.  
- Error covariance propagation under nonlinear transformations.  
- Trade-off: computational efficiency vs approximation error.  
- Extensions: UKF (sigma points), Particle Filters.  

---

# My Notes
- EKF = **Kalman filter for the real world**, since most systems are nonlinear.  
- Still ubiquitous in robotics, even with deep learning alternatives.  
- Open question: How to blend **neural dynamics models** with EKF-like filtering?  
- Possible extension: differentiable EKFs, hybrid deep-learning + classical filtering.  

---
