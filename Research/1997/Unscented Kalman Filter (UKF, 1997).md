---
title: "Unscented Kalman Filter (UKF, 1997)"
aliases:
  - UKF
  - Sigma-Point Kalman Filter
authors:
  - Simon J. Julier
  - Jeffrey K. Uhlmann
year: 1997
venue: "IEEE Aerospace Conference Proceedings"
doi: "10.1109/AERO.1997.599611"
citations: 20000+
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
  - "[[Extended Kalman Filter (1969)]]"
  - "[[Particle Filter (1993)]]"
predecessors:
  - "[[Extended Kalman Filter (1969)]]"
successors:
  - "[[Particle Filter (1993)]]"
  - "[[Ensemble Kalman Filter (2000)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
The **Unscented Kalman Filter (UKF)** improves on the **Extended Kalman Filter (EKF)** by avoiding explicit linearization. Instead, it uses the **unscented transform**: a deterministic sampling of “sigma points” that capture the mean and covariance of the state distribution, propagated through the nonlinear functions.

# Key Idea
> Approximate the distribution (not the nonlinear function) by propagating carefully chosen sigma points through the nonlinear system, achieving higher-order accuracy without Jacobians.

# Method
- **System model**:  
  - State: $x_k = f(x_{k-1}, u_k) + w_k$  
  - Measurement: $z_k = h(x_k) + v_k$  
- **Unscented Transform**:  
  - Select sigma points around the mean using covariance.  
  - Propagate sigma points through nonlinear functions $f$ and $h$.  
  - Reconstruct mean and covariance from transformed sigma points.  
- **Recursion**: Similar to Kalman filter (prediction + update), but with sigma points.  

# Results
- More accurate than EKF for nonlinear systems (captures 2nd-order effects).  
- No need for Jacobians or linearization.  
- Widely used in aerospace navigation, SLAM, and robotics.  

# Why it Mattered
- Solved EKF’s limitations in highly nonlinear systems.  
- Efficient alternative to Particle Filters for moderate nonlinearity.  
- Easier to implement (no symbolic derivatives).  

# Architectural Pattern
- Recursive Bayesian filter.  
- Uses sigma-point sampling for state distribution propagation.  

# Connections
- Successor to **EKF (1969)**.  
- Predecessor to **Ensemble Kalman Filter (2000)**.  
- Alternative to **Particle Filter (1993)**.  

# Implementation Notes
- Complexity: O(n³) per step (like EKF).  
- Number of sigma points = 2n+1 (where n = state dimension).  
- Choice of scaling parameters critical for stability.  

# Critiques / Limitations
- Assumes unimodal Gaussian distributions (like KF/EKF).  
- Can still fail under extreme nonlinearity or non-Gaussian noise.  
- Higher compute cost than EKF due to sigma point propagation.  

---

# Educational Connections

## Undergraduate-Level Concepts
- EKF uses Jacobians; UKF skips them.  
- UKF instead samples “sigma points” to approximate the state distribution.  
- Example: tracking a drone with nonlinear dynamics.  

## Postgraduate-Level Concepts
- Derivation of the Unscented Transform.  
- Comparison: linearization error in EKF vs distribution approximation in UKF.  
- UKF vs Particle Filter: deterministic sampling vs Monte Carlo.  
- Applications in SLAM, sensor fusion, aerospace navigation.  

---

# My Notes
- UKF = **EKF without the pain of Jacobians**.  
- Great trade-off between accuracy and efficiency.  
- Open question: Can unscented transforms be combined with deep generative models for richer priors?  
- Possible extension: Differentiable UKF modules embedded in deep learning architectures.  

---
