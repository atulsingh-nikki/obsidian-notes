---
layout: post
title: "Complete Mathematical Derivation of the Kalman Filter"
description: "Step-by-step mathematical derivation of the Kalman filter equations from first principles, including detailed proofs and alternative formulations."
tags: [kalman-filter, mathematical-derivation, linear-algebra, series]
---

*This is Part 4 of an 8-part series on Kalman Filtering. [Part 3]({{ site.baseurl }}{% link _posts/2024-09-22-bayesian-foundations-kalman.md %}) established the Bayesian foundations.*

## From Theory to Equations

In our previous post, we saw how Bayesian inference provides the theoretical foundation for optimal estimation. Now we'll derive the complete Kalman filter equations step-by-step, showing exactly how abstract probability theory becomes a practical algorithm.

## System Setup and Assumptions

### State Space Model

We consider a discrete-time linear dynamic system:

**State equation**:
$$x_k = F_k x_{k-1} + B_k u_k + w_k$$

**Measurement equation**:
$$z_k = H_k x_k + v_k$$

### Noise Assumptions

**Process noise**: $w_k \sim N(0, Q_k)$  
**Measurement noise**: $v_k \sim N(0, R_k)$  
**Independence**: $E[w_i v_j^T] = 0$ for all $i,j$

### Prior Distribution

**Initial state**: $p(x_0) = N(\hat{x}_0, P_0)$

### Notation Convention

- $\hat{x}_{k|j}$: estimate of $x_k$ given measurements up to time $j$
- $P_{k|j}$: covariance of estimation error at time $k$ given measurements up to time $j$

## The Prediction Step

### State Prediction

**Goal**: Find $E[x_k | z_{1:k-1}]$

Starting from the state equation:
$$x_k = F_k x_{k-1} + B_k u_k + w_k$$

Taking expectation given $z_{1:k-1}$:
$$E[x_k | z_{1:k-1}] = E[F_k x_{k-1} + B_k u_k + w_k | z_{1:k-1}]$$

Since $F_k$, $B_k$, and $u_k$ are known (non-random), and $w_k$ is independent of past measurements:

$$E[x_k | z_{1:k-1}] = F_k E[x_{k-1} | z_{1:k-1}] + B_k u_k + E[w_k]$$

Since $E[w_k] = 0$:

$$\boxed{\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1} + B_k u_k}$$

### Covariance Prediction

**Goal**: Find $\text{Cov}[x_k | z_{1:k-1}]$

Define the prediction error: $\tilde{x}_{k|k-1} = x_k - \hat{x}_{k|k-1}$

$$\tilde{x}_{k|k-1} = x_k - \hat{x}_{k|k-1}$$
$$= (F_k x_{k-1} + B_k u_k + w_k) - (F_k \hat{x}_{k-1|k-1} + B_k u_k)$$
$$= F_k (x_{k-1} - \hat{x}_{k-1|k-1}) + w_k$$
$$= F_k \tilde{x}_{k-1|k-1} + w_k$$

Taking covariance:
$$P_{k|k-1} = \text{Cov}[\tilde{x}_{k|k-1}]$$
$$= \text{Cov}[F_k \tilde{x}_{k-1|k-1} + w_k]$$
$$= F_k \text{Cov}[\tilde{x}_{k-1|k-1}] F_k^T + \text{Cov}[w_k]$$

Since $\tilde{x}_{k-1|k-1}$ and $w_k$ are independent:

$$\boxed{P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k}$$

## The Update Step

### Joint Distribution Setup

We need to find $p(x_k | z_{1:k})$. The key insight is to work with the joint distribution of state and measurement.

From our assumptions:
- $x_k | z_{1:k-1} \sim N(\hat{x}_{k|k-1}, P_{k|k-1})$
- $z_k | x_k \sim N(H_k x_k, R_k)$

### Deriving the Joint Distribution

The measurement is: $z_k = H_k x_k + v_k$

For the joint distribution $[x_k, z_k]^T$:

**Mean vector**:
$$E\begin{bmatrix} x_k \\ z_k \end{bmatrix} = \begin{bmatrix} \hat{x}_{k|k-1} \\ H_k \hat{x}_{k|k-1} \end{bmatrix}$$

**Covariance matrix** (this requires careful calculation):

$$\text{Cov}[x_k, x_k] = P_{k|k-1}$$

$$\text{Cov}[z_k, z_k] = \text{Cov}[H_k x_k + v_k] = H_k P_{k|k-1} H_k^T + R_k$$

$$\text{Cov}[x_k, z_k] = \text{Cov}[x_k, H_k x_k + v_k] = \text{Cov}[x_k, H_k x_k] = P_{k|k-1} H_k^T$$

Therefore:
$$\begin{bmatrix} x_k \\ z_k \end{bmatrix} \sim N\left(\begin{bmatrix} \hat{x}_{k|k-1} \\ H_k \hat{x}_{k|k-1} \end{bmatrix}, \begin{bmatrix} P_{k|k-1} & P_{k|k-1} H_k^T \\ H_k P_{k|k-1} & H_k P_{k|k-1} H_k^T + R_k \end{bmatrix}\right)$$

### Conditional Gaussian Formula

For joint Gaussian distribution $\begin{bmatrix} X \\ Y \end{bmatrix} \sim N\left(\begin{bmatrix} \mu_X \\ \mu_Y \end{bmatrix}, \begin{bmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY} \end{bmatrix}\right)$

The conditional distribution is:
$$p(X|Y) = N\left(\mu_X + \Sigma_{XY} \Sigma_{YY}^{-1}(Y - \mu_Y), \Sigma_{XX} - \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}\right)$$

### Applying the Formula

Identifying terms:
- $X = x_k$, $Y = z_k$
- $\mu_X = \hat{x}_{k|k-1}$, $\mu_Y = H_k \hat{x}_{k|k-1}$
- $\Sigma_{XX} = P_{k|k-1}$
- $\Sigma_{XY} = P_{k|k-1} H_k^T$  
- $\Sigma_{YY} = H_k P_{k|k-1} H_k^T + R_k$

**Updated mean**:
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1} (z_k - H_k \hat{x}_{k|k-1})$$

**Updated covariance**:
$$P_{k|k} = P_{k|k-1} - P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1} H_k P_{k|k-1}$$

## Standard Kalman Filter Form

### Introducing Key Variables

**Innovation** (measurement residual):
$$\tilde{y}_k = z_k - H_k \hat{x}_{k|k-1}$$

**Innovation covariance**:
$$S_k = H_k P_{k|k-1} H_k^T + R_k$$

**Kalman gain**:
$$K_k = P_{k|k-1} H_k^T S_k^{-1}$$

### Final Kalman Filter Equations

**Prediction Step**:
$$\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1} + B_k u_k$$
$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$$

**Update Step**:
$$\tilde{y}_k = z_k - H_k \hat{x}_{k|k-1}$$
$$S_k = H_k P_{k|k-1} H_k^T + R_k$$  
$$K_k = P_{k|k-1} H_k^T S_k^{-1}$$
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \tilde{y}_k$$
$$P_{k|k} = (I - K_k H_k) P_{k|k-1}$$

## Alternative Covariance Update Forms

### Joseph Form (Numerically Stable)

The standard update $P_{k|k} = (I - K_k H_k) P_{k|k-1}$ can lose positive definiteness due to numerical errors.

**Joseph form** guarantees positive definiteness:
$$P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$

**Proof of equivalence** (when $K_k$ is optimal):
$$P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$

Expanding:
$$= P_{k|k-1} - K_k H_k P_{k|k-1} - P_{k|k-1} H_k^T K_k^T + K_k H_k P_{k|k-1} H_k^T K_k^T + K_k R_k K_k^T$$

Substituting $K_k = P_{k|k-1} H_k^T S_k^{-1}$ and $S_k = H_k P_{k|k-1} H_k^T + R_k$:

After algebraic manipulation (which we'll skip), this reduces to:
$$P_{k|k} = P_{k|k-1} - K_k S_k K_k^T = (I - K_k H_k) P_{k|k-1}$$

### Information Filter Form

**Information matrix**: $Y_k = P_k^{-1}$  
**Information vector**: $y_k = P_k^{-1} \hat{x}_k$

**Information filter equations**:
$$Y_{k|k-1}^{-1} = F_k Y_{k-1|k-1}^{-1} F_k^T + Q_k$$
$$Y_{k|k} = Y_{k|k-1} + H_k^T R_k^{-1} H_k$$
$$y_{k|k} = y_{k|k-1} + H_k^T R_k^{-1} z_k$$

**Advantages**: 
- Easier to handle multiple measurements
- Natural for distributed systems
- Numerically better when $R_k^{-1}$ exists and is well-conditioned

## Matrix Calculus Verification

### Deriving the Optimal Gain

**Goal**: Minimize trace of posterior covariance

The posterior covariance in general form:
$$P_{k|k} = P_{k|k-1} - K_k S_k K_k^T$$

To find optimal $K_k$, minimize $\text{tr}(P_{k|k})$:
$$\frac{\partial \text{tr}(P_{k|k})}{\partial K_k} = -2 S_k K_k^T + 2 K_k S_k = 0$$

This gives: $K_k S_k = S_k K_k^T$, which implies: $K_k = P_{k|k-1} H_k^T S_k^{-1}$

**Verification**: This matches our derived gain exactly!

## Properties and Interpretations

### Innovation Properties

Under optimal filtering:
- **White**: $E[\tilde{y}_i \tilde{y}_j^T] = 0$ for $i \neq j$
- **Gaussian**: $\tilde{y}_k \sim N(0, S_k)$
- **Consistency check**: Actual innovation statistics should match theoretical

### Gain Matrix Insights

**High measurement noise** ($R_k$ large): $K_k \approx 0$ (trust prediction)  
**High prediction uncertainty** ($P_{k|k-1}$ large): $K_k \approx H_k^{\dagger}$ (trust measurement)  
**Poor observability** ($H_k \approx 0$): $K_k \approx 0$ (can't learn from measurement)

## Computational Complexity

**Per time step**:
- Prediction: $O(n^3)$ for covariance update
- Update: $O(n^2 m + m^3)$ where $n$ = state dim, $m$ = measurement dim
- Total: $O(n^3 + n^2 m + m^3)$

**Memory**: $O(n^2)$ for covariance matrices

## Key Takeaways

1. **Rigorous Foundation**: Every Kalman filter equation derives from first principles

2. **Optimal Fusion**: The gain matrix automatically balances prediction vs. measurement uncertainty

3. **Gaussian Propagation**: Linear operations preserve Gaussian distributions exactly

4. **Multiple Formulations**: Standard, Joseph, and Information forms each have advantages

5. **Verifiable Optimality**: Matrix calculus confirms the derived gain is optimal

6. **Practical Considerations**: Numerical stability requires careful implementation

## Looking Forward

With the complete mathematical foundation in place, our next post will transform these equations into working Python code. We'll see how to implement each step, handle edge cases, and create a robust, practical Kalman filter implementation.

The journey from abstract mathematics to running code reveals both the elegance of the theory and the care required for reliable implementation.

*Continue to [Part 5: Implementing the Kalman Filter in Python]({{ site.baseurl }}{% link _posts/2024-09-24-kalman-filter-implementation.md %})*
