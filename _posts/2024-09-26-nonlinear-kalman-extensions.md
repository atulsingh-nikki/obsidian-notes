---
layout: post
title: "Nonlinear Extensions: EKF, UKF, and Particle Filters"
description: "Moving beyond linear-Gaussian assumptions with Extended Kalman Filters, Unscented Kalman Filters, and Particle Filters for real-world nonlinear systems."
tags: [kalman-filter, ekf, ukf, particle-filter, nonlinear, series]
---

*This is Part 7 of an 8-part series on Kalman Filtering. [Part 6]({{ site.baseurl }}{% link _posts/2024-09-25-kalman-filter-applications.md %}) explored real-world applications.*


## Table of Contents

- [Beyond Linear-Gaussian: The Real World](#beyond-linear-gaussian-the-real-world)
- [The Challenge of Nonlinearity](#the-challenge-of-nonlinearity)
  - [When Linearity Breaks Down](#when-linearity-breaks-down)
  - [Why Standard Kalman Filter Fails](#why-standard-kalman-filter-fails)
- [Extended Kalman Filter (EKF)](#extended-kalman-filter-ekf)
  - [The Core Idea: Linearize Locally](#the-core-idea-linearize-locally)
  - [Mathematical Framework](#mathematical-framework)
  - [EKF Algorithm](#ekf-algorithm)
  - [Robot Localization Example](#robot-localization-example)
  - [EKF Advantages](#ekf-advantages)
  - [EKF Limitations](#ekf-limitations)
- [Unscented Kalman Filter (UKF)](#unscented-kalman-filter-ukf)
  - [The Innovation: Deterministic Sampling](#the-innovation-deterministic-sampling)
  - [Key Insight](#key-insight)
  - [The Unscented Transform](#the-unscented-transform)
  - [UKF Algorithm](#ukf-algorithm)
  - [UKF Advantages](#ukf-advantages)
  - [UKF Limitations](#ukf-limitations)
- [Particle Filters](#particle-filters)
  - [When Gaussians Aren't Enough](#when-gaussians-arent-enough)
  - [Core Concept: Weighted Particles](#core-concept-weighted-particles)
  - [Sequential Importance Resampling (SIR)](#sequential-importance-resampling-sir)
  - [Particle Filter Applications](#particle-filter-applications)
  - [Particle Filter Advantages](#particle-filter-advantages)
  - [Particle Filter Limitations](#particle-filter-limitations)
- [Comparison and Selection Criteria](#comparison-and-selection-criteria)
  - [Selection Guidelines](#selection-guidelines)
- [Implementation Best Practices](#implementation-best-practices)
  - [Hybrid Approaches](#hybrid-approaches)
  - [Numerical Considerations](#numerical-considerations)
- [Key Takeaways](#key-takeaways)
- [Looking Forward](#looking-forward)

## Beyond Linear-Gaussian: The Real World

The standard Kalman filter assumes linear dynamics and Gaussian noise – elegant mathematical assumptions that make optimal estimation tractable. But real-world systems are often stubbornly nonlinear:

- **Robot motion**: Turning involves trigonometric functions
- **Satellite orbits**: Governed by nonlinear gravitational dynamics
- **Chemical processes**: Exponential reaction rates
- **Economic systems**: Nonlinear feedback loops

This post explores three major approaches for handling nonlinearity while preserving the recursive, real-time nature that makes Kalman filtering so valuable.

## The Challenge of Nonlinearity

### When Linearity Breaks Down

Consider a robot turning with angular velocity ω:

**True nonlinear dynamics**:
```
x(k+1) = x(k) + v*cos(θ(k))*dt
y(k+1) = y(k) + v*sin(θ(k))*dt  
θ(k+1) = θ(k) + ω*dt
```

**Measurement model** (bearing and range to landmark):
```
range = √[(x_landmark - x)² + (y_landmark - y)²]
bearing = atan2(y_landmark - y, x_landmark - x) - θ
```

These equations can't be expressed as linear matrix operations `x = Fx + w` and `z = Hx + v`.

### Why Standard Kalman Filter Fails

1. **Gaussian distributions aren't preserved** through nonlinear transformations
2. **Covariance propagation** requires linearization or approximation
3. **Optimality guarantees** no longer hold
4. **Divergence** can occur from accumulated linearization errors

## Extended Kalman Filter (EKF)

### The Core Idea: Linearize Locally

The EKF linearizes nonlinear functions around the current state estimate using **Taylor series approximation**.

### Mathematical Framework

**Nonlinear system model**:
```
x(k) = f(x(k-1), u(k-1)) + w(k-1)
z(k) = h(x(k)) + v(k)
```

**Linearization**:
```
F(k) = ∂f/∂x |_(x̂(k-1|k-1))    # Jacobian of state transition
H(k) = ∂h/∂x |_(x̂(k|k-1))      # Jacobian of measurement function
```

### EKF Algorithm

**Prediction Step**:
```python
def ekf_predict(x_est, P, u, Q, f_function, F_jacobian, dt):
    """EKF Prediction step"""
    # Nonlinear state prediction
    x_pred = f_function(x_est, u, dt)
    
    # Linearize around current estimate  
    F = F_jacobian(x_est, u, dt)
    
    # Linear covariance prediction
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z, R, h_function, H_jacobian):
    """EKF Update step"""
    # Nonlinear measurement prediction
    z_pred = h_function(x_pred)
    
    # Linearize around predicted state
    H = H_jacobian(x_pred)
    
    # Innovation and covariance
    y = z - z_pred
    S = H @ P_pred @ H.T + R
    
    # Kalman gain and update (same as linear case)
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P_est = (np.eye(len(x_pred)) - K @ H) @ P_pred
    
    return x_est, P_est
```

### Robot Localization Example

```python
class RobotEKF:
    def __init__(self):
        # State: [x, y, θ] - position and heading
        self.state_dim = 3
        
    def motion_model(self, state, control, dt):
        """Nonlinear motion model"""
        x, y, theta = state[0], state[1], state[2]
        v, w = control[0], control[1]  # velocity and angular rate
        
        return np.array([
            x + v * np.cos(theta) * dt,
            y + v * np.sin(theta) * dt,
            theta + w * dt
        ])
    
    def motion_jacobian(self, state, control, dt):
        """Jacobian of motion model"""
        x, y, theta = state[0], state[1], state[2]  
        v, w = control[0], control[1]
        
        return np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0,  1]
        ])
    
    def measurement_model(self, state, landmark_pos):
        """Range and bearing to landmark"""
        x, y, theta = state[0], state[1], state[2]
        lx, ly = landmark_pos[0], landmark_pos[1]
        
        dx, dy = lx - x, ly - y
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) - theta
        
        return np.array([r, phi])
    
    def measurement_jacobian(self, state, landmark_pos):
        """Jacobian of measurement model"""
        x, y, theta = state[0], state[1], state[2]
        lx, ly = landmark_pos[0], landmark_pos[1]
        
        dx, dy = lx - x, ly - y
        r = np.sqrt(dx**2 + dy**2)
        
        return np.array([
            [-dx/r, -dy/r,  0],
            [ dy/r**2, -dx/r**2, -1]
        ])
```

### EKF Advantages

1. **Familiar structure**: Similar to linear Kalman filter
2. **Computational efficiency**: O(n³) complexity like linear case  
3. **Well-established**: Decades of successful applications
4. **Good performance**: Works well for mildly nonlinear systems

### EKF Limitations

1. **Linearization errors**: Can accumulate and cause divergence
2. **Jacobian computation**: Requires analytical derivatives
3. **Local optimality only**: Can get stuck in local minima
4. **Gaussian assumption**: Still assumes Gaussian distributions

## Unscented Kalman Filter (UKF)

### The Innovation: Deterministic Sampling

Rather than linearizing nonlinear functions, the UKF uses the **Unscented Transform** – a deterministic sampling technique that propagates carefully chosen "sigma points" through nonlinear transformations.

### Key Insight

**"It's easier to approximate a probability distribution than it is to approximate an arbitrary nonlinear function."** - Jeffrey Uhlmann

### The Unscented Transform

**Step 1: Generate Sigma Points**
```python
def generate_sigma_points(mean, cov, alpha=1e-3, beta=2, kappa=None):
    """Generate sigma points for unscented transform"""
    n = len(mean)
    if kappa is None:
        kappa = 3 - n
        
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Compute square root of covariance
    sqrt_cov = np.linalg.cholesky((n + lambda_) * cov)
    
    # Generate sigma points
    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = mean
    
    for i in range(n):
        sigma_points[i + 1] = mean + sqrt_cov[i]
        sigma_points[n + i + 1] = mean - sqrt_cov[i]
    
    # Compute weights
    wm = np.zeros(2*n + 1)  # weights for means
    wc = np.zeros(2*n + 1)  # weights for covariances
    
    wm[0] = lambda_ / (n + lambda_)
    wc[0] = lambda_ / (n + lambda_) + 1 - alpha**2 + beta
    
    for i in range(1, 2*n + 1):
        wm[i] = wc[i] = 0.5 / (n + lambda_)
    
    return sigma_points, wm, wc
```

**Step 2: Transform Sigma Points**
```python
def unscented_transform(sigma_points, wm, wc, noise_cov, transform_func):
    """Apply unscented transform"""
    # Transform each sigma point
    transformed_points = np.array([transform_func(sp) for sp in sigma_points])
    
    # Compute transformed mean
    transformed_mean = np.sum(wm[:, np.newaxis] * transformed_points, axis=0)
    
    # Compute transformed covariance  
    centered_points = transformed_points - transformed_mean
    transformed_cov = noise_cov.copy()
    
    for i in range(len(wc)):
        transformed_cov += wc[i] * np.outer(centered_points[i], centered_points[i])
    
    return transformed_mean, transformed_cov, transformed_points
```

### UKF Algorithm

```python
class UnscentedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, alpha=1e-3, beta=2, kappa=None):
        self.n = state_dim
        self.m = measurement_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - state_dim
        
    def predict(self, mean, cov, process_noise, process_model, dt):
        """UKF Prediction step"""
        # Generate sigma points
        sigma_points, wm, wc = generate_sigma_points(mean, cov, self.alpha, self.beta, self.kappa)
        
        # Propagate sigma points through process model
        predicted_mean, predicted_cov, _ = unscented_transform(
            sigma_points, wm, wc, process_noise, 
            lambda sp: process_model(sp, dt)
        )
        
        return predicted_mean, predicted_cov
    
    def update(self, predicted_mean, predicted_cov, measurement, measurement_noise, measurement_model):
        """UKF Update step"""  
        # Generate sigma points from prediction
        sigma_points, wm, wc = generate_sigma_points(predicted_mean, predicted_cov, 
                                                    self.alpha, self.beta, self.kappa)
        
        # Transform through measurement model
        z_pred, S, transformed_points = unscented_transform(
            sigma_points, wm, wc, measurement_noise, measurement_model
        )
        
        # Compute cross-covariance
        cross_cov = np.zeros((self.n, self.m))
        for i in range(len(wc)):
            cross_cov += wc[i] * np.outer(sigma_points[i] - predicted_mean,
                                         transformed_points[i] - z_pred)
        
        # Kalman gain and update
        K = cross_cov @ np.linalg.inv(S)
        updated_mean = predicted_mean + K @ (measurement - z_pred)
        updated_cov = predicted_cov - K @ S @ K.T
        
        return updated_mean, updated_cov
```

### UKF Advantages

1. **No Jacobians required**: Uses sigma point sampling instead of derivatives
2. **Higher-order accuracy**: Captures nonlinearities up to 3rd order
3. **Same computational complexity**: O(n³) like EKF and linear KF
4. **More robust**: Better handling of strong nonlinearities
5. **Easy implementation**: Straightforward algorithm structure

### UKF Limitations

1. **Still assumes Gaussian**: Underlying distributions assumed Gaussian
2. **Parameter tuning**: Alpha, beta, kappa parameters need tuning
3. **Numerical stability**: Can have issues with ill-conditioned covariances
4. **Memory overhead**: Requires storing multiple sigma points

## Particle Filters

### When Gaussians Aren't Enough

For highly nonlinear systems with non-Gaussian noise, we need a more general approach. **Particle filters** use Monte Carlo methods to represent arbitrary probability distributions.

### Core Concept: Weighted Particles

Instead of representing distributions with mean and covariance, use a set of weighted samples (particles):

```
p(x) ≈ ∑(i=1 to N) w_i * δ(x - x_i)
```

Where:
- `x_i` are particle locations
- `w_i` are particle weights  
- `N` is the number of particles
- `δ()` is the Dirac delta function

### Sequential Importance Resampling (SIR)

```python
class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.N = num_particles
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
        
    def predict(self, process_model, process_noise, dt):
        """Prediction step: propagate particles"""
        for i in range(self.N):
            # Add process noise to each particle
            noise = np.random.multivariate_normal(np.zeros(len(self.particles[i])), 
                                                 process_noise)
            self.particles[i] = process_model(self.particles[i], dt) + noise
    
    def update(self, measurement, measurement_model, measurement_noise):
        """Update step: reweight particles based on likelihood"""
        for i in range(self.N):
            # Predicted measurement for this particle
            z_pred = measurement_model(self.particles[i])
            
            # Compute likelihood of actual measurement
            residual = measurement - z_pred
            likelihood = multivariate_gaussian_pdf(residual, np.zeros(len(residual)), 
                                                  measurement_noise)
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
        # Resample if effective sample size is too low
        if 1.0 / np.sum(self.weights**2) < self.N / 2:
            self.resample()
    
    def resample(self):
        """Systematic resampling"""
        indices = systematic_resample(self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N
    
    def get_estimate(self):
        """Compute weighted mean and covariance"""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        
        # Weighted covariance calculation
        diff = self.particles - mean
        cov = np.average(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
                        weights=self.weights, axis=0)
        
        return mean, cov

def systematic_resample(weights):
    """Systematic resampling algorithm"""
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N
    
    indices = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
            
    return indices
```

### Particle Filter Applications

**Robot Localization in Unknown Environments**:
```python
class RobotParticleFilter:
    def __init__(self, num_particles, map_data):
        self.pf = ParticleFilter(num_particles, 3)  # [x, y, θ]
        self.map = map_data
        
    def motion_model(self, particle, control, dt):
        """Robot motion with noise"""
        x, y, theta = particle
        v, w = control
        
        # Add motion noise
        v_noise = v + np.random.normal(0, 0.1)
        w_noise = w + np.random.normal(0, 0.05)
        
        return np.array([
            x + v_noise * np.cos(theta) * dt,
            y + v_noise * np.sin(theta) * dt,
            theta + w_noise * dt
        ])
    
    def sensor_model(self, particle, laser_scan):
        """Likelihood of laser scan given particle pose"""
        # Ray-cast from particle pose through map
        # Compare predicted scan with actual scan
        # Return likelihood score
        pass
```

### Particle Filter Advantages

1. **No distributional assumptions**: Can handle arbitrary probability distributions
2. **Handles multimodality**: Multiple hypotheses represented naturally
3. **Robust to nonlinearity**: No linearization required
4. **Conceptually simple**: Easy to understand and implement
5. **Flexible**: Easy to incorporate complex noise models

### Particle Filter Limitations

1. **Computational cost**: O(N) where N can be large (thousands of particles)
2. **Particle depletion**: All particles can converge to single mode
3. **Curse of dimensionality**: Exponentially more particles needed for high dimensions
4. **Parameter tuning**: Number of particles and resampling thresholds need tuning

## Comparison and Selection Criteria

| Method | Computational Cost | Nonlinearity Handling | Distributional Assumptions | Best Use Cases |
|--------|-------------------|---------------------|---------------------------|----------------|
| **Linear KF** | O(n³) | None | Linear-Gaussian | Linear systems, sensor fusion |
| **EKF** | O(n³) | Linearization | Gaussian | Mildly nonlinear, real-time systems |
| **UKF** | O(n³) | Sigma points | Gaussian | Moderate nonlinearity, no derivatives |
| **Particle Filter** | O(N⋅n²) | Monte Carlo | None | Highly nonlinear, non-Gaussian |

### Selection Guidelines

**Choose Linear Kalman Filter when**:
- System is truly linear (or very close)
- Gaussian noise assumptions hold
- Maximum computational efficiency needed

**Choose Extended Kalman Filter when**:
- Mildly nonlinear system
- Jacobians are easy to compute
- Real-time constraints are tight
- Well-understood system dynamics

**Choose Unscented Kalman Filter when**:
- Moderate nonlinearities present
- Jacobians are difficult/expensive to compute  
- Better accuracy needed than EKF
- System remains approximately Gaussian

**Choose Particle Filter when**:
- Highly nonlinear system
- Non-Gaussian noise or multi-modal distributions
- Computational resources available
- Robustness more important than efficiency

## Implementation Best Practices

### Hybrid Approaches

**Multiple Model Filtering**:
```python
class MultipleModelFilter:
    def __init__(self, models):
        self.models = models  # List of different KF variants
        self.model_probabilities = np.ones(len(models)) / len(models)
        
    def update(self, measurement):
        # Run each model
        for model in self.models:
            model.predict()
            model.update(measurement)
            
        # Update model probabilities based on innovation likelihoods
        # Choose best model or blend results
```

**Switching Between Methods**:
- Use EKF for normal operation
- Switch to UKF when nonlinearity increases  
- Fall back to particle filter for extreme cases

### Numerical Considerations

1. **Square Root Filtering**: Use Cholesky decomposition to maintain positive definiteness
2. **Joseph Form Updates**: Ensure numerical stability in covariance updates
3. **Regularization**: Add small values to diagonal elements to prevent singular matrices
4. **Scaling**: Normalize state variables to similar magnitudes

## Key Takeaways

1. **No Universal Solution**: Each method has trade-offs between accuracy, computational cost, and robustness

2. **Progressive Complexity**: Start with simplest method that meets requirements

3. **Application-Dependent**: Choice depends on system nonlinearity, noise characteristics, and computational constraints

4. **Hybrid Approaches**: Combining methods can leverage individual strengths

5. **Validation Critical**: Extensive testing needed to verify performance in real conditions

## Looking Forward

The nonlinear extensions we've explored represent the current state-of-the-art in recursive filtering. Our final post will examine **advanced topics and future directions** – including modern machine learning integration, distributed filtering, and emerging applications.

The evolution from linear Kalman filters to sophisticated nonlinear estimators mirrors the broader trajectory of engineering: starting with elegant mathematical foundations and extending them to handle real-world complexity while preserving practical utility.

*Continue to [Part 8: Advanced Topics and Future Directions]({{ site.baseurl }}{% link _posts/2024-09-27-advanced-kalman-topics.md %})*
