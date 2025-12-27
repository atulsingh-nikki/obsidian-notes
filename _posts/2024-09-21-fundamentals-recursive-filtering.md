---
layout: post
title: "Fundamentals of Recursive Filtering"
description: "Understanding the mathematical framework and universal patterns that make recursive filters so powerful for real-time estimation."
tags: [kalman-filter, recursive-filtering, signal-processing, series]
---

*This is Part 2 of an 8-part series on Kalman Filtering. [Part 1]({{ site.baseurl }}{% link _posts/2024-09-20-introduction-to-kalman-filtering.md %}) introduced state estimation concepts.*


## Table of Contents

- [What Makes a Filter "Recursive"?](#what-makes-a-filter-recursive)
- [The Universal Recursive Pattern](#the-universal-recursive-pattern)
  - [Mathematical Framework](#mathematical-framework)
  - [The Innovation-Based Update](#the-innovation-based-update)
- [Building Intuition: The Recursive Average](#building-intuition-the-recursive-average)
  - [Problem: Computing a Running Average](#problem-computing-a-running-average)
  - [Mathematical Derivation](#mathematical-derivation)
  - [The Beautiful Result](#the-beautiful-result)
  - [Key Insights](#key-insights)
- [Exponential Smoothing: Adding Forgetting](#exponential-smoothing-adding-forgetting)
  - [Mathematical Form](#mathematical-form)
  - [Rewritten in Innovation Form](#rewritten-in-innovation-form)
  - [Exponential Weighting](#exponential-weighting)
  - [Key Properties](#key-properties)
- [Why Recursive Filters are Powerful](#why-recursive-filters-are-powerful)
  - [1. Memory Efficiency](#1-memory-efficiency)
  - [2. Real-Time Processing](#2-real-time-processing)
  - [3. Predictive Capability](#3-predictive-capability)
  - [4. Computational Efficiency](#4-computational-efficiency)
- [The Bridge to Kalman Filtering](#the-bridge-to-kalman-filtering)
  - [Limitations of Simple Filters](#limitations-of-simple-filters)
  - [Kalman Filter Extensions](#kalman-filter-extensions)
- [Applications of Simple Recursive Filters](#applications-of-simple-recursive-filters)
  - [Financial Data Smoothing](#financial-data-smoothing)
  - [Sensor Noise Reduction](#sensor-noise-reduction)
  - [Network Latency Estimation](#network-latency-estimation)
- [Implementation Considerations](#implementation-considerations)
  - [Initialization](#initialization)
  - [Numerical Stability](#numerical-stability)
  - [Parameter Tuning](#parameter-tuning)
- [Key Takeaways](#key-takeaways)
- [Looking Ahead](#looking-ahead)

## What Makes a Filter "Recursive"?

A **recursive filter** processes data sequentially, updating its internal state with each new observation. Unlike batch processors that need all data upfront, recursive filters:

- Process **one measurement at a time**
- Maintain an **internal state** summarizing all past information  
- **Update** this state when new data arrives
- Operate in **real-time** with bounded memory

The Kalman filter is the most famous recursive filter, but it's part of a larger family sharing this elegant approach.

## The Universal Recursive Pattern

Every recursive filter follows the same fundamental structure:

### Mathematical Framework

**State Update**: $\hat{x}_k = f(\hat{x}_{k-1}, z_k, k)$  
**Uncertainty Update**: $P_k = g(P_{k-1}, z_k, k)$

Where:
- $\hat{x}_k$ = estimated state at time k
- $z_k$ = measurement at time k  
- $P_k$ = measure of estimation uncertainty
- $f()$, $g()$ = update functions specific to each filter

### The Innovation-Based Update

Most successful recursive filters use this pattern:

$$
New Estimate = Old Estimate + Gain × Innovation
$$

Where:
- **Innovation** = $(Measurement - Prediction)$
- **Gain** determines trust between measurement and prediction

## Building Intuition: The Recursive Average

Let's derive the simplest recursive filter to understand the core concepts.

### Problem: Computing a Running Average

Given measurements $z_1, z_2, z_3, ...$, we want the running average without storing all past values.

**Traditional approach**:
$$
Average = (z_1 + z_2 + ... + z_k) / k
$$

**Memory problem**: Requires storing all k measurements!

### Mathematical Derivation

**Step 1**: Start with the definition of sample mean
$$
\bar{z}_k = \frac{1}{k} \sum_{i=1}^{k} z_i
$$

**Step 2**: Separate the latest measurement
$$
\bar{z}_k = \frac{1}{k} \left[(k-1) \times \bar{z}_{k-1} + z_k\right]
$$

**Step 3**: Expand and rearrange
$$
\bar{z}_k = \frac{k-1}{k} \times \bar{z}_{k-1} + \frac{1}{k} \times z_k
$$

**Step 4**: Rearrange to innovation form
$$
\bar{z}_k = \bar{z}_{k-1} + \frac{1}{k} \times (z_k - \bar{z}_{k-1})
$$

### The Beautiful Result

$$
\text{New Average} = \text{Old Average} + \frac{1}{k} \times \text{Innovation}
$$

Where $\text{Innovation} = (z_k - \bar{z}_{k-1})$ is how much the new measurement differs from our current estimate.

### Key Insights

1. **Innovation Interpretation**: $(z_k - \bar{z}_{k-1})$ measures surprise – how much the measurement differs from expectation

2. **Adaptive Gain**: As $k$ increases, gain $\frac{1}{k}$ decreases, making the filter less responsive to new data

3. **Memory Efficiency**: Only need to store $\bar{z}_{k-1}$ and $k$, not all past measurements

4. **Universal Pattern**: This same structure appears in every recursive filter!

## Exponential Smoothing: Adding Forgetting

The recursive average treats all measurements equally, but what if we want recent data to matter more?

### Mathematical Form
$$y_k = \alpha \times z_k + (1-\alpha) \times y_{k-1}, \quad 0 < \alpha \leq 1$$

### Rewritten in Innovation Form
$$y_k = y_{k-1} + \alpha \times (z_k - y_{k-1})$$

### Exponential Weighting

Expanding the recursion reveals how past measurements contribute:
$$y_k = \alpha z_k + \alpha(1-\alpha) z_{k-1} + \alpha(1-\alpha)^2 z_{k-2} + \ldots$$

The weights form a geometric series: $\alpha, \alpha(1-\alpha), \alpha(1-\alpha)^2, \ldots$

### Key Properties

- **Exponential decay**: Older measurements have exponentially decreasing influence
- **Tunable memory**: Parameter $\alpha$ controls the "memory length"
  - Large $\alpha$: Quick adaptation, short memory
  - Small $\alpha$: Slow adaptation, long memory
- **Constant memory**: Always just one number to store

## Why Recursive Filters are Powerful

### 1. Memory Efficiency
**Fixed memory** regardless of data length:
- Recursive average: Store $\bar{z}_k$ and $k$
- Exponential smoothing: Store $y_{k-1}$
- Kalman filter: Store $\hat{x}_{k-1}$ and $P_{k-1}$

### 2. Real-Time Processing
- Process measurements **as they arrive**
- No batch processing delays
- Perfect for streaming applications

### 3. Predictive Capability
- Current state estimate can predict future values
- Handle missing measurements gracefully
- Enable proactive decision making

### 4. Computational Efficiency
- **O(1)** computational complexity per update for simple filters
- **O(n³)** for Kalman filter (where n is state dimension)
- Much faster than batch methods for sequential data

## The Bridge to Kalman Filtering

The recursive average and exponential smoothing reveal the essential pattern, but they're limited:

### Limitations of Simple Filters
1. **Scalar only**: Can't handle multi-dimensional states
2. **No dynamics**: Don't model how systems evolve over time  
3. **No uncertainty**: Don't quantify confidence in estimates
4. **Static parameters**: Can't adapt gain based on measurement quality

### Kalman Filter Extensions
The Kalman filter generalizes these concepts:

1. **Vector states**: Handle multi-dimensional systems
2. **System dynamics**: Model how states evolve over time
3. **Uncertainty propagation**: Track and update confidence measures
4. **Optimal gain**: Automatically compute the best gain based on uncertainties

## Applications of Simple Recursive Filters

### Financial Data Smoothing
```python
# Exponential moving average for stock prices
def update_ema(current_ema, new_price, alpha=0.1):
    return current_ema + alpha * (new_price - current_ema)
```

### Sensor Noise Reduction
```python
# Recursive average for temperature sensor
def update_temperature(avg_temp, new_reading, count):
    gain = 1.0 / count
    return avg_temp + gain * (new_reading - avg_temp)
```

### Network Latency Estimation
```python
# Exponential smoothing for RTT estimation
def update_rtt(current_rtt, measured_rtt, alpha=0.125):
    return current_rtt + alpha * (measured_rtt - current_rtt)
```

## Implementation Considerations

### Initialization
- **Recursive average**: Start with first measurement $\bar{z}_1 = z_1$
- **Exponential smoothing**: Initialize with first measurement or prior knowledge

### Numerical Stability
- Use appropriate data types (float vs. double)
- Watch for overflow in running counts
- Consider numerical precision in gain calculations

### Parameter Tuning
- **Recursive average**: No parameters to tune!
- **Exponential smoothing**: Choose $\alpha$ based on desired responsiveness

## Key Takeaways

1. **Universal Pattern**: All recursive filters follow $\text{New} = \text{Old} + \text{Gain} \times \text{Innovation}$

2. **Innovation is Key**: The difference between measurement and prediction drives all updates

3. **Memory Efficiency**: Recursive structure enables constant memory usage

4. **Tunable Behavior**: Parameters control the balance between responsiveness and smoothing

5. **Foundation for Complexity**: Simple patterns scale to sophisticated multi-dimensional filters

## Looking Ahead

Understanding these simple recursive filters provides the foundation for comprehending the Kalman filter. In our next post, we'll explore the **Bayesian foundations** that give the Kalman filter its optimality properties and mathematical rigor.

The journey from recursive average to Kalman filter is really about:
- **Single values → Vector states**
- **Fixed parameters → Adaptive gains**  
- **No uncertainty → Optimal uncertainty propagation**
- **Simple updates → Mathematically optimal estimation**

*Continue to [Part 3: Bayesian Foundations of Kalman Filtering]({{ site.baseurl }}{% link _posts/2024-09-22-bayesian-foundations-kalman.md %})*
