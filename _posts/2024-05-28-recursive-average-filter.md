---
layout: post
title: "Deriving the Recursive Average Filter"
description: "Step-by-step derivation of the exponentially weighted moving average used in streaming signal smoothing."
tags: [signal-processing, math]
---

When smoothing a live measurement, storing an ever-growing window of past samples quickly becomes impractical. The recursive
average filter, sometimes called an exponentially weighted moving average (EWMA), offers a compact alternative: each new sample
updates a single state variable that carries the whole history of the signal in compressed form.

### Starting from the sliding window mean

Suppose we want the discrete-time mean of the most recent $N$ samples of a signal $x[n]$:

$$m[n] = \frac{1}{N} \sum_{k=0}^{N-1} x[n-k].$$

Directly evaluating this sum at every step requires $N$ additions and a division. We can save work by reusing the previous
average $m[n-1]$ and subtracting the oldest sample that just left the window:

$$m[n] = m[n-1] + \frac{1}{N} \big(x[n] - x[n-N]\big).$$

This recursion still needs access to the sample $N$ steps in the past. For embedded systems with tight memory constraints, even
that can be too expensive.

### Introducing exponential forgetting

Instead of a hard cutoff after $N$ points, we allow older samples to decay geometrically. Define the recursively weighted mean
$y[n]$ by combining the new sample with the previous state:

$$y[n] = \alpha x[n] + (1-\alpha) y[n-1], \quad 0 < \alpha \leq 1.$$

Expanding this recursion by repeated substitution reveals how each past sample contributes:

\begin{align*}
y[n] &= \alpha x[n] + (1-\alpha) y[n-1] \\
     &= \alpha x[n] + (1-\alpha) \big(\alpha x[n-1] + (1-\alpha) y[n-2]\big) \\
     &= \alpha x[n] + \alpha (1-\alpha) x[n-1] + (1-\alpha)^2 y[n-2] \\
     &\phantom{=} \vdots \\
     &= \alpha \sum_{k=0}^{\infty} (1-\alpha)^k x[n-k].
\end{align*}

The weights form a geometric series that sums to one, ensuring the filter preserves the DC level of the signal. Unlike the
finite window mean, we never reference a specific sample in the past; we only store the previous output $y[n-1]$.

### Choosing the smoothing factor

Setting $\alpha = 2/(N+1)$ makes the recursive filter match the variance reduction of an $N$-point sliding average, a common
rule of thumb when converting between the two. Smaller $\alpha$ values slow the response but increase smoothing because the
weights decay more gradually. In implementation, $\alpha$ becomes the knob that balances responsiveness against noise
suppression.

### Practical considerations

* **Initialization:** Start with $y[0] = x[0]$ or a known baseline to avoid the filter ramping up from zero.
* **Numerical stability:** Keep $\alpha$ in floating-point even if the sensor readings are integers; quantization noise on the
  weight can bias the long-term mean.
* **Streaming efficiency:** Each update costs one multiply, one addition, and minimal memory, making the recursion ideal for
  microcontrollers and high-rate telemetry pipelines.

The recursive average filter emerges naturally once we relax the idea of a finite window and embrace exponential forgetting. The
result is a mathematically elegant smoother that is easy to implement in any environment where memory and cycles are at a
premium.
