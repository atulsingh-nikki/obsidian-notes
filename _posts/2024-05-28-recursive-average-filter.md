---
layout: post
title: "Deriving the Recursive Average Filter"
description: "Mathematical derivation of the Recursive Moving Average Filter and its applications in signal processing, communications, and embedded systems."
tags: [signal-processing, math]
---

The Recursive Moving Average Filter (RMAF) is a digital filter that computes a running average of input samples using a recursive structure. Unlike simple moving averages that require storing multiple past samples, the RMAF maintains only a few state variables, making it memory-efficient and computationally lightweight.

### Mathematical Derivation

The recursive moving average filter can be derived from the standard moving average formula. For a moving average of $N$ samples:

$$y[n] = \frac{1}{N} \sum_{k=0}^{N-1} x[n-k]$$

We can express this recursively by relating the current output to the previous output:

$$y[n] = y[n-1] + \frac{1}{N}(x[n] - x[n-N])$$

This formulation requires storing the sample from $N$ steps ago. To eliminate this requirement, we can derive a purely recursive form:

$$y[n] = \frac{N-1}{N} y[n-1] + \frac{1}{N} x[n]$$

Let $a = \frac{N-1}{N}$ and $b = \frac{1}{N}$, then:

$$y[n] = a \cdot y[n-1] + b \cdot x[n]$$

where $a + b = 1$, ensuring unity gain for DC signals.

### Transfer Function Analysis

Taking the Z-transform of the recursive equation:

$$Y(z) = a \cdot z^{-1} Y(z) + b \cdot X(z)$$

The transfer function becomes:

$$H(z) = \frac{Y(z)}{X(z)} = \frac{b}{1 - a z^{-1}} = \frac{b z}{z - a}$$

This is a first-order IIR (Infinite Impulse Response) filter with a single pole at $z = a$.

### Frequency Response

The magnitude response is:

$$|H(\omega)| = \frac{b}{\sqrt{1 + a^2 - 2a\cos(\omega)}}$$

This creates a low-pass characteristic, with the cutoff frequency determined by the parameter $a$.

### Key Properties

1. **Memory Efficiency**: Only requires storing one previous output value
2. **Computational Efficiency**: One multiplication and one addition per sample
3. **Stability**: Always stable since $|a| < 1$ for practical values of $N$
4. **Phase Response**: Introduces phase lag, especially at higher frequencies

### Applications

**1. Signal Smoothing**
- Noise reduction in sensor readings
- Smoothing control system signals
- Audio signal processing

**2. Digital Communications**
- Channel equalization
- Symbol timing recovery
- Carrier frequency estimation

**3. Biomedical Signal Processing**
- ECG baseline wandering removal
- EEG artifact reduction
- Real-time vital sign monitoring

**4. Financial Data Analysis**
- Stock price trend analysis
- Risk assessment smoothing
- High-frequency trading algorithms

**5. Embedded Systems**
- Battery voltage monitoring
- Temperature regulation
- Motor speed control

### Implementation Considerations

**Initialization**: Set $y[0] = x[0]$ to avoid transient startup behavior.

**Numerical Precision**: Use sufficient precision for coefficient $a$ to prevent quantization errors.

**Parameter Selection**: Choose $N$ based on desired smoothing vs. responsiveness trade-off:
- Large $N$: More smoothing, slower response
- Small $N$: Less smoothing, faster response

The Recursive Moving Average Filter provides an elegant solution for real-time signal processing applications where memory and computational resources are constrained, while maintaining the fundamental smoothing properties of traditional moving average filters.
