---
layout: post
title: "SIMD Intrinsics: From SSE to AVX2 in Practice"
description: "Learn how compiler intrinsics expose SIMD instructions, how AVX widens your arithmetic, and where these low-level tools still shine."
tags: [simd, intrinsics, avx, performance, c++]
---

Modern CPUs have wide vector units capable of executing the same operation across multiple data elements at once. Compiler auto-vectorization handles the simple cases, but when you need precise control—special data layouts, mixed operations, or custom reductions—SIMD intrinsics become essential. This post explains the basics of SSE/AVX intrinsics, walks through a few examples, and points out when the low-level approach pays off.

### Why Intrinsics Exist

- **Hardware has moved faster than compilers**: Since Intel’s Pentium III (1999) introduced SSE, CPUs have steadily widened their vector units. AVX (2011) and AVX2 (2013) doubled the width to 256 bits, but compilers do not always auto-vectorize real-world loops (non-contiguous memory, conditionals, or custom reductions).
- **Predictable performance**: Intrinsics let you choose the exact instruction sequence, avoiding surprises from changing compiler heuristics or optimization levels.
- **Access to specialty instructions**: Saturating arithmetic, horizontal adds, fused multiply-add, and gathers are not always emitted automatically.
- **Portability by design**: You can detect CPU features at runtime (CPUID) or compile-time (`__AVX2__`) and dispatch to the best kernel; the scalar fallback keeps the program running on older hardware.

### ISA Support Cheat Sheet

| Instruction Set | Register Width | First Intel CPU | First AMD CPU | Typical Flag |
|-----------------|----------------|-----------------|---------------|--------------|
| SSE             | 128-bit        | Pentium III (Katmai, 1999) | Athlon 64 (2003) | `-msse` (implied by modern compilers) |
| SSE2            | 128-bit        | Pentium 4 (Willamette, 2000) | Athlon 64 (2003) | `-msse2` |
| SSE3 / SSSE3    | 128-bit        | Pentium 4 Prescott (2004) / Core 2 (2006) | Phenom (2007) | `-msse3`, `-mssse3` |
| SSE4.1 / 4.2    | 128-bit        | Penryn / Nehalem (2007/2008) | Bulldozer (2011) | `-msse4.1`, `-msse4.2` |
| AVX             | 256-bit        | Sandy Bridge (2011) | Bulldozer (2011) | `-mavx` |
| AVX2            | 256-bit        | Haswell (2013) | Excavator (2015) | `-mavx2` |
| FMA3            | 256-bit        | Haswell (2013) | Piledriver (2012) | `-mfma` |
| AVX-512         | 512-bit        | Xeon Phi / Skylake-SP (2016) | — (limited Zen 4, 2022) | `-mavx512f` |

Use runtime checks (`std::bitset` over CPUID, or libraries like cpu_features) to select the highest ISA available on the user’s machine.

## Intrinsics Primer

- **SIMD**: Single Instruction, Multiple Data. One instruction, many elements.
- **SSE**: 128-bit vectors (4 floats or 2 doubles).
- **AVX / AVX2**: 256-bit vectors (8 floats or 4 doubles). AVX2 adds full integer support and gather instructions.
- **AVX-512**: 512-bit vectors, masks, more complex ops (not the focus here but worth knowing).

Intrinsics are C/C++ functions that map almost one-to-one to assembly instructions. The header includes give you access to specific instruction sets:

- `<xmmintrin.h>` for SSE
- `<emmintrin.h>` for SSE2
- `<immintrin.h>` for AVX, AVX2, AVX-512 (includes the earlier sets)

Check your compiler flags (`-msse2`, `-mavx`, `-mavx2`) and target CPU before relying on these features.

## Example: Vector Addition with AVX

```cpp
#include <immintrin.h>

void add_avx(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

- `_mm256_loadu_ps` loads 8 floats (unaligned) into a 256-bit register.
- `_mm256_add_ps` performs element-wise addition.
- `_mm256_storeu_ps` writes the result back.

Compilers can auto-vectorize this simple loop, but the intrinsic version makes the vector width explicit and allows plug-in replacements (e.g., fused multiply-add).

## Alignment Matters

Aligned loads/stores (`_mm256_load_ps`) are faster on some CPUs but demand 32-byte aligned pointers. You can align data using C++17 `std::aligned_alloc` or by over-allocating and adjusting the pointer. If alignment is guaranteed, using aligned loads helps the compiler and hardware.

```cpp
float* ptr = static_cast<float*>(std::aligned_alloc(32, n * sizeof(float)));
__m256 data = _mm256_load_ps(ptr); // safe because ptr is aligned
```

## Example: Dot Product with FMA

Fused multiply-add (FMA) instructions combine multiply and add in one step with better precision.

```cpp
#include <immintrin.h>

float dot_avx2(const float* a, const float* b, std::size_t n) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum); // sum += va * vb
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 partial = _mm_add_ps(hi, lo);
    float temp[4];
    _mm_storeu_ps(temp, partial);
    float result = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
```

Notes:

- `_mm256_fmadd_ps` requires FMA3 support (`-mfma` on GCC/Clang).
- Reducing the 256-bit accumulator to a scalar takes a few steps: split into two 128-bit halves, add, store, and finish the reduction in scalar code.

## Integer Operations with AVX2

AVX2 extends full-width operations to 256-bit integer vectors. For example, saturating additions on 8-bit integers:

```cpp
__m256i saturating_add(const std::uint8_t* a, const std::uint8_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    return _mm256_adds_epu8(va, vb);
}
```

- `_mm256_adds_epu8` adds unsigned 8-bit integers with saturation (clamps at 255).
- Use these intrinsics for image processing, signal processing, or any workload that needs predictable overflow behavior.

## Gather and Scatter

AVX2 introduced gathers (`_mm256_i32gather_ps`) which load data from non-contiguous addresses using an index vector. Gathers are powerful for lookup-heavy algorithms but have higher latency—use them when coalescing data isn’t feasible.

```cpp
__m256 gather_example(const float* base, const int* indices) {
    __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices));
    return _mm256_i32gather_ps(base, idx, sizeof(float));
}
```

AVX didn’t provide scatter stores; they arrive with AVX-512. Until then, multithreaded scatter often requires permuting registers and writing scalars in loops.

## When to Reach for Intrinsics

- **Specialized data layouts**: e.g., manual transpose, interleaving, or deinterleaving.
- **Non-standard arithmetic**: saturating math, integer blend, fused operations.
- **Performance critical kernels**: convolution inners loops, financial analytics, DSP.
- **Portability constraints**: you can guard intrinsics with `#ifdef __AVX2__` and fall back to scalar code.

Pitfalls:

- Harder to maintain than scalar or auto-vectorized code.
- Cross-ISA portability (ARM NEON, SVE) requires separate code paths.
- Compiler optimizations may be limited once inline assembly or certain intrinsics appear—profile and inspect generated assembly to confirm gains.

## Tooling Tips

- Use `objdump -d` or Compiler Explorer to verify the vector instructions.
- `std::chrono` microbenchmarks help validate speedups.
- Consider libraries: Intel SVML, OpenCV UMat, Eigen, and xtensor wrap intrinsics behind cleaner APIs.
- For higher-level abstractions, C++20 `std::experimental::simd` (a.k.a. `std::simd`) provides portable vector types that compile down to SSE/AVX instructions.

## Further Reading

- Agner Fog, *Optimizing software in C++*, free online guide.
- Intel Intrinsics Guide (web/ mobile app) for quick instruction lookup.
- Matt Godbolt’s Compiler Explorer for experimenting with different compilers and flags.
- Del Sigala, *An Introduction to Intel AVX-512 Programming*, 2017.

Intrinsics sit at the sweet spot between raw assembly and pure C++: you keep type checking and function calls, but gain precise control over SIMD execution. With practice, you can refactor hot loops to process 8 or 16 elements at a time, delivering speedups that compilers can’t always reach on their own.
