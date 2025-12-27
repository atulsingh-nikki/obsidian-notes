---
layout: post
title: "Bitmap Lookups for Fast YUV → RGB Gamut Decisions"
description: "Precompute RGB gamut membership in a compact bitmap so your YUV pipeline can avoid repeated color conversion and mapping costs."
tags: [color, rendering, optimization]
---

Color pipelines that ingest YUV video frames but render or analyze in RGB often have to answer two expensive questions before any correction work begins:


## Table of Contents

- [Quantizing the YUV lattice](#quantizing-the-yuv-lattice)
- [Precomputing RGB gamut membership](#precomputing-rgb-gamut-membership)
- [Runtime checks in the shader or CPU path](#runtime-checks-in-the-shader-or-cpu-path)
- [Extending the bitmap with auxiliary payloads](#extending-the-bitmap-with-auxiliary-payloads)
- [Memory footprint and cache behavior](#memory-footprint-and-cache-behavior)
- [Putting it together](#putting-it-together)
- [Where this ships today](#where-this-ships-today)

1. **What RGB value would this YUV triplet produce after decoding and tone/gamut transforms?**
2. **Does that RGB point land inside the display gamut, or do we need to apply mapping and clipping?**

Evaluating both for every pixel is costly because the math behind YUV → RGB conversion, opto-electronic transfer functions (OETFs), and gamut mapping is filled with matrix multiplies, non-linear transforms, and conditional logic. A bitmap-based lookup table replaces those repeated computations with a single memory fetch.

## Quantizing the YUV lattice

The first design choice is how finely to sample the continuous Y′UV cube. Suppose we choose 10 bits for luma (Y′) and 8 bits for each chroma axis. That yields a lattice of 2^10 × 2^8 × 2^8 ≈ 16.8 million points—large, but still tractable when stored as a bitmap rather than a dense array of structs.

To index the bitmap:

- Normalize Y′, U, and V to the sampling ranges (for example Y′ ∈ [16, 235] and UV ∈ [16, 240] for BT.601 limited range).
- Quantize each component by rounding to its nearest lattice coordinate.
- Compute a linear index `idx = (y << (u_bits + v_bits)) | (u << v_bits) | v`.

The bitmap bit at `idx` encodes whether the gamut-mapped RGB triple is inside the display gamut.

## Precomputing RGB gamut membership

Building the bitmap is an offline step performed once for each display or color space target:

1. Iterate over every quantized Y′UV triple.
2. Convert to linear RGB using the appropriate matrix and transfer function.
3. Apply gamut mapping (for example, convert to display-referred space and apply a soft-clip or hue-preserving compression).
4. Check if the mapped RGB values remain inside [0, 1] for each channel (or within the display's primary triangle in XYZ space).
5. Write `1` into the bitmap if the triple is inside gamut, `0` otherwise.

The output is a flat `std::vector<uint64_t>` (in C++) or a packed Python `array('Q')`. Each 64-bit word represents 64 lattice points.

```cpp
constexpr uint32_t Y_BITS = 10;
constexpr uint32_t U_BITS = 8;
constexpr uint32_t V_BITS = 8;
constexpr size_t LUT_BITS = size_t{1} << (Y_BITS + U_BITS + V_BITS);
constexpr size_t WORDS = LUT_BITS / 64;
std::array<uint64_t, WORDS> gamut_bitmap{};

for (uint32_t y = 0; y < (1u << Y_BITS); ++y) {
    for (uint32_t u = 0; u < (1u << U_BITS); ++u) {
        for (uint32_t v = 0; v < (1u << V_BITS); ++v) {
            const size_t idx = (size_t(y) << (U_BITS + V_BITS)) |
                               (size_t(u) << V_BITS) |
                               size_t(v);
            RGB rgb = map_to_display(convert_yuv_to_rgb(y, u, v));
            const bool in_gamut = rgb.r >= 0.0f && rgb.r <= 1.0f &&
                                  rgb.g >= 0.0f && rgb.g <= 1.0f &&
                                  rgb.b >= 0.0f && rgb.b <= 1.0f;
            gamut_bitmap[idx >> 6] |= uint64_t(in_gamut) << (idx & 63);
        }
    }
}
```

The conversion and mapping functions are still expensive, but we pay that cost once during the build. Afterward, the runtime path simply indexes the bitmap.

## Runtime checks in the shader or CPU path

On the decoding side, retrieving the gamut membership becomes a few instructions:

1. Quantize incoming Y′UV to the same lattice resolution.
2. Form the linear index.
3. Read the 64-bit word, shift, and mask to extract the single bit.

A CPU implementation can inline these operations; on the GPU, the bitmap can live in constant or texture memory. The key is to align the layout with the hardware's memory coalescing rules. For instance, when processing 8×8 macroblocks, group Y′UV quantization results so that adjacent threads access contiguous bitmap words.

## Extending the bitmap with auxiliary payloads

A single bit tells us whether the gamut mapping is necessary. Sometimes we want more:

- **Nearest in-gamut RGB:** Store a 10-bit index into a secondary LUT that holds precomputed mapped RGB values.
- **Error metric:** Keep a second bitmap that flags when the deltaE exceeds a threshold, signaling that a more sophisticated tone mapper should run.
- **Multiple gamuts:** Maintain a bitmap per target (Rec.709, Display P3, BT.2020) and select at runtime.

These extensions trade memory for richer decisions while preserving the constant-time lookup.

## Memory footprint and cache behavior

For the 16.8-million-point example, the bitmap occupies roughly 2 MB (16.8 Mbits / 8). That easily fits into last-level cache on modern CPUs and into L2 on many GPUs. Compressing further is possible with run-length encoding or Zstd, but simple bitmaps tend to win because the decode cost would reintroduce latency.

To keep cache hit rates high:

- Quantize so that adjacent pixels map to nearby indices.
- Align the bitmap to cache-line boundaries (64-byte multiples).
- Consider tiled or Morton-order indexing if the sampling grid is sparse in certain regions.

## Putting it together

A bitmap lookup table turns the per-pixel gamut query from “matrix multiply + non-linear mapping” into “quantize + bit test.” The approach excels when:

- You have a fixed YUV encoding and display gamut.
- You can afford a one-time offline build.
- The runtime path must be deterministic and real-time (broadcast encoders, live grading tools, game engines).

When those conditions hold, a bitmap-backed precomputation removes the hottest branch from your inner loop, letting the rest of the pipeline focus on tone mapping, spatial filtering, or perceptual tweaks without worrying about color space boundaries.

## Where this ships today

The bitmap-or-LUT pattern is not hypothetical; it underpins multiple production color pipelines:

- **FFmpeg via zimg:** The `zscale` filter relies on the zimg library, which builds packed lookup tables for Y′CbCr ↔ RGB conversions and gamut checks so that repeated frame processing in transcode jobs stays SIMD-friendly.
- **OpenColorIO GPU renderer:** OCIO bakes complex color transforms into 3D textures (bitmaps) uploaded to GLSL shaders, allowing DCC tools like Blender and Foundry Nuke to decide whether pixels land inside a target gamut without recomputing transfer curves.
- **libplacebo / mpv video path:** mpv’s libplacebo backend precomputes gamut-mapping LUTs and uploads them as textures so that HDR-to-SDR tone mapping, particularly in Vulkan compute shaders, becomes a series of table fetches and mask tests.
- **Game engine post pipelines:** Unreal Engine’s filmic tonemapper compiles color grading LUTs into small 3D textures that the renderer samples per pixel to determine clipping and gamut-constrained output when composing UI over HDR scenes.

Each of these stacks treats the LUT as a bitmap of prevalidated color states, demonstrating that the technique scales from media encoders to interactive renderers.
