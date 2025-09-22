---
layout: post
title: "Understanding ACES, Scene-Referred Workflows, and Color Space Conversions"
description: "Connect the math of color pipelines to the pictures on screen by following a Rec.709 image through ACES, scene-referred grading, and back to delivery formats."
tags: [color, aces, rec709, color-management, workflows]
---

Color pipelines can feel abstract until you tie each transform to what your eye sees. In this post we follow a Rec.709 image as it moves into ACES, gets graded in a scene-referred workflow, and lands back in delivery-friendly color spaces. Along the way we clarify why ACES exists, how scene-referred thinking differs from display-referred grading, and what role YUV encoding really plays.

## Converting Rec.709 to ACES AP0

Rec.709 defines a comparatively small triangle on the CIE chromaticity diagram and uses a display-referred gamma. ACES AP0, by contrast, is a massive, scene-linear gamut designed to hold *any* color you might encounter on a production.

When you apply the official 709 → ACES Input Transform (IDT), three key things happen:

1. **Linearization** – The gamma-encoded Rec.709 values are converted to linear light so that the numbers once again correspond to scene energy.
2. **Matrix mapping** – The linear RGB values are remapped into the ACES AP0 primaries, placing them inside AP0’s larger triangle.
3. **No magic color boost** – The pixels are numerically different, but the image looks identical because the transform is lossless within the shared gamut.

The important implications:

- No new hues appear out of thin air; ACES cannot invent colors that the original camera never captured.
- Visual differences only appear if the IDT is wrong or if you skip the linearization step.
- Most of AP0 remains empty real estate after the conversion because Rec.709 never occupied that territory.

## Roundtripping 709 → AP0 → 709

A roundtrip through ACES has three phases:

1. **Ingest (709 → AP0)** – The picture should look unchanged. You now have scene-linear values with huge headroom.
2. **Grading in AP0** – You can push exposure, saturation, and hues far beyond the constraints of Rec.709 without clipping. This is where ACES shines: you’re sculpting light, not fighting display limits.
3. **Output (AP0 → 709)** – When you render through the Rec.709 Output Transform (ODT), everything has to squeeze back into the smaller gamut and dynamic range.

Depending on the ODT and your grading decisions, the squeeze can manifest in different ways:

- **Hard clipping** produces flat, neon edges in extreme colors.
- **Saturation compression** rolls off the intensity near the boundary, creating gentler but sometimes desaturated results.
- **Tone mapping** must compress HDR highlights into an SDR envelope, inevitably reducing the sparkle of bright elements.

If you leave the image untouched in ACES, the roundtrip is mathematically lossless. The moment you grade beyond Rec.709 limits, you’re choosing how those extremes fold back into the delivery spec.

## Guardrails to Avoid Surprises

You can catch most nasty surprises before delivery by baking the following checks into your workflow:

1. **Out-of-gamut warnings** – Enable gamut overlays or false colors that flag pixels venturing outside the Rec.709 triangle.
2. **Preview through the right ODT** – Always monitor through the exact Rec.709 ODT you plan to deliver. Looking at “raw” AP0 values is misleading.
3. **Read the scopes** – Use the vectorscope to watch saturation edges and the waveform to spot values heading far above 100 IRE.
4. **Run test exports** – Render a representative clip back to Rec.709 and compare it to your grading timeline so you can iterate before the final pass.

These habits turn ACES from a mystery box into a predictable toolset.

## ACES Is a Framework, Not a Look

It’s tempting to think ACES will automatically make footage look better. In reality its value lies in infrastructure:

- **Standardization** – IDTs, the ACEScg working space, and ODTs give every department a common language for color.
- **Future-proofing** – AP0’s huge gamut ensures today’s grades survive tomorrow’s display technologies.
- **Flexible deliveries** – Grade once in a scene-referred domain and generate Rec.709, HDR10, Dolby Vision, P3, or anything else with consistent intent.
- **Collaboration** – VFX, DI, and finishing teams all see the same reference, reducing guesswork and version churn.

ACES doesn’t dictate style—it just keeps the math honest so your creative choices translate across formats.

## Scene-Referred vs. Display-Referred Thinking

**Scene-referred** imagery represents actual light in the photographed scene. Double the number, double the photons. The range is theoretically limitless, and no contrast curve is baked in.

**Display-referred** imagery is tuned for a particular output device. Middle gray, white, and black are anchored to that display’s capabilities, and the tonal curve bakes in compression to make everything fit.

Why the distinction matters:

- Scene-referred grading preserves real-world relationships between light sources. A candle might sit at 0.001 while direct sun hits 10,000 in linear units—a 10-million-to-one ratio you can later remap to SDR, PQ, or HLG without repainting the scene.
- Once you move to display-referred space, both the candle and sun get squeezed into roughly 0–100 nits (SDR), collapsing their physical contrast. Re-grading for HDR means rebuilding that relationship manually.

Working scene-referred in ACES keeps those original light ratios intact until the last possible moment.

## Where YUV Fits In

YUV is often blamed for the limitations of delivery media, but it’s simply a way to encode color. The real question is whether the values inside the container are scene-referred or display-referred.

- **At the sensor** – Cameras capture linear RGB through photosites. This data is inherently scene-referred.
- **Camera processing**
  - Professional workflows store RAW or log-encoded footage, preserving scene-referred data.
  - Consumer pipelines often convert immediately to YUV 4:2:2 or 4:2:0 with a Rec.709, HLG, or PQ transfer baked in, effectively becoming display-referred right away.
- **Encoding** – Virtually every delivery format (ProRes, DNxHR, H.264, HEVC) uses YUV subsampling for efficiency. If the footage is destined for SDR or HDR displays, the values are display-referred. If it’s a log master like ProRes 4444 XQ storing LogC, the file is technically YUV but still scene-referred because the transfer function encodes scene light.

So the container doesn’t determine the workflow; the transfer function and intent do.

## Putting the Pipeline Together

```
Scene light → Sensor (linear RGB)
           → RAW / Log (scene-referred) OR YUV (display-referred)
           → ACES (scene-referred, wide gamut, floating point)
           → ODT (Rec.709, PQ, HLG, etc.)
           → Encoding (YUV delivery format)
```

Thinking of ACES as the “middle layer” in this pipeline helps keep responsibilities clear: upstream captures reality, ACES maintains it, and ODTs translate it for specific screens.

## Final Thought

ACES doesn’t erase the differences between SDR and HDR, nor does it give you a universal grade for free. What it delivers is consistency and predictability. By understanding where the limits lie—gamut boundaries, tone mapping choices, transfer functions—you can decide how to manage trade-offs instead of discovering them at final delivery.

Embrace ACES as the scaffolding that protects your creative intent, and the math starts working *for* you rather than against you.
