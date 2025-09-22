---
layout: post
title: "How Cameras and Eyes See Light: From Spectra to Illuminants"
description: "A step-by-step journey through spectral power distribution, surface reflectance, camera sensitivity, illuminant estimation, and the challenges of multiple light sources."
tags: [computer-vision, color-theory, camera-systems, spectral-analysis]
---

Light is never just "white." It has a shape, a fingerprint, a dance across wavelengths. Cameras record it, eyes interpret it, and both struggle in their own ways. In this post, we'll walk step by step through the chain: spectral power distribution, surface reflectance, camera sensitivity, illuminant estimation, and the knotty problem of multiple light sources.

## Spectral Power Distribution: The Fingerprint of Light
Every light source has its own **spectral power distribution (SPD)**: how much energy it emits at each wavelength.  

- Sunlight → smooth, broad spectrum.  
- Fluorescent lamps → spiky, narrow peaks.  
- White LEDs → sharp blue peak + broad yellow hump.  

**Units:** watts per nanometer (W/nm), telling you how much power is packed into each tiny slice of the spectrum.  

![Spectral Power Distributions of different light sources]({{ '/assets/images/spd_placeholder.png' | relative_url }})

---

## Surface Reflectance: How Materials Shape Light
Objects don’t emit light (unless they’re glowing); they **reflect** it. Reflectance is a ratio:  

$$
R(\lambda) = \frac{\text{Reflected power at }\lambda}{\text{Incident power at }\lambda}
$$ 

- White paper → flat, high reflectance across visible wavelengths.  
- Red apple → high in red, low elsewhere.  
- Asphalt → very low everywhere.  

![Surface reflectance spectra for different materials]({{ '/assets/images/reflectance_placeholder.png' | relative_url }})

---

## Camera Spectral Sensitivity: What Sensors Respond To
Cameras can’t sense wavelength directly — they need **filters**. A Bayer array puts red, green, and blue filters over the sensor. Each channel has its own **spectral sensitivity curve**: broad, overlapping ranges rather than neat slices.  

So for a channel \(c\):  

$$
E_c = \int L(\lambda)\,R(\lambda)\,S_c(\lambda)\,d\lambda
$$  

![Camera spectral sensitivity curves for RGB channels]({{ '/assets/images/camera_sensitivity_placeholder.png' | relative_url }})

---

## Why Multiplication?
Each stage scales the signal:  

- The light source provides energy $L(\lambda)$.  
- The surface keeps only a fraction $R(\lambda)$.  
- The sensor responds with sensitivity $S_c(\lambda)$.  

At each wavelength, multiply them. Then integrate over all wavelengths.  

![Per-channel contribution: L(λ) × R(λ) × S_c(λ)]({{ '/assets/images/multiplication_chain_placeholder.png' | relative_url }})

---

## From Cones to Debayering
The human eye samples color with **three cone types** (S, M, L). Each has its own spectral sensitivity. The brain compares their relative outputs to reconstruct color.  

A camera with a Bayer filter does something similar but clumsier: each pixel sees only one filter (R, G, or B), so it needs **debayering** (interpolating missing channels) to reconstruct full RGB.  

![Human cone sensitivity vs camera Bayer filter comparison]({{ '/assets/images/cones_vs_bayer_placeholder.png' | relative_url }})

---

## Illuminant Estimation: Guessing the Light Source
From a single image, can we infer the color of the light source (the illuminant)?  

Why it’s **ill-posed**:  
- A white sheet under yellow light looks the same as a yellow sheet under white light.  
- Camera gives only three numbers (R, G, B) per pixel, but both illuminant \(L(\lambda)\) and reflectance \(R(\lambda)\) are unknown.  

![The illuminant estimation problem: different lights can produce identical camera responses]({{ '/assets/images/illuminant_problem_placeholder.png' | relative_url }})

---

## The Gray World Assumption
Classic trick: assume that, on average, the world is gray.  

- Compute the mean of each channel across the image.  
- If the averages aren’t equal, the imbalance is blamed on the illuminant.  
- Correct by scaling channels to equalize the averages.  

![Gray World assumption: channel means before and after correction]({{ '/assets/images/grayworld_placeholder.png' | relative_url }})

---

## Minkowski Norm Estimates: A Family of Fixes
Generalize Gray World with the **Minkowski \(p\)-norm**:  

\[
E_c^{(p)} = \left(\frac{1}{N} \sum_i I_c(i)^p \right)^{1/p}
\]  

- \(p=1\): Gray World (average).  
- \(p=\infty\): White Patch (brightest pixel).  
- Intermediate \(p\): Shades of Gray methods.  

![Minkowski p-norm: varying emphasis from mean (p=1) to max (p→∞)]({{ '/assets/images/minkowski_placeholder.png' | relative_url }})

---

## Coping with Limitations
Researchers improved on Gray World:  

- **Smarter stats**: Gray-Edge, Shades-of-Gray.  
- **Region-based**: local estimates, then merging.  
- **World priors**: daylight color curves, known object colors.  
- **Learning-based**: CNNs, transformers, illuminant maps.  

---

## Multiple Illuminants: The Real Mess
Life rarely has one light source. A window and a lamp, fluorescent tubes and skylight — every pixel becomes a blend of multiple SPDs.  

Computer vision strategies:  
- Local estimation per region.  
- Use edges, shadows, specular highlights.  
- Train deep models to output illuminant fields.  
- Sometimes, ask the user (click the white patch in software).  

![Multiple illuminants: example spectral power distributions and mixed lighting]({{ '/assets/images/multiple_lights_placeholder.png' | relative_url }})

---

## How Humans Do It
Our eyes and brains cheat elegantly:  

- **Local adaptation**: different retinal regions recalibrate.  
- **Contextual priors**: bananas are yellow, walls are white.  
- **Attention-driven tuning**: fixating on lamp-lit vs daylight-lit regions shifts balance.  
- **Specular cues**: highlights give away the true illuminant.  

![Human color constancy: how context affects perceived color]({{ '/assets/images/human_constancy_placeholder.png' | relative_url }})

---

## Closing Thoughts
From physics to perception, the chain is the same:  

$$
\text{Pixel value} = L(\lambda) \times R(\lambda) \times S(\lambda)
$$  

The real trick is separating them. Cameras rely on algorithms; humans rely on context and adaptation. Neither is perfect, but both are ingenious.  

---

