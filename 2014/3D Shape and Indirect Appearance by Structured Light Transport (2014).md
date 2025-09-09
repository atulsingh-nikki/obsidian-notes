---
title: "3D Shape and Indirect Appearance by Structured Light Transport (2014)"
aliases: 
  - SLT
  - Structured Light Transport
authors:
  - Matthew O'Toole
  - John Mather
  - Kiriakos N. Kutulakos
year: 2014
venue: "CVPR (Oral)"
doi: "10.1109/CVPR.2014.421"
arxiv: ""
code: ""
citations: 115+
dataset: []
tags:
  - paper
  - structured-light
  - computational-imaging
  - light-transport
fields:
  - vision
  - imaging
related:
  - "[[Inverse Light Transport]]"
  - "[[Time-of-Flight Imaging]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
This paper introduces **Structured Light Transport (SLT)**, a method to separate **direct and indirect light transport** in real time using structured patterns. By leveraging the epipolar geometry between a projector and a camera, the authors design an optical system that captures **indirect-only images and videos**, and enables **shape reconstruction robust to global illumination effects** such as interreflections and subsurface scattering.

# Key Idea
> Epipolar alignment + structured patterns let us cancel direct paths and reveal the indirect light field for 3D shape and appearance recovery.

# Method
- Defines a **light transport matrix** between projector pixels and camera pixels.  
- Uses **epipolar-plane masks** to modulate light such that direct paths align, while indirect transport gets separated.  
- Implements the method optically with two synchronized **DMDs** (one for projection, one for modulation).  
- Supports **indirect-only imaging** and **indirect-robust structured-light 3D reconstruction**.  

# Results
- First demonstration of **indirect-only video** captured at interactive rates.  
- Enables **real-time visualization** of indirect lighting in complex scenes.  
- Recovers 3D shape even in challenging environments with strong interreflections.  

# Why it Mattered
- Overcame a long-standing limitation of structured light: fragility under indirect transport.  
- Pushed computational imaging from **post-processing separation** into **optical-domain separation**.  
- Set the stage for later advances in **non-line-of-sight imaging** and **time-of-flight indirect separation**.  

# Architectural Pattern
- Treat imaging as a **linear light transport system**.  
- Epipolar-based optical coding to isolate direct vs indirect components.  
- Optical pre-processing rather than purely digital computation.  

# Connections
- **Predecessors**: Photometric stereo (Woodham, 1980), Inverse light transport (Nayar et al., 2004).  
- **Contemporaries**: Frequency-based separation, primal-dual light probing.  
- **Successors**: Non-line-of-sight imaging (Velten et al., 2012), time-of-flight imaging with coded separation, neural light transport models.  
- **Influence**: Widely cited in computational photography, indirect imaging, and physics-based vision.  

# Implementation Notes
- Requires high-speed DMDs and precise synchronization with a camera.  
- Heavy reliance on calibration of projector-camera geometry.  
- Computation shifts from digital to optical domain, reducing processing cost but raising hardware complexity.  

# Critiques / Limitations
- Hardware-intensive and not suitable for consumer setups.  
- Limited scalability to very large or highly dynamic scenes.  
- Later methods using ToF cameras or learning-based separation provide more practical pipelines.  

# Repro / Resources
- Paper link: [PDF](https://www.cs.toronto.edu/~kyros/pubs/14.cvpr.slt.pdf)  
- No official code released.  
- Supplementary demos and videos available on project page.  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: light transport matrices, epipolar geometry.  
- **Signals & Systems**: coded patterns as basis functions.  
- **Calculus**: derivatives in transport inversion.  
- **Data Structures**: sparse matrix representations of transport.  

## Postgraduate-Level Concepts
- **Numerical Methods**: stable inversion of transport matrices.  
- **Computational Imaging**: structured illumination and transport coding.  
- **Research Methodology**: controlled experiments separating direct/indirect components.  
- **Neural Extensions**: priors for transport separation using learned models.  

---

# My Notes
- Relevant to my interest in **optical flow + geometry**: transport separation could reduce noise in object selection pipelines.  
- Strong connection to **object removal** and **video editing**—indirect separation may help isolate fine details.  
- Open question: can **deep generative priors** replicate SLT’s separation without heavy optics?  
- Possible extension: combine SLT ideas with **diffusion-based relighting** or **transformer-based scene decomposition**.  
