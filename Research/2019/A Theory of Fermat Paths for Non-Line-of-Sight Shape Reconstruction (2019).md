---
title: "A Theory of Fermat Paths for Non-Line-of-Sight Shape Reconstruction (2019)"
aliases: 
  - Fermat Paths NLOS
  - Non-Line-of-Sight Shape Reconstruction
authors:
  - Xiaochun Liu
  - Felix Heide
  - Matthew O'Toole
  - David B. Lindell
  - Gordon Wetzstein
  - Qionghai Dai
  - Wolfgang Heidrich
year: 2019
venue: "CVPR"
doi: "10.1109/CVPR.2019.00127"
arxiv: "https://arxiv.org/abs/1905.02634"
code: "https://github.com/computational-imaging/fermat-paths"
citations: 400+
dataset:
  - Synthetic NLOS datasets
  - Experimental NLOS captures
tags:
  - paper
  - computational-imaging
  - nlos
  - inverse-problems
fields:
  - vision
  - physics-based-vision
  - computational-photography
related:
  - "[[Confocal NLOS Imaging (O’Toole et al., 2018)]]"
  - "[[Transient Imaging Methods]]"
predecessors:
  - "[[NLOS Imaging with Time-of-Flight]]"
successors:
  - "[[Learning-based NLOS Reconstruction]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
This paper introduced a **theoretical framework based on Fermat’s principle of least time** for **non-line-of-sight (NLOS) imaging**, where hidden shapes are reconstructed from indirect light paths. By characterizing **Fermat paths**, the authors derived an exact forward model and provided an efficient algorithm for NLOS reconstruction.

# Key Idea
> Light follows **Fermat paths** (paths of stationary optical length). By modeling hidden object reconstruction in terms of these paths, NLOS imaging can be solved as an **inverse problem** grounded in physics.

# Method
- Developed theory of **Fermat paths** for NLOS: paths that minimize or extremize travel time.  
- Formulated forward model of transient light transport under confocal NLOS setups.  
- Introduced **elliptical tomography** formulation: reconstruction corresponds to inverting ellipsoidal projections.  
- Proposed efficient reconstruction algorithm leveraging the Fermat path structure.  

# Results
- Demonstrated **improved NLOS reconstruction** over back-projection and filtered back-projection methods.  
- Validated framework on synthetic and experimental NLOS datasets.  
- Achieved higher accuracy in recovering hidden shapes with fewer artifacts.  

# Why it Mattered
- First **physics-grounded theory** for NLOS reconstruction.  
- Unified different NLOS approaches under Fermat-path framework.  
- Influenced later physics-inspired and learning-based NLOS imaging methods.  

# Architectural Pattern
- Physics-based inverse problem.  
- Fermat path characterization of indirect light transport.  
- Tomographic inversion algorithm.  

# Connections
- **Contemporaries**: Confocal NLOS imaging (O’Toole 2018).  
- **Influence**: Neural NLOS imaging, hybrid physics+ML approaches.  

# Implementation Notes
- Works best with confocal NLOS setups (laser + single-pixel detectors).  
- Sensitive to noise and multipath effects.  
- Computationally more efficient than naive time-resolved back-projection.  

# Critiques / Limitations
- Requires accurate transient measurements.  
- Limited by temporal resolution of sensors.  
- Struggles with complex materials (scattering, non-Lambertian surfaces).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1905.02634)  
- [Code release](https://github.com/computational-imaging/fermat-paths)  
- [Stanford Computational Imaging Lab resources](https://computationalimaging.org/)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Fermat’s principle, stationary paths.  
- **Geometry**: Ellipses and ellipsoidal tomography.  
- **Physics**: Time-of-flight of light, optics fundamentals.  

## Postgraduate-Level Concepts
- **Inverse Problems**: Tomographic reconstruction from transient data.  
- **Computational Imaging**: Physics-guided algorithms.  
- **Research Methodology**: Combining analytical theory with experimental validation.  
- **Advanced Optimization**: Regularization in ill-posed inverse problems.  

---

# My Notes
- A beautiful example of **physics-based priors guiding vision algorithms**.  
- Open question: Can **diffusion priors for shapes** combine with Fermat path physics for NLOS?  
- Possible extension: Apply Fermat path theory to **video-based indirect light transport editing**.  

---
