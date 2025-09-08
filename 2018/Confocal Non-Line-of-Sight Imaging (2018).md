---
title: "Confocal Non-Line-of-Sight Imaging (2018)"
aliases: 
  - Confocal NLOS Imaging
  - NLOS Confocal Reconstruction
authors:
  - Matthew O'Toole
  - David B. Lindell
  - Gordon Wetzstein
year: 2018
venue: "CVPR"
doi: "10.1109/CVPR.2018.00182"
arxiv: "https://arxiv.org/abs/1805.03965"
code: "https://github.com/computational-imaging/confocal-nlos"  
citations: 900+
dataset:
  - Synthetic NLOS data
  - Experimental wall-based NLOS captures
tags:
  - paper
  - nlos
  - computational-imaging
  - tomography
fields:
  - vision
  - physics-based-vision
  - computational-photography
related:
  - "[[A Theory of Fermat Paths for NLOS Reconstruction (2019)]]"
  - "[[Transient Imaging Methods]]"
predecessors:
  - "[[Time-of-Flight NLOS Imaging (Velten et al., 2012)]]"
successors:
  - "[[Fermat Paths NLOS (2019)]]"
  - "[[Learning-based NLOS Reconstruction]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
This paper introduced **Confocal NLOS imaging**, a powerful approach for reconstructing hidden objects using **time-resolved light transport** in a confocal setup (laser and detector share the same point on a visible wall). By leveraging confocal measurements, the reconstruction problem becomes a form of **elliptical tomography**, enabling efficient recovery of hidden 3D geometry.

# Key Idea
> Restricting NLOS measurements to a **confocal configuration** simplifies the reconstruction geometry and enables efficient inversion methods.

# Method
- Setup: Laser illuminates points on a visible wall, detector captures returning light from the same points.  
- Measurements: Time-resolved photon transport encodes hidden scene geometry.  
- Formulation: Reduces NLOS imaging to **elliptical tomography**.  
- Algorithm: Efficient back-projection method reconstructs hidden shapes.  

# Results
- High-quality NLOS reconstructions from real experimental data.  
- Improved robustness compared to non-confocal NLOS setups.  
- Orders of magnitude faster than full time-resolved volumetric reconstruction.  

# Why it Mattered
- Established confocal NLOS as a **standard experimental setup**.  
- Provided theoretical foundation that later works (Fermat paths, neural NLOS) built on.  
- Brought NLOS imaging closer to practical applications in vision and robotics.  

# Architectural Pattern
- Physics-based imaging + tomographic inversion.  
- Confocal measurement simplifies path geometry.  
- Back-projection reconstruction.  

# Connections
- **Contemporaries**: Early NLOS back-projection methods (Velten 2012).  
- **Influence**: Fermat path theory, neural NLOS learning approaches.  

# Implementation Notes
- Requires ultrafast time-of-flight sensors (e.g., SPADs, streak cameras).  
- Confocal restriction trades measurement flexibility for efficiency.  
- Sensitive to sensor noise and limited photon counts.  

# Critiques / Limitations
- Resolution limited by temporal resolution of detectors.  
- Confocal constraint may miss multi-bounce geometry outside main paths.  
- Reconstruction still computationally heavy for large scenes.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1805.03965)  
- [Code release](https://github.com/computational-imaging/confocal-nlos)  
- [Stanford Computational Imaging Lab NLOS resources](https://computationalimaging.org/)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Travel-time of light and path integrals.  
- **Geometry**: Ellipse-based tomography.  
- **Physics**: Fermat’s principle, time-of-flight optics.  

## Postgraduate-Level Concepts
- **Inverse Problems**: Tomographic inversion from time-of-flight data.  
- **Computational Imaging**: Physics-based forward models.  
- **Research Methodology**: Combining hardware and algorithm design.  
- **Advanced Optimization**: Regularization of ill-posed reconstructions.  

---

# My Notes
- Strong complement to **Fermat Paths (2019)** as the experimental/algorithmic precursor.  
- Relevant for **indirect video editing and hidden scene understanding**.  
- Open question: Can **confocal + diffusion priors** give real-time NLOS reconstruction?  
- Possible extension: Integrate confocal NLOS with **neural implicit representations** for video-level hidden geometry capture.  

---
