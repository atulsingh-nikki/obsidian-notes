---
title: "Recovering 3D Shape Around a Corner Using Ultrafast Time-of-Flight Imaging (2012)"
aliases: 
  - First NLOS Imaging
  - Velten NLOS 2012
authors:
  - Andreas Velten
  - Thomas Willwacher
  - Otkrist Gupta
  - Ashok Veeraraghavan
  - Moungi Bawendi
  - Ramesh Raskar
year: 2012
venue: "Nature Communications"
doi: "10.1038/ncomms1747"
arxiv: "https://arxiv.org/abs/1105.1160"
code: "https://www.media.mit.edu/research/nlos/" # no official open code
citations: 3000+
dataset:
  - Experimental wall-based captures (ultrafast laser + streak camera)
tags:
  - paper
  - nlos
  - computational-imaging
  - time-of-flight
fields:
  - vision
  - physics-based-vision
  - computational-photography
related:
  - "[[Confocal NLOS Imaging (2018)]]"
  - "[[Fermat Paths NLOS (2019)]]"
predecessors:
  - "[[Time-of-Flight Cameras (Structured Light, 2000s)]]"
successors:
  - "[[Confocal NLOS Imaging (2018)]]"
  - "[[Learning-based NLOS Reconstruction]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
This work marked the **birth of NLOS (Non-Line-of-Sight) imaging**, demonstrating that it is possible to reconstruct **hidden 3D geometry around a corner** using ultrafast time-of-flight imaging. By illuminating a wall with a femtosecond laser and capturing returning light with a streak camera, the team recovered hidden objects beyond direct line of sight.

# Key Idea
> Use **time-resolved photon transport** and back-projection tomography to reconstruct 3D shapes of objects hidden from direct view.

# Method
- **Setup**:  
  - Femtosecond pulsed laser illuminates a visible wall.  
  - Streak camera records returning light at picosecond resolution.  
  - Captured light includes multiple bounces: laser → wall → hidden object → wall → camera.  
- **Algorithm**:  
  - Back-project light paths consistent with photon travel times.  
  - Accumulate reconstructions into a 3D voxel grid.  
- Essentially performs **elliptical tomography** using time-of-flight ellipsoids.  

# Results
- First demonstration of hidden shape reconstruction from indirect light.  
- Reconstructed complex shapes (e.g., foam cutouts, toy objects) placed out of line-of-sight.  
- Proved feasibility of **seeing around corners**.  

# Why it Mattered
- Established **NLOS imaging** as a research field.  
- Inspired physics-based, confocal, Fermat-path, and deep learning NLOS methods.  
- Demonstrated the fusion of **ultrafast optics + computational reconstruction**.  

# Architectural Pattern
- Ultrafast optics hardware + tomographic back-projection.  
- Ellipsoidal path geometry from time-of-flight.  
- Voxel reconstruction.  

# Connections
- **Contemporaries**: Early transient imaging (Femto-photography, Raskar 2011).  
- **Influence**: Confocal NLOS imaging, Fermat path theory, neural NLOS.  

# Implementation Notes
- Requires expensive femtosecond lasers and streak cameras.  
- Very sensitive to noise and alignment.  
- Reconstruction slow (minutes to hours).  

# Critiques / Limitations
- Not real-time; impractical outside lab environments.  
- Requires specialized ultrafast hardware.  
- Reconstructions low resolution compared to later methods.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1105.1160)  
- [MIT Media Lab project page](https://www.media.mit.edu/research/nlos/)  
- [Follow-up: Confocal NLOS Imaging (2018)](https://arxiv.org/abs/1805.03965)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Travel-time computations of photons.  
- **Geometry**: Ellipsoids defined by time-of-flight constraints.  
- **Physics**: Fermat’s principle, ultrafast optics.  

## Postgraduate-Level Concepts
- **Inverse Problems**: Tomographic back-projection.  
- **Computational Imaging**: Physics + computation for hidden vision.  
- **Research Methodology**: Feasibility demonstration → new field creation.  
- **Advanced Optimization**: Dealing with noise in ultrafast photon measurements.  

---

# My Notes
- This is the **origin story of NLOS imaging** → physics + algorithms in synergy.  
- Connects directly to my interests in **video editing from indirect cues**.  
- Open question: Can NLOS move from ultrafast optics to **commodity sensors** (event cameras, SPADs, diffusion priors)?  
- Possible extension: Apply NLOS + diffusion priors for **hidden video generation**.  

---
