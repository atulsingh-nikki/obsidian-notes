---
title: Bayesian Color Constancy (2008)
aliases:
  - Bayesian AWB
  - Bayesian Color Constancy
authors:
  - Peter Gehler
  - Carsten Rother
  - Andrew Blake
  - Thomas Minka
  - Toby Sharp
year: 2008
venue: NIPS
doi: 10.5555/2981780.2981920
arxiv: ""
citations: 1500+
dataset:
  - Gehler-Shi dataset (later refined in 2010 by Shi & Funt)
tags:
  - paper
  - color-constancy
  - white-balance
  - bayesian-inference
fields:
  - vision
  - computational-photography
  - image-processing
related:
  - "[[Fast Fourier Color Constancy (2017) 2|Fast Fourier Color Constancy]]"
  - "[[Convolutional Color Constancy (2015)|Convolutional Color Constancy]]"
predecessors:
  - "[[Gray World Assumption (1980)]]"
successors:
  - "[[Fast Fourier Color Constancy (2017)|Fast Fourier Color Constancy]]"
  - "[[Convolutional Color Constancy (2015)|Convolutional Color Constancy]]"
impact: ⭐⭐⭐⭐☆
status: read
---

# Summary
**Bayesian Color Constancy (Gehler et al., NIPS 2008)** cast white balance as a **Bayesian inference problem**. Given an observed image, the algorithm infers the posterior distribution over possible illuminants by combining a **likelihood model of scene reflectance** with a **prior over illuminants**.

# Key Idea
 Formulate color constancy as **Bayes’ rule**:  

$$
p(\text{illuminant} \mid \text{image}) \propto p(\text{image} \mid \text{illuminant}) \cdot p(\text{illuminant})
$$

Bayesian Color Constancy (2008)
Bayesian Color Constancy (2008)  
The estimate = illuminant maximizing this posterior.

# Method
- **Likelihood**: learned distribution of reflectances under candidate illuminants.  
- **Prior**: captures common illuminant colors (e.g., daylight).  
- **Inference**: MAP (maximum a posteriori) estimate of illuminant.  
- **Dataset**: introduced the **Gehler dataset** of 568 images (later refined by Shi & Funt in 2010).  

# Results
- Significantly improved accuracy over heuristic methods (e.g., Gray World, White Patch).  
- First large-scale **learning-based** color constancy approach.  
- Dataset became a **benchmark standard** for AWB.  

# Why it Mattered
- Established **Bayesian formulation** as the dominant probabilistic model for AWB.  
- Dataset remains the most cited benchmark for color constancy.  
- Provided baseline that CCC (2015) and FFCC (2017) improved upon.  

# Architectural Pattern
- Bayesian generative model.  
- Posterior inference → illuminant estimation.  

# Connections
- Builds on heuristic methods (Gray World, White Patch, Gamut Mapping).  
- Predecessor to discriminative approaches (CCC, FFCC).  
- Related to probabilistic low-level vision modeling.  

# Implementation Notes
- Training requires reflectance data under known illuminants.  
- Computation slower than heuristics but tractable.  
- Provided ground for dataset-driven research.  

# Critiques / Limitations
- Assumes a **single global illuminant**.  
- Priors may bias toward “typical” illuminants, hurting rare cases.  
- Superseded in accuracy by discriminative (CCC) and FFT methods (FFCC).  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Bayes’ Rule**: combine prior knowledge + observed evidence.  
- Color constancy = estimating “true white” of the scene.  
- Example: guessing light source color by balancing object colors.  

## Postgraduate-Level Concepts
- Bayesian generative models in vision.  
- Likelihood modeling of reflectance distributions.  
- MAP inference vs full posterior over illuminants.  
- Limitations of single-illuminant assumptions.  

---

# My Notes
- Bayesian Color Constancy = **first big dataset + probabilistic AWB model**.  
- Strength: principled Bayesian framing; Weakness: computational + single-illuminant assumption.  
- Still historically central because of the **Gehler-Shi dataset**.  

---
