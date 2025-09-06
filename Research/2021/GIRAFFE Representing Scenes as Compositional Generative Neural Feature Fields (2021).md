---
title: "GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields (2021)"
aliases:
  - GIRAFFE
  - Compositional Neural Feature Fields
authors:
  - Michael Niemeyer
  - Andreas Geiger
year: 2021
venue: "CVPR (Best Paper)"
doi: "10.1109/CVPR46437.2021.01096"
arxiv: "https://arxiv.org/abs/2011.12100"
code: "https://github.com/autonomousvision/giraffe"
citations: 1500+
dataset:
  - Cars (ShapeNet)
  - FFHQ (faces)
  - Other unstructured 2D image datasets
tags:
  - paper
  - generative-model
  - 3d-aware
  - neural-fields
  - compositionality
fields:
  - vision
  - graphics
  - generative-models
related:
  - "[[NeRF (2020)]]"
  - "[[pi-GAN (2020)]]"
  - "[[EG3D (2022)]]"
predecessors:
  - "[[pi-GAN (2020)]]"
successors:
  - "[[EG3D (2022)]]"
  - "[[StyleNeRF (2021)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**GIRAFFE** introduced a **3D-aware generative model** that represents a scene as a composition of **neural feature fields**, rather than a monolithic radiance field. This enabled controllable image synthesis from unstructured 2D images, allowing edits like changing object pose, camera viewpoint, or background independently.

# Key Idea
> Represent a scene as a **composition of object-specific neural feature fields** + background field, combined in a 3D-aware rendering process. This compositional design allows disentangled and controllable generation.

# Method
- **Feature fields**: Each object (e.g., car, face) represented by its own neural feature field.  
- **Background field**: Separate neural field for scene background.  
- **Compositional rendering**: Project rays through scene, accumulate features from objects + background, then decode into RGB.  
- **Generative training**: Uses adversarial loss (GAN setup) with only 2D image supervision.  
- **Latent codes**:  
  - Object latent: controls object shape, texture, pose.  
  - Camera latent: controls viewpoint.  
  - Background latent: controls background style.  

# Results
- Learned **3D-aware image generation** from unstructured 2D datasets.  
- Allowed controllable edits (object pose, camera view, object identity).  
- Outperformed prior NeRF-GAN hybrids (e.g., pi-GAN) in disentanglement and quality.  

# Why it Mattered
- First to show **compositional scene representation** with neural fields in a GAN.  
- Enabled **controllable 3D image synthesis** from raw 2D data, without 3D supervision.  
- Influenced later works like StyleNeRF and EG3D (efficient high-quality 3D-aware GANs).  

# Architectural Pattern
- Compositional neural fields (object + background).  
- Differentiable volume rendering.  
- GAN training loop.  

# Connections
- Built on **NeRF (2020)** and **pi-GAN (2020)**.  
- Predecessor to **EG3D (2022)** and **StyleNeRF (2021)**.  
- Parallel to **NeRF-in-GAN** trend for 3D-aware generative modeling.  

# Implementation Notes
- Training only requires **unstructured 2D images** (no 3D ground truth).  
- GAN-based training, not likelihood-based.  
- Public PyTorch implementation available.  

# Critiques / Limitations
- Limited resolution (compared to later EG3D/StyleNeRF).  
- Struggles with complex multi-object scenes beyond toy examples.  
- Quality still below 2D GANs of the time.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What makes a model **3D-aware** vs 2D GANs.  
- How compositionality (separating object vs background) enables control.  
- Basics of volume rendering: rays accumulate information along their path.  
- Why GANs only need 2D images to learn useful representations.  

## Postgraduate-Level Concepts
- Compositional neural feature fields vs monolithic NeRFs.  
- Adversarial training with differentiable rendering pipelines.  
- Disentanglement of latent codes for object identity, pose, and background.  
- Role of inductive biases (object–background separation) in controllable generative modeling.  

---

# My Notes
- GIRAFFE feels like the **first step toward compositional 3D GANs**.  
- NeRF was about *representation*; GIRAFFE turned it into *controllable generation*.  
- Open question: Can compositional fields scale to **open-vocabulary, multi-object scenes** (like text-to-3D)?  
- Possible extension: Use GIRAFFE-style compositionality with **diffusion-based 3D generative models** for more structured controllability.  

---
