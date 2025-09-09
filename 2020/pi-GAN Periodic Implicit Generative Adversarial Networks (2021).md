---
title: "pi-GAN: Periodic Implicit Generative Adversarial Networks (2021)"
aliases:
  - pi-GAN
  - Periodic Implicit GAN
authors:
  - Eric R. Chan
  - Marco Monteiro
  - Petr Kellnhofer
  - Jiajun Wu
  - Gordon Wetzstein
year: 2021
venue: "CVPR (Oral)"
doi: "10.1109/CVPR46437.2021.01258"
arxiv: "https://arxiv.org/abs/2012.00926"
code: "https://github.com/marcoamonteiro/pi-GAN"
citations: 1200+
dataset:
  - Cars (ShapeNet)
  - Faces (CelebA-HQ)
  - Cats
tags:
  - paper
  - generative-model
  - gan
  - 3d-aware
  - implicit-representations
fields:
  - vision
  - graphics
  - generative-models
related:
  - "[[NeRF (2020)]]"
  - "[[GIRAFFE (2021)]]"
  - "[[EG3D (2022)]]"
predecessors:
  - "[[NeRF (2020)]]"
successors:
  - "[[GIRAFFE (2021)]]"
  - "[[EG3D (2022)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**pi-GAN** introduced a **3D-aware GAN** that generates images via implicit neural representations and differentiable rendering, building on NeRF. Unlike traditional GANs, pi-GAN learns to represent objects as neural radiance fields, allowing controllable novel view synthesis from unstructured 2D images.

# Key Idea
> Use **implicit neural representations (NeRF)** inside a GAN generator, with a sinusoidal MLP backbone, to learn 3D-aware generative models from only 2D supervision.

# Method
- **Implicit representation**: MLP maps 3D coordinates + viewing direction + latent code → density + color.  
- **Volume rendering**: Differentiable NeRF-style rendering produces 2D images.  
- **Generator**: Implicit MLP with sinusoidal activation functions (SIREN), enabling high-frequency detail.  
- **Discriminator**: Operates in image space (standard GAN).  
- **Training**: Adversarial loss with 2D image datasets only.  

# Results
- Learned **3D-aware generators** from only 2D datasets.  
- Enabled controllable view synthesis (rotate camera, change viewpoint).  
- Produced sharper images than early NeRF-based generative models.  

# Why it Mattered
- First successful **NeRF-in-GAN hybrid** for 3D-aware image synthesis.  
- Showed that GANs can learn 3D structure without 3D supervision.  
- Directly inspired **GIRAFFE (2021)** and **EG3D (2022)**.  

# Architectural Pattern
- SIREN-based MLP implicit generator.  
- NeRF-style volume rendering for image synthesis.  
- GAN training with 2D discriminator.  

# Connections
- Predecessor to **GIRAFFE**, which added compositionality.  
- Complementary to **StyleNeRF and EG3D**, which improved efficiency and fidelity.  
- Related to implicit neural fields in NeRF.  

# Implementation Notes
- Training slower than 2D GANs due to volumetric rendering.  
- Resolution limited (~64–128 px).  
- Public implementation with SIREN MLPs.  

# Critiques / Limitations
- Lower resolution and fidelity than 2D GANs of the time (StyleGAN2).  
- Inefficient rendering compared to later EG3D/tri-plane approaches.  
- No explicit disentanglement between objects and background.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Difference between 2D GANs and 3D-aware GANs.  
- Basics of NeRF and volume rendering.  
- Why sinusoidal activations (SIREN) help model fine details.  

## Postgraduate-Level Concepts
- Implicit neural representations in generative models.  
- Training 3D-aware GANs with only 2D discriminators.  
- Trade-offs: adversarial vs likelihood training for 3D representation learning.  

---

# My Notes
- pi-GAN is the **bridge from NeRF → GAN world**.  
- Clever use of SIREN activations for high-frequency detail.  
- Open question: Can implicit MLP-based GANs scale to **high-resolution photo-realism**?  
- Possible extension: Hybridize pi-GAN with **tri-plane EG3D representations** for efficiency.  

---
