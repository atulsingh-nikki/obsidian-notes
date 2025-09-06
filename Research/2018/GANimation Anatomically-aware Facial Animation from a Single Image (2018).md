---
title: "GANimation: Anatomically-aware Facial Animation from a Single Image (2018)"
aliases: 
  - GANimation
  - Facial Animation with GANs
authors:
  - Albert Pumarola
  - Antonio Agudo
  - Alberto Sanfeliu
  - Francesc Moreno-Noguer
year: 2018
venue: "ECCV"
doi: "10.1007/978-3-030-01219-9_30"
arxiv: "https://arxiv.org/abs/1807.09251"
code: "https://github.com/albertpumarola/GANimation"
citations: 1000+
dataset:
  - CelebA
  - EmotioNet
tags:
  - paper
  - gan
  - facial-animation
  - image-to-image
fields:
  - vision
  - graphics
  - generative-models
related:
  - "[[Pix2Pix (2017)]]"
  - "[[StarGAN (2018)]]"
predecessors:
  - "[[Pix2Pix (2017)]]"
successors:
  - "[[StarGAN (2018)]]"
  - "[[First Order Motion Model (2019)]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
GANimation introduced a framework for **controllable facial animation** from a single still image using **GANs** guided by **Facial Action Units (AUs)**. Unlike categorical emotion models (happy, sad, angry), GANimation operates on anatomically-based AUs, enabling **fine-grained, realistic facial expression control**.

# Key Idea
> Drive facial animation using **action units (AUs)**, which correspond to anatomical muscle movements, and use adversarial training to produce realistic, identity-preserving expressions.

# Method
- **Inputs**:  
  - A still face image.  
  - Target **AU vector** (intensity of facial muscle movements).  
- **Generator**: Produces animated face matching AU controls.  
- **Discriminator**: Distinguishes real vs fake expressions, conditioned on AUs.  
- **Losses**:  
  - Adversarial loss.  
  - Identity-preserving loss.  
  - Attention mask loss (focus modifications only on expression-related regions).  
  - AU regression loss to ensure anatomical consistency.  

# Results
- Produced highly realistic facial animations across **CelebA** and **EmotioNet** datasets.  
- Outperformed previous categorical-emotion GANs in realism and controllability.  
- Demonstrated interpolation of expressions by blending AU intensities.  

# Why it Mattered
- Shifted facial animation from **categorical emotions → anatomically grounded AU control**.  
- Enabled **continuous and fine-grained expression synthesis**.  
- Foundation for later controllable face synthesis models.  

# Architectural Pattern
- Conditional GAN with AU vector conditioning.  
- Attention-based generator focusing on expression-relevant regions.  
- Multi-loss training for realism + identity + control.  

# Connections
- **Contemporaries**: StarGAN (multi-domain translation), Face2Face (graphics-based).  
- **Influence**: First Order Motion Model (2019), face reenactment and deepfake research.  

# Implementation Notes
- Requires annotated AUs for training.  
- Attention mask avoids global artifacts.  
- Identity-preserving constraints critical for robustness.  

# Critiques / Limitations
- Relies on accurate AU labels (annotation bottleneck).  
- Struggles with extreme poses or occlusions.  
- Limited temporal consistency (frame-by-frame generation).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1807.09251)  
- [Official PyTorch implementation](https://github.com/albertpumarola/GANimation)  
- [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: AU vectors as feature encodings.  
- **Probability & Statistics**: Adversarial training dynamics.  
- **Optimization Basics**: Multi-objective GAN loss balancing.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Conditional GAN with attention.  
- **Computer Vision**: Facial Action Coding System (FACS).  
- **Research Methodology**: Evaluation on realism vs control trade-offs.  
- **Advanced Optimization**: Handling disentangled, continuous control spaces.  

---

# My Notes
- Connects to **controllable character animation** for creative video tools.  
- Open question: Can AU-based control be integrated with **diffusion models** for smoother temporal editing?  
- Possible extension: Extend GANimation to **video facial reenactment with AU-driven dynamics**.  

---
