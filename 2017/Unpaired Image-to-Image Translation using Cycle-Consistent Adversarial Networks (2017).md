---
title: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (2017)"
aliases: 
  - CycleGAN
  - Cycle-Consistent GAN
authors:
  - Jun-Yan Zhu
  - Taesung Park
  - Phillip Isola
  - Alexei A. Efros
year: 2017
venue: "ICCV"
doi: "10.1109/ICCV.2017.244"
arxiv: "https://arxiv.org/abs/1703.10593"
code: "https://github.com/junyanz/CycleGAN"
citations: 40,000+
dataset:
  - Cityscapes
  - Horse↔Zebra
  - Monet↔Photo
  - Maps↔Satellite
tags:
  - paper
  - gan
  - image-translation
  - unsupervised
fields:
  - vision
  - generative-models
related:
  - "[[Pix2Pix (2017)]]"
  - "[[StyleGAN (2019)]]"
predecessors:
  - "[[Pix2Pix (2017)]]"
successors:
  - "[[MUNIT (2018)]]"
  - "[[CUT (Contrastive Unpaired Translation, 2020)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
CycleGAN introduced **unpaired image-to-image translation** using **cycle-consistency loss**. Unlike Pix2Pix, which required paired training data, CycleGAN learned mappings between two visual domains (e.g., horses ↔ zebras) using only unpaired datasets.

# Key Idea
> Enforce **cycle consistency**: mapping an image from domain A→B and back B→A should reproduce the original image.

# Method
- Two GANs for domain mappings:  
  - Generator \(G: A → B\)  
  - Generator \(F: B → A\)  
- Two discriminators for adversarial training in each domain.  
- **Losses**:  
  - Adversarial loss (real vs fake discrimination).  
  - **Cycle-consistency loss**:  
	$$   
    L_{cyc}(G,F) = E_{x∼A}[||F(G(x)) - x||_1] + E_{y∼B}[||G(F(y)) - y||_1]
    $$ 
  - Optional identity loss to preserve color/style.  

# Results
- Successful translation across many domains (art ↔ photo, summer ↔ winter, maps ↔ satellite).  
- Outperformed Pix2Pix when paired data was unavailable.  
- Produced visually compelling results with unpaired datasets.  

# Why it Mattered
- Made **unsupervised image translation feasible**.  
- Inspired a wave of research in **unpaired generative modeling**.  
- Became a widely used baseline for artistic style transfer, domain adaptation, and medical imaging.  

# Architectural Pattern
- Dual GANs with cycle-consistency constraint.  
- Adversarial + reconstruction hybrid losses.  
- Fully unsupervised training.  

# Connections
- **Contemporaries**: Pix2Pix (paired I2I).  
- **Influence**: MUNIT, DRIT, CUT, diffusion-based unpaired translation.  

# Implementation Notes
- Relies heavily on cycle-consistency; sometimes over-constrains.  
- Identity loss useful for tasks requiring color preservation.  
- Training requires balance between adversarial and cycle losses.  

# Critiques / Limitations
- Struggles with many-to-many mappings (mode collapse).  
- Cannot capture large geometric transformations well.  
- Cycle-consistency may force trivial solutions (near-identity mappings).  
- Later approaches (MUNIT, diffusion) improved diversity and realism.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1703.10593)  
- [Official PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
- [TensorFlow implementations (3rd-party)](https://github.com/vanhuyz/CycleGAN-TensorFlow)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Transformations between vector spaces (domains).  
- **Probability & Statistics**: Adversarial training, distribution alignment.  
- **Optimization Basics**: Balancing multiple loss terms.  

## Postgraduate-Level Concepts
- **Generative Models**: Unpaired adversarial learning.  
- **Neural Network Design**: Dual-generator–dual-discriminator framework.  
- **Computer Vision**: Domain adaptation, style transfer.  
- **Advanced Optimization**: Cycle-consistency vs adversarial loss trade-off.  

---

# My Notes
- Highly relevant to **video domain adaptation** (e.g., day ↔ night, style editing).  
- Open question: Can cycle-consistency be replaced by **contrastive or diffusion-based objectives**?  
- Possible extension: Apply CycleGAN-like training for **temporal style transfer in video**.  
