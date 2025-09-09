---
title: "Bringing Old Photos Back to Life (2020)"
aliases:
  - Old Photo Restoration
  - Triplet Domain Translation for Photo Restoration
authors:
  - Ziyu Wan
  - Bo Zhang
  - Dongdong Chen
  - Pan Zhang
  - Dong Chen
  - Fang Wen
  - Baining Guo
year: 2020
venue: "CVPR (Oral)"
doi: "10.1109/CVPR42600.2020.01118"
arxiv: "https://arxiv.org/abs/2004.09484"
code: "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life"
citations: ~1200+
dataset:
  - Real old photo collections
  - Synthetic degraded photo dataset
tags:
  - paper
  - restoration
  - generative-model
  - domain-translation
  - unsupervised
fields:
  - vision
  - generative-models
  - image-restoration
related:
  - "[[CycleGAN (2017)]]"
  - "[[Image Inpainting Methods]]"
  - "[[Diffusion-based Restoration (2022+)]]"
predecessors:
  - "[[CycleGAN (2017)]]"
successors:
  - "[[Diffusion-based Image Restoration Models (2022+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**Bringing Old Photos Back to Life** proposed a **triplet domain translation network** for old photo restoration. It addressed complex degradations such as scratches, fading, blurriness, and noise, by bridging the **simulation-to-real gap** in latent space between synthetic training data and real old photos.

# Key Idea
> Use a **triplet domain translation framework** (synthetic degraded → clean, real degraded → synthetic degraded, real degraded → clean) to bridge the sim-to-real gap for old photo restoration.

# Method
- **Triplet framework**:  
  - Domain 1: Synthetic degraded photos (with ground truth).  
  - Domain 2: Real degraded photos (no ground truth).  
  - Domain 3: Clean modern photos.  
- **Networks**:  
  - Restoration generator trained on synthetic-clean pairs.  
  - Domain adaptation network aligns real degraded with synthetic degraded in latent space.  
- **Losses**:  
  - Reconstruction loss (synthetic pairs).  
  - Adversarial + cycle consistency losses for domain alignment.  
  - Perceptual and identity losses for realism.  

# Results
- Produced **state-of-the-art restorations** of real old photos with scratches, stains, and complex degradations.  
- Outperformed prior GAN/inpainting-based approaches.  
- Released large-scale real old photo dataset and restoration toolbox.  

# Why it Mattered
- First major work to **tackle real-world old photo restoration at scale**.  
- Introduced **triplet domain translation** as a way to handle sim-to-real gaps.  
- Widely adopted in practical photo restoration applications.  

# Architectural Pattern
- GAN-based restoration network.  
- Triplet domain latent alignment.  
- Adversarial + perceptual training.  

# Connections
- Related to **CycleGAN** (domain translation).  
- Predecessor to diffusion-based image restoration.  
- Complementary to inpainting and deblurring literature.  

# Implementation Notes
- Training requires synthetic degraded → clean pairs for supervised signal.  
- Real degraded → synthetic degraded adaptation key to generalization.  
- Official code and dataset released by Microsoft.  

# Critiques / Limitations
- Struggles with extreme degradations (missing large regions).  
- Training pipeline complex due to triplet setup.  
- Limited diversity in synthetic degradation simulation.  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Image degradation types**: blur, scratches, fading, and noise in old photos.  
- **GAN basics**: how a generator–discriminator setup works for producing clean images.  
- **Domain gap**: why models trained on synthetic degraded photos may fail on real old photos.  
- **Cycle-consistency**: ensuring translations between domains don’t lose core identity.  

## Postgraduate-Level Concepts
- **Triplet domain translation**: how synthetic–real–clean mapping helps bridge sim-to-real gaps.  
- **Loss design in restoration**: reconstruction loss for synthetic pairs, adversarial losses for realism, perceptual losses for fine detail.  
- **Latent space alignment**: mapping real degraded images into the synthetic degradation space before restoration.  
- **Evaluation metrics**: FID, PSNR, SSIM for restoration performance on old photos.  


---

# My Notes
- This work stands out because it’s **practical and impactful**: real-world photo restoration.  
- Clever use of sim-to-real adaptation via latent space triplet alignment.  
- Open question: Can diffusion-based generative priors now **outperform GAN-based triplet approaches**?  
- Possible extension: Plug in pretrained foundation diffusion models for **universal photo restoration pipelines**.  

---
