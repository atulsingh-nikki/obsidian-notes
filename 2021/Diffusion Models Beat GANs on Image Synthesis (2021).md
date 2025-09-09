---
title: "Diffusion Models Beat GANs on Image Synthesis (2021)"
aliases:
  - Diffusion Models Beat GANs
  - ADM (Improved Diffusion Models)
authors:
  - Prafulla Dhariwal
  - Alex Nichol
year: 2021
venue: "NeurIPS"
doi: "10.48550/arXiv.2105.05233"
arxiv: "https://arxiv.org/abs/2105.05233"
code: "https://github.com/openai/guided-diffusion"
citations: 694+
dataset:
  - CIFAR-10
  - ImageNet (64×64, 128×128, 256×256)
tags:
  - paper
  - diffusion
  - generative-models
  - image-synthesis
  - guidance
fields:
  - vision
  - generative-models
related:
  - "[[DDPM (2020)]]"
  - "[[DDIM (2020)]]"
  - "[[Classifier-Free Guidance (2022)]]"
predecessors:
  - "[[DDPM (2020)]]"
  - "[[GANs (2014–2020)]]"
successors:
  - "[[GLIDE (2021)]]"
  - "[[Imagen (2022)]]"
  - "[[Stable Diffusion (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
This paper demonstrated that **diffusion models**, with the right architectural improvements and **classifier guidance**, could **outperform GANs** on standard image synthesis benchmarks. It marked a **major turning point** in generative modeling, establishing diffusion as the new state-of-the-art approach.

# Key Idea
> Combine improved U-Net architectures with **classifier-guided diffusion sampling** to achieve **FID scores better than GANs**, proving diffusion models’ superior scalability and image quality.

# Method
- **Base model**: Denoising Diffusion Probabilistic Models (DDPM).  
- **Architecture**: U-Net backbone with attention and improved residual blocks.  
- **Classifier guidance**: Train a separate classifier; during sampling, guide diffusion toward desired class by modifying gradients.  
- **Training**: Large-scale training on CIFAR-10 and ImageNet at multiple resolutions.  

# Results
- Achieved **SOTA FID scores** on CIFAR-10 and ImageNet, surpassing BigGAN.  
- Generated high-quality, diverse, and stable images.  
- Proved diffusion models scale better than GANs without training instabilities.  

# Why it Mattered
- First clear evidence that **diffusion models could surpass GANs** in image generation quality.  
- Marked the **shift of generative modeling research** toward diffusion and away from GAN-dominance.  
- Foundations for later text-to-image models (GLIDE, Imagen, Stable Diffusion).  

# Architectural Pattern
- U-Net with attention layers.  
- Diffusion training with noise schedules.  
- Classifier guidance at inference time.  

# Connections
- Predecessor: **DDPM (2020)** introduced diffusion models.  
- Successors: **GLIDE, Imagen, Stable Diffusion** — added text-conditioning and classifier-free guidance.  
- Complementary to **GANs** but more stable and scalable.  

# Implementation Notes
- Sampling slower than GANs but higher fidelity.  
- Classifier guidance boosts quality but requires extra model.  
- OpenAI released **guided-diffusion code** widely used in research.  

# Critiques / Limitations
- Sampling very slow compared to GANs (hundreds of steps).  
- Classifier guidance adds extra training overhead.  
- Resolution limited to 256×256 at the time.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What diffusion models are (progressive denoising from noise).  
- Difference between GANs and diffusion models.  
- FID score: measuring realism of generated images.  
- Why classifier guidance improves sample quality.  

## Postgraduate-Level Concepts
- Noise schedule design and U-Net backbone for diffusion.  
- Classifier-guided vs classifier-free guidance.  
- Comparison of GAN instabilities vs diffusion stability.  
- How this paper set the stage for conditional and text-guided diffusion.  

---

# My Notes
- This paper was the **turning point**: diffusion > GANs.  
- OpenAI’s release made it the **de facto reference implementation**.  
- Open question: Can diffusion be made **fast enough** for real-time generation? (later answered by DDIM, Latent Diffusion, etc.).  
- Possible extension: Apply classifier-free guidance + acceleration to push diffusion into interactive domains (art, video, AR).  

---
