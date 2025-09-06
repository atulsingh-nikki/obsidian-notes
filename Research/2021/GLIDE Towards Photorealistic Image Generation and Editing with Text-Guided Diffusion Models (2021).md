---
title: "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models (2021)"
aliases:
  - GLIDE (2021)
  - Guided Diffusion with Language
authors:
  - Alex Nichol
  - Prafulla Dhariwal
  - Aditya Ramesh
  - Pranav Shyam
  - Pamela Mishkin
  - Bob McGrew
  - Ilya Sutskever
  - Mark Chen
year: 2021
venue: "arXiv preprint"
doi: "10.48550/arXiv.2112.10741"
arxiv: "https://arxiv.org/abs/2112.10741"
code: "https://github.com/openai/glide-text2im"
citations: 1000+
dataset:
  - 250M image–text pairs (internal dataset)
  - COCO (evaluation)
tags:
  - paper
  - diffusion
  - text-to-image
  - image-editing
  - guidance
fields:
  - vision
  - language
  - generative-models
related:
  - "[[Diffusion Models Beat GANs (2021)]]"
  - "[[DALL·E (2021)]]"
  - "[[Imagen (2022)]]"
  - "[[Stable Diffusion (2022)]]"
predecessors:
  - "[[Diffusion Models Beat GANs (2021)]]"
  - "[[DALL·E (2021)]]"
successors:
  - "[[Imagen (2022)]]"
  - "[[Stable Diffusion (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**GLIDE** was the first large-scale **text-guided diffusion model**, capable of **photorealistic text-to-image generation and image editing**. It introduced **classifier-free guidance** as a simpler, more effective alternative to classifier guidance, laying the foundation for modern diffusion-based image generation.

# Key Idea
> Extend diffusion models with **text conditioning** and replace classifier guidance with **classifier-free guidance**, enabling both high-quality **text-to-image generation** and **text-based editing**.

# Method
- **Base model**: Improved diffusion model (U-Net backbone).  
- **Text conditioning**: Incorporated language embeddings into the diffusion denoiser.  
- **Classifier-free guidance**:  
  - Train the model with and without conditioning.  
  - At sampling, interpolate between unconditional and conditional predictions.  
- **Tasks**:  
  - Text-to-image synthesis.  
  - Inpainting (image editing with text prompts).  

# Results
- Generated **more photorealistic and diverse images** than DALL·E (2021).  
- Enabled **text-guided inpainting** for image editing.  
- Classifier-free guidance improved controllability without training an external classifier.  

# Why it Mattered
- First demonstration of **text-guided diffusion** at scale.  
- Classifier-free guidance became the **standard conditioning method** in diffusion research.  
- Direct precursor to **Imagen** and **Stable Diffusion**.  

# Architectural Pattern
- U-Net diffusion backbone with cross-attention text conditioning.  
- Classifier-free guidance for controlled sampling.  

# Connections
- Built upon **Diffusion Models Beat GANs (2021)**.  
- Complementary to **DALL·E (2021)** (autoregressive text-to-image).  
- Successor to be surpassed by **Imagen (2022)** and **Stable Diffusion (2022)**.  

# Implementation Notes
- Training required very large-scale text–image data.  
- Guidance scale hyperparameter balances fidelity vs diversity.  
- Public demo and code sparked interest in open-source diffusion.  

# Critiques / Limitations
- Still limited resolution compared to later works.  
- Dataset biases present in generations.  
- Sampling speed remained slow (hundreds of steps).  

---

# Educational Connections

## Undergraduate-Level Concepts
- What diffusion models are, and how they differ from GANs.  
- What "guidance" means in conditional generation.  
- Basics of text-to-image synthesis.  
- Why inpainting is a form of conditional generation.  

## Postgraduate-Level Concepts
- Classifier-free guidance vs classifier-based guidance: trade-offs.  
- Cross-attention as a mechanism for text conditioning.  
- Diffusion models as **universal generative priors** (usable for both synthesis and editing).  
- Scaling implications: moving from research demos to production-grade systems.  

---

# My Notes
- GLIDE was the **bridge**: from generic diffusion models → text-guided diffusion.  
- Classifier-free guidance = elegant and hugely influential.  
- Open question: How to make GLIDE’s inpainting **real-time** for creative tools?  
- Possible extension: Fuse GLIDE-like conditioning with **video diffusion** for controllable video editing.  

---
