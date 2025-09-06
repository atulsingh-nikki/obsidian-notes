---
title: "Muse: Text-to-Image Generation via Masked Generative Transformers (2023)"
aliases:
  - Muse
  - Google Muse
authors:
  - Andrew Chang
  - Yuanzhong Xu
  - Prafulla Dhariwal
  - William Chan
  - Chitwan Saharia
  - Mohammad Norouzi
  - David Fleet
  - et al.
year: 2023
venue: "arXiv preprint"
doi: "10.48550/arXiv.2301.00704"
arxiv: "https://arxiv.org/abs/2301.00704"
code: "https://muse-model.github.io/"
citations: 400+
dataset:
  - Large-scale web-scraped image–text pairs (internal Google scale, Imagen/Parti-like)
  - COCO (evaluation)
tags:
  - paper
  - text-to-image
  - masked-modeling
  - transformer
  - generative-models
fields:
  - vision
  - language
  - multimodal
related:
  - "[[Parti (2022)]]"
  - "[[Imagen (2022)]]"
  - "[[Stable Diffusion (2022)]]"
predecessors:
  - "[[Parti (2022)]]"
successors:
  - "[[Next-Gen Google Multimodal Models (2024+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**Muse** is a **text-to-image model** that combines **masked image modeling** with **Transformer-based generation**, offering **faster sampling** and competitive quality compared to diffusion. Unlike diffusion (iterative denoising), Muse generates images in a **few iterative mask-filling steps**, making it significantly more efficient.

# Key Idea
> Treat text-to-image as a **masked token prediction problem** in the discrete image token space, enabling fast, high-quality synthesis with a Transformer conditioned on language embeddings.

# Method
- **Image tokenization**: Images discretized into tokens with a VQ-VAE-like tokenizer.  
- **Model architecture**: Transformer trained with masked modeling (predict missing tokens).  
- **Text conditioning**: Uses frozen large language model embeddings (like Imagen).  
- **Sampling**: Iteratively masks and fills tokens, requiring far fewer steps than diffusion.  
- **Editing**: Inpainting and editing via selective masking.  

# Results
- Competitive with Imagen and Parti in **photorealism** and **text alignment**.  
- Faster sampling than diffusion models (orders of magnitude fewer steps).  
- Strong compositionality from autoregressive/transformer backbone.  

# Why it Mattered
- Showed that **masked modeling Transformers** are a viable alternative to diffusion for text-to-image.  
- Demonstrated **faster inference**, tackling diffusion’s main bottleneck.  
- Extended Google’s multimodal research beyond Imagen and Parti.  

# Architectural Pattern
- Discrete tokens → masked prediction → iterative refinement.  
- Transformer backbone.  
- LLM text embeddings for conditioning.  

# Connections
- Built directly on **Parti** (autoregressive Transformer).  
- Alternative to **diffusion** (Imagen, Stable Diffusion).  
- Related to **BERT-style masked modeling** for generative tasks.  

# Implementation Notes
- Requires large-scale compute for training, but sampling is efficient.  
- Masking strategy critical for balancing speed vs quality.  
- Released demo and paper, but not fully open-sourced models.  

# Critiques / Limitations
- Still closed-source dataset and full models.  
- Tokenization bottleneck (VQ quality) limits ultimate realism.  
- Less explored in community compared to diffusion (due to ecosystem momentum).  

---

# Educational Connections

## Undergraduate-Level Concepts
- What image tokenization is (breaking image into discrete units).  
- Masked modeling vs autoregressive prediction vs diffusion.  
- Why fewer steps → faster generation.  
- Applications: creative design, inpainting, text-based editing.  

## Postgraduate-Level Concepts
- Trade-offs: masked modeling vs diffusion in text-to-image.  
- Scaling laws for token-based Transformers in vision-language generation.  
- Efficiency challenges in generative models.  
- Hybrid pipelines: can masked Transformers + diffusion be combined?  

---

# My Notes
- Muse was Google’s **efficiency play**: faster than diffusion, but closed-source limited adoption.  
- Core insight: **masked modeling scales to image generation** just like BERT scaled NLP.  
- Open question: Can masked modeling methods overtake diffusion if fully open-sourced?  
- Possible extension: Apply Muse-style masked generation to **video tokens** for text-to-video.  

---
