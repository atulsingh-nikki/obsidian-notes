---
title: "Parti: Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (2022)"
aliases:
  - Parti
  - Google Parti
authors:
  - Yuanzhong Xu
  - Chitwan Saharia
  - William Chan
  - Saurabh Saxena
  - David Fleet
  - Mohammad Norouzi
  - et al.
year: 2022
venue: "arXiv preprint"
doi: "10.48550/arXiv.2206.10789"
arxiv: "https://arxiv.org/abs/2206.10789"
code: "https://parti-research-demo.github.io/"
citations: 500+
dataset:
  - Large-scale web-scraped image–text pairs (internal Google scale)
  - COCO (evaluation)
tags:
  - paper
  - text-to-image
  - autoregressive
  - transformer
  - generative-models
fields:
  - vision
  - language
  - multimodal
related:
  - "[[Imagen (2022)]]"
  - "[[DALL·E (2021)]]"
  - "[[Stable Diffusion (2022)]]"
predecessors:
  - "[[DALL·E (2021)]]"
successors:
  - "[[Muse (2023)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**Parti** (short for *Pathways Autoregressive Text-to-Image*) explored an alternative to diffusion: using **autoregressive Transformers** trained on image tokens to achieve high-quality **text-to-image generation**. By scaling model size and training data, Parti achieved **competitive photorealism and compositional ability**, rivaling Imagen.

# Key Idea
> Treat text-to-image as a **sequence modeling problem** over **discrete image tokens**, scaling autoregressive Transformers with large data to match diffusion models in quality.

# Method
- **Image tokenization**: Discrete image patches using VQ-VAE / tokenizer.  
- **Architecture**: Large-scale Transformer trained autoregressively to map text tokens → image tokens.  
- **Text conditioning**: Uses pretrained large language models for embeddings.  
- **Generation**: Images decoded from token sequences using VQ decoder.  

# Results
- Strong compositional ability (complex text prompts).  
- Achieved photorealism comparable to Imagen.  
- Performed well in **human preference studies** on COCO captions.  

# Why it Mattered
- Showed autoregressive approaches could **still compete with diffusion** when scaled.  
- Highlighted trade-offs: autoregressive better at **fine compositional structure**, diffusion better at **photorealism**.  
- Part of Google’s exploration of multimodal foundation models.  

# Architectural Pattern
- Text embeddings from large language model.  
- Autoregressive Transformer for token prediction.  
- VQ decoder reconstructs images.  

# Connections
- Imagen’s sibling model: diffusion vs autoregressive exploration.  
- Direct successor to **DALL·E (2021)**.  
- Predecessor to **Muse (2023)**, which improved efficiency with masked modeling.  

# Implementation Notes
- Requires large-scale compute.  
- Autoregressive generation slower than diffusion for large images.  
- Demo examples released by Google, but not fully open-sourced.  

# Critiques / Limitations
- Slower sampling compared to diffusion.  
- Limited resolution compared to Imagen’s cascaded diffusion.  
- Like Imagen, closed-source and dataset undisclosed.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What autoregressive models are (predict next token).  
- How images can be represented as sequences of discrete tokens.  
- Difference between autoregressive and diffusion generation.  
- Applications: creative content generation from text.  

## Postgraduate-Level Concepts
- VQ-VAE tokenization trade-offs (discretization errors vs efficiency).  
- Scaling laws for autoregressive vs diffusion models.  
- Compositional generalization in text-to-image generation.  
- Comparative analysis: when to use autoregression vs diffusion for multimodal tasks.  

---

# My Notes
- Parti was a **counterpoint to Imagen**: showed autoregressive models weren’t obsolete.  
- Reinforced idea: **scale matters more than architecture**, though trade-offs exist.  
- Open question: Can autoregressive approaches match diffusion in **speed + quality** with hybrid tokenization?  
- Possible extension: Combine Parti’s compositional strength with diffusion’s realism (hybrid generative pipeline).  

---
