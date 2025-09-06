---
title: "VideoBERT: A Joint Model for Video and Language Representation Learning (2019)"
aliases:
  - VideoBERT
  - Video-Language BERT
authors:
  - Chen Sun
  - Austin Myers
  - Carl Vondrick
  - Kevin Murphy
  - Cordelia Schmid
year: 2019
venue: "ICCV (Oral)"
doi: "10.1109/ICCV.2019.00948"
arxiv: "https://arxiv.org/abs/1904.01766"
code: "https://github.com/google-research-datasets/YouCook2"
citations: 3000+
dataset:
  - YouCookII (video-text dataset)
  - HowTo100M
tags:
  - paper
  - video-language
  - transformer
  - multimodal
  - pretraining
fields:
  - vision
  - language
  - multimodal
related:
  - "[[BERT (2018)]]"
  - "[[ViLBERT (2019)]]"
  - "[[ClipBERT (2021)]]"
predecessors:
  - "[[BERT (2018)]]"
successors:
  - "[[ClipBERT (2021)]]"
  - "[[Video-Language Pretraining (2022+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**VideoBERT** was one of the first attempts to extend **BERT-style transformers** to **video + language joint modeling**. It treated video as a sequence of discrete "visual tokens" (quantized features) and trained a transformer with masked language modeling and cross-modal prediction tasks.

# Key Idea
> Extend BERT from text-only to **video-text sequences**, by representing video clips as discrete tokens and learning joint contextual embeddings with language.

# Method
- **Video tokenization**: Extract visual features from video clips and discretize them into tokens using vector quantization.  
- **Language modeling**: Use a BERT-style transformer trained with masked prediction tasks.  
- **Cross-modal learning**: Train on both text-only, video-only, and video+text data.  
- **Pretraining corpora**: HowTo100M and YouCookII cooking videos.  

# Results
- Learned joint video-language embeddings without supervision.  
- Improved performance on video captioning and retrieval tasks.  
- Showed feasibility of large-scale video-text pretraining.  

# Why it Mattered
- One of the first **transformer-based multimodal pretraining models** for video and language.  
- Proved that BERT could extend beyond text to video understanding.  
- Inspired later efficient approaches like **ClipBERT**.  

# Architectural Pattern
- Discretized video tokens → transformer encoder (BERT).  
- Masked prediction across modalities.  
- Joint representation space for video and text.  

# Connections
- Predecessor to **ClipBERT (2021)**, which simplified with sparse sampling.  
- Related to **ViLBERT (2019)** (vision-language transformers).  
- Foundation for multimodal video-text pretraining models.  

# Implementation Notes
- Requires vector quantization to tokenize visual features.  
- Training requires large-scale video-text corpora (e.g., YouTube cooking videos).  
- Not end-to-end (video encoder frozen).  

# Critiques / Limitations
- Discretizing video into tokens loses information.  
- Computationally heavy with long sequences.  
- Limited temporal modeling resolution.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What BERT is and how masked language modeling works.  
- Why video understanding is harder than text: variable length, high dimensionality.  
- The idea of "tokenizing" video like words in text.  
- Applications of joint video-language embeddings (captioning, retrieval).  

## Postgraduate-Level Concepts
- Vector quantization of visual features.  
- Pretraining objectives for multimodal transformers.  
- Comparison of VideoBERT vs ViLBERT vs ClipBERT.  
- Implications of pretraining on large-scale noisy video-text corpora (e.g., YouTube).  

---

# My Notes
- VideoBERT was the **first serious bridge** between transformers and video-language understanding.  
- Clever trick: discretize video into tokens → unify with text.  
- Open question: Can we remove discretization and train **end-to-end**? (answered by ClipBERT).  
- Possible extension: Combine VideoBERT-style pretraining with **video diffusion models** for generative multimodal understanding.  

---
