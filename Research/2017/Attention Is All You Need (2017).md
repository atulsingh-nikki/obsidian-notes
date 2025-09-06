---
title: "Attention Is All You Need (2017)"
aliases: 
  - Transformer
  - Self-Attention Networks
authors:
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  - Lukasz Kaiser
  - Illia Polosukhin
year: 2017
venue: "NeurIPS"
doi: "10.48550/arXiv.1706.03762"
arxiv: "https://arxiv.org/abs/1706.03762"
code: "https://github.com/tensorflow/tensor2tensor"
citations: 100,000+
dataset:
  - WMT 2014 English-to-German
  - WMT 2014 English-to-French
tags:
  - paper
  - transformer
  - attention
  - deep-learning
fields:
  - nlp
  - vision
  - multimodal
related:
  - "[[Seq2Seq with Attention (Bahdanau et al., 2014)]]"
  - "[[BERT (2018)]]"
predecessors:
  - "[[RNN-based Seq2Seq Models]]"
  - "[[Bahdanau Attention (2014)]]"
successors:
  - "[[BERT (2018)]]"
  - "[[GPT (2018–2020)]]"
  - "[[Vision Transformer (ViT, 2020)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---

# Summary
This paper introduced the **Transformer architecture**, a model relying solely on **attention mechanisms**, dispensing with recurrence and convolutions entirely. It redefined sequence modeling, first for NLP tasks like machine translation, and later as the foundation for modern large language models and vision transformers.

# Key Idea
> Replace recurrence with **self-attention**, allowing global context modeling with parallelizable computation.

# Method
- **Core mechanism**: Scaled Dot-Product Attention  
  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
 $$  
- **Multi-Head Attention**: Parallel attention layers capture different types of relationships.  
- **Positional Encoding**: Injects sequence order into input embeddings (since no recurrence).  
- **Encoder–Decoder**:  
  - Encoder: stack of self-attention + feed-forward layers.  
  - Decoder: masked self-attention + encoder–decoder attention + feed-forward.  
- **Training**: Adam optimizer, label smoothing, residual connections, and layer normalization.  

# Results
- Achieved state-of-the-art BLEU scores on **WMT 2014 En→De (28.4)** and **En→Fr (41.8)**.  
- Trained significantly faster than RNN/CNN-based seq2seq models.  
- Demonstrated superior scalability with larger models and datasets.  

# Why it Mattered
- Eliminated sequential bottlenecks of RNNs, enabling **parallel training**.  
- Self-attention scales gracefully and captures **long-range dependencies**.  
- Became the backbone of modern AI: BERT, GPT, ViT, multimodal transformers.  

# Architectural Pattern
- **Attention-only** encoder–decoder.  
- Residual + layer norm after each sublayer.  
- Fully parallelizable sequence modeling.  

# Connections
- **Contemporaries**: Convolutional seq2seq (Gehring et al., 2017).  
- **Influence**: Every major NLP/Vision model post-2018 (BERT, GPT, ViT, Stable Diffusion).  

# Implementation Notes
- Needs large-scale data to shine.  
- Positional encodings critical for maintaining sequence order.  
- Multi-head attention benefits from careful dimensional splits.  

# Critiques / Limitations
- Quadratic cost in sequence length (attention computation).  
- Early models required careful regularization (label smoothing, dropout).  
- Generalization sometimes weaker in low-data regimes vs RNNs.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1706.03762)  
- [Tensor2Tensor implementation](https://github.com/tensorflow/tensor2tensor)  
- [Annotated Transformer tutorial](http://nlp.seas.harvard.edu/2018/04/03/attention.html)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Matrix multiplications in attention.  
- **Probability & Statistics**: Softmax normalization.  
- **Optimization Basics**: Training with Adam, cross-entropy loss.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Self-attention, residual connections.  
- **Machine Learning Theory**: Inductive bias vs recurrence/convolutions.  
- **Research Methodology**: Benchmarking on MT tasks.  
- **Advanced Optimization**: Large-batch training, scaling laws.  

---

# My Notes
- The most important milestone for **generative AI**.  
- Relevant to my interests in **video transformers** and **diffusion models**.  
- Open question: Can **linear attention or state-space models** solve the quadratic bottleneck?  
- Possible extension: Apply **hierarchical temporal transformers** for long-form video editing.  
