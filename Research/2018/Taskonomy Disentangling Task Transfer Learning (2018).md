---
title: "Taskonomy: Disentangling Task Transfer Learning (2018)"
aliases: 
  - Taskonomy
  - Task Transfer Learning
authors:
  - Amir R. Zamir
  - Alexander Sax
  - William Shen
  - Leonidas J. Guibas
  - Jitendra Malik
  - Silvio Savarese
year: 2018
venue: "CVPR"
doi: "10.1109/CVPR.2018.00681"
arxiv: "https://arxiv.org/abs/1804.08328"
code: "http://taskonomy.stanford.edu/"
citations: 4000+
dataset:
  - Taskonomy dataset (4.5M images, indoor environments)
tags:
  - paper
  - transfer-learning
  - multitask
  - representation-learning
fields:
  - vision
  - machine-learning
  - transfer-learning
related:
  - "[[Multi-Task Learning]]"
  - "[[Self-Supervised Learning]]"
predecessors:
  - "[[Pretraining on ImageNet]]"
successors:
  - "[[Meta-Learning]]"
  - "[[Visual Decathlon Benchmark]]"
impact: ⭐⭐⭐⭐⭐
status: "read"
---


# Summary
Taskonomy is a large-scale study of **transfer learning between visual tasks**. It introduced a dataset of millions of images annotated for multiple tasks, and used it to map out a **task transfer graph** showing which tasks are best sources for transfer to others.

# Key Idea
> Build a taxonomy of tasks by empirically measuring transferability, revealing structure in the space of visual tasks.

# Method
- **Dataset**: Taskonomy dataset with 4.5M images from 600 buildings, annotated for 26 tasks (e.g., depth, normals, edges, segmentation, keypoints).  
- **Approach**:  
  - Train task-specific networks.  
  - Transfer representations across tasks (freezing encoder, retraining decoder).  
  - Measure transfer performance quantitatively.  
- **Analysis**: Build a **directed graph of tasks** where edges represent useful transferability.  
- Identified **clusters of tasks** (e.g., geometry, semantics, 2D).  

# Results
- Showed tasks like **surface normals** and **depth estimation** are good sources for transfer.  
- Semantic tasks benefit more from geometric pretraining than vice versa.  
- Produced the first large-scale **task transfer graph** in computer vision.  

# Why it Mattered
- Moved beyond single-dataset pretraining (ImageNet) toward a **principled view of transfer learning**.  
- Provided insights into **task relationships** that guide multi-task and self-supervised learning.  
- Inspired new approaches for efficient training pipelines in vision.  

# Architectural Pattern
- Encoder–decoder CNNs trained per task.  
- Frozen encoder + retrained decoder to measure transferability.  
- Transfer graph construction via systematic experiments.  

# Connections
- **Contemporaries**: Multi-task CNNs, self-supervised pretraining (2017–18).  
- **Influence**: Meta-learning, task transfer in robotics, multi-task benchmarks.  

# Implementation Notes
- Requires huge compute (training many task pairs).  
- Dataset indoor-only → limited diversity.  
- Transfer graph sensitive to architecture and training setup.  

# Critiques / Limitations
- Indoor bias, limited to synthetic indoor data.  
- Transferability conclusions may not generalize to outdoor/natural images.  
- Analysis static; later works explored **dynamic transfer and continual learning**.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1804.08328)  
- [Project site](http://taskonomy.stanford.edu/)  
- [Code + dataset info](http://taskonomy.stanford.edu/)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Encoder–decoder mappings.  
- **Probability & Statistics**: Transfer performance evaluation.  
- **Optimization Basics**: Freezing and fine-tuning.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Encoder–decoder reuse.  
- **Computer Vision**: Task relationships (geometry vs semantics).  
- **Research Methodology**: Benchmark design for transferability.  
- **Advanced Optimization**: Multi-task trade-offs.  

---

# My Notes
- Useful blueprint for **deciding which auxiliary tasks help my team’s video models**.  
- Could inspire a “**Taskonomy for Video**” → mapping transferability across temporal vision tasks.  
- Open question: Can modern **foundation models** make task transfer graphs obsolete, or do relationships still matter?  
- Possible extension: Combine Taskonomy with **diffusion features** for richer transfer learning maps.  
