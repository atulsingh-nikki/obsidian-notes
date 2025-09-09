---
title: "World Models: Generative Models as Agents’ Internal Simulators (2018)"
aliases:
  - World Models
  - Ha & Schmidhuber 2018
authors:
  - David Ha
  - Jürgen Schmidhuber
year: 2018
venue: "arXiv (preprint)"
doi: "10.48550/arXiv.1803.10122"
arxiv: "https://arxiv.org/abs/1803.10122"
code: "https://worldmodels.github.io/"
citations: 5000+
dataset:
  - CarRacing-v0 (OpenAI Gym)
  - VizDoom
tags:
  - paper
  - generative-model
  - reinforcement-learning
  - simulation
  - unsupervised
fields:
  - vision
  - reinforcement-learning
  - generative-models
related:
  - "[[GANs (2014)]]"
  - "[[VAE (2013)]]"
  - "[[GameGAN (2020)]]"
predecessors:
  - "[[VAE (2013)]]"
  - "[[RNNs for sequence modeling]]"
successors:
  - "[[GameGAN (2020)]]"
  - "[[Diffusion World Simulators (2022+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**World Models** proposed that agents can learn compact **generative models of their environments** and use them as **internal simulators** for planning and policy learning. The idea was to separate perception, memory, and control into modular networks, allowing reinforcement learning agents to train inside their own dreamed environments.

# Key Idea
> Agents don’t need to interact with the real environment all the time — they can learn a **world model** (a generative simulator) and then train policies inside it, saving data and enabling imagination-based planning.

# Method
- **VAE (Vision module)**: Compresses observations (frames) into a low-dimensional latent vector.  
- **MDN-RNN (Memory module)**: Predicts sequences of latent states over time, modeling environment dynamics.  
- **Controller (C module)**: A simple linear policy that selects actions given latent + memory states.  
- **Training loop**:  
  - Train VAE + RNN world model on environment rollouts.  
  - Train controller inside the world model (dreamed environment).  
  - Deploy controller back in the real environment.  

# Results
- Learned compact world models of **CarRacing-v0** and **VizDoom**.  
- Trained policies entirely inside the dreamed world that transferred successfully to real environments.  
- Demonstrated sample efficiency gains vs direct RL.  

# Why it Mattered
- Pioneered the idea of **generative models as world simulators** for RL.  
- Influenced later work in GameGAN, Dreamer, MuZero, and diffusion-based world models.  
- Showed the power of modular separation: perception, memory, control.  

# Architectural Pattern
- VAE → RNN → Controller.  
- Latent imagination loop for policy training.  

# Connections
- Predecessor to **GameGAN (2020)** (GAN-based world simulation).  
- Conceptually related to **MuZero (2019)** (model-based RL without explicit dynamics).  
- Inspired Dreamer, PlaNet, and diffusion world simulators.  

# Implementation Notes
- Training controller inside learned model is efficient but subject to model bias.  
- Visualization: agents dreaming imagined rollouts.  
- Open-source implementation widely studied.  

# Critiques / Limitations
- World models imperfect → controllers exploit flaws in the simulator.  
- Limited to simple environments (CarRacing, Doom).  
- Not scalable to complex 3D or photorealistic worlds.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What is a latent space?  
- VAE basics.  
- Reinforcement learning with models vs without models.  

## Postgraduate-Level Concepts
- Mixture Density Networks (MDN) + RNN for dynamics.  
- Trade-offs of model-based vs model-free RL.  
- Model exploitation problem in learned simulators.  

---

# My Notes
- World Models was the **spark**: agents can dream in their own simulators.  
- Beautifully simple architecture (VAE + RNN + linear controller).  
- Open question: Can modern diffusion or transformer-based models **replace VAEs/RNNs** for scalable world models?  
- Possible extension: Diffusion-based **video world models** with long-term coherence, applied to robotics.  

---
