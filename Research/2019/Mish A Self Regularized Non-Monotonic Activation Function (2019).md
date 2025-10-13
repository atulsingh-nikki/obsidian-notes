---
title: "Mish: A Self Regularized Non-Monotonic Activation Function (2019)"
aliases:
  - Mish
authors:
  - Diganta Misra
year: 2019
venue: British Machine Vision Conference (BMVC) 2020
doi: 10.48550/arXiv.1908.08681
arxiv: https://arxiv.org/abs/1908.08681
code: https://github.com/digantamisra98/Mish
citations: 1000+
dataset:
  - ImageNet
  - CIFAR-100
  - MS-COCO
tags:
  - paper
  - deep-learning
  - activation-function
  - computer-vision
fields:
  - vision
  - optimization
related:
  - "[[Gaussian Error Linear Units (GELU)|GELU (2016)]]"
  - "[[Rectified Linear Units (ReLU) for Deep Learning (2010–2012)|ReLU]]"
  - "[[Searching for Activation Functions (2017)|Swish]]"
predecessors:
  - "[[Rectified Linear Units (ReLU) for Deep Learning (2010–2012)|ReLU]]"
  - "[[Searching for Activation Functions (2017)|Swish]]"
successors:
  - "[[YOLOv4]]"
impact: ⭐⭐⭐☆☆
status: read
---

# Summary
This paper introduces **Mish**, a novel self-regularized, non-monotonic activation function. Mish aims to improve upon the performance of ReLU and Swish by providing an even smoother activation landscape. The function is defined as $x \cdot \tanh(\text{softplus}(x))$. The author provides extensive empirical evidence showing that Mish can outperform Swish and ReLU on a variety of computer vision tasks, particularly in very deep networks, by helping to stabilize network gradients and improve optimization. 🧠

---

# Key Idea
> A smoother, non-monotonic activation function that is continuously differentiable can improve deep learning model performance by creating a better-conditioned optimization landscape.

---

# Method
- **Mathematical Formulation**: Mish is a smooth, non-monotonic function that is unbounded above and bounded below. It is defined by the following equation:
$$
\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x))
$$
where $\text{softplus}(x) = \ln(1 + e^x)$ and $\tanh(x)$ is the hyperbolic tangent function.
- **Properties**:
    - **Smoothness**: Mish is continuously differentiable ($C^\infty$), unlike ReLU which has a point of non-differentiability at zero. This smoothness helps with gradient-based optimization.
    - **Non-Monotonicity**: Like Swish and GELU, it has a small negative dip, which is believed to help with regularization.
    - **Self-Gating**: It follows the self-gating pattern of $x \cdot g(x)$, similar to Swish.


---

# Results
- **Benchmarks**: The paper presents a comprehensive set of experiments on image classification (CIFAR-100, ImageNet-1k) and object detection (MS-COCO).
- **Improvements vs Baselines**:
    - On ImageNet, a SqueezeExcite-ResNet18 with Mish achieved a **1% higher top-1 accuracy** compared to the same model with Swish, and **~2.8%** higher than with ReLU.
    - It showed consistent, albeit sometimes small, improvements across a wide range of architectures, including ResNets, DenseNets, and CSPNets.
- **Notable Numbers**: The function gained significant attention when it was used as a key component in the **YOLOv4** object detection model, which set a new state-of-the-art at the time of its release.

---

# Why it Mattered
- **Pushed the Envelope on Smoothness**: While Swish and GELU were smoother than ReLU, Mish took it a step further. Its analysis focused heavily on the properties of its derivative and the resulting smooth loss landscape, encouraging deeper investigation into the theoretical properties of activation functions.
- **Showcased "Next-Generation" Activations**: It demonstrated that there were still performance gains to be had by carefully designing activation functions, even after the success of Swish.
- **Practical Impact in YOLOv4**: Its adoption in a major, high-impact model like YOLOv4 gave it significant credibility and led to its wider recognition and use, particularly within the object detection community.

---

# Architectural Pattern
- **Self-Gating**: Mish continues the successful pattern of self-gating, where the input is modulated by a function of itself. This pattern has proven to be a robust way to introduce effective non-linearities in deep networks.
- **Drop-in Replacement**: Like its predecessors, it's designed to be a simple drop-in replacement for ReLU or other activations, requiring no architectural changes to the host network.

---

# Connections
- **Contemporaries**: Mish is a direct successor and competitor to **Swish**. Much of the paper is dedicated to comparing these two functions.
- **Influence**: It inspired further research into hand-crafting even more complex and potentially better-performing activation functions. Its use in YOLOv4 heavily influenced subsequent object detection models.

---

# Implementation Notes
- **Computational Cost**: Mish is significantly more computationally expensive than ReLU and even Swish, due to the combination of `log`, `exp`, and `tanh` operations.
- **Custom Kernels**: To mitigate the performance overhead, the author and community developed custom, highly optimized CUDA kernels to compute Mish and its gradient efficiently on GPUs.
- **Framework Support**: While not always a built-in layer in the earliest versions of frameworks, it is now available in major libraries, often in community-supported or "add-on" packages. PyTorch includes it in `torch.nn.Mish`.

---

# Critiques / Limitations
- **High Computational Overhead**: This is the biggest drawback. The performance gain from Mish is often marginal and may not justify the significant increase in training and inference time, especially on hardware without optimized kernels.
- **Marginal Gains**: The improvements, while consistent in the paper's experiments, can be small and architecture-dependent. In many cases, Swish or GELU provide a better balance of performance and efficiency.
- **Limited Adoption**: Despite its strong empirical results and use in YOLOv4, Mish has not been adopted as widely as Swish or GELU in mainstream SOTA models, especially in NLP.

---

# Repro / Resources
- **Paper link**: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
- **Official code repo**: [https://github.com/digantamisra98/Mish](https://github.com/digantamisra98/Mish)
- **Framework Implementations**:
    - **PyTorch**: `torch.nn.Mish`
    - **TensorFlow**: Available in `tensorflow_addons.activations.mish`

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: Mish is an excellent case study in the importance of a function's derivatives. Its continuous first and second derivatives are key to its proposed benefits for optimization.
- **Numerical Methods**: The `softplus` function, $ln(1+e^x)$, can be numerically unstable for large $x$. Understanding stable implementations (the log-sum-exp trick) is important.
- **Optimization Basics**: The paper's argument revolves around creating a smoother loss landscape, which directly relates to how optimizers like SGD and Adam navigate the parameter space to find a minimum.

## Postgraduate-Level Concepts
- **Advanced Optimization**: The concept of a "well-conditioned" loss landscape, which Mish aims to create, is a central topic in advanced optimization. A smoother landscape can prevent optimizer stalling and lead to better generalization.
- **Machine Learning Theory**: The self-regularizing property of Mish, where it pushes mean activations closer to zero, is a subtle form of regularization that can improve model robustness.
- **Neural Network Design**: Demonstrates the micro-optimization of network components and the trade-offs between mathematical elegance, empirical performance, and computational efficiency.

---

# My Notes
- **How this connects to my projects**: If I am working on a project where squeezing out the last fraction of a percent of accuracy is critical (e.g., a competition or a state-of-the-art model), and I have the computational budget, trying Mish as a replacement for Swish/ReLU could be a worthwhile experiment.
- **Open questions**: Is the increased computational cost of Mish ever truly worth it in large-scale models, or do other factors (architecture, data augmentation) have a much larger impact?
- **Possible extensions**: Could a faster, approximate version of Mish be developed that retains most of its beneficial properties but is cheaper to compute, similar to the GELU approximations?