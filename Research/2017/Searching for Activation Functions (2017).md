---
title: Searching for Activation Functions (2017)
aliases:
  - Swish
  - SiLU
authors:
  - Prajit Ramachandran
  - Barret Zoph
  - Quoc V. Le
year: 2017
venue: arXiv
doi: 10.48550/arXiv.1710.05941
arxiv: https://arxiv.org/abs/1710.05941
code: https://github.com/tensorflow/models/blob/master/research/swnas/swish.py
citations: 4000+
dataset:
  - ImageNet
  - WMT 2014 English-French
tags:
  - paper
  - deep-learning
  - activation-function
  - neural-architecture-search
fields:
  - vision
  - nlp
  - optimization
related:
  - "[[Neural Architecture Search]]"
  - "[[Gaussian Error Linear Units (GELU)|GELU (2016)]]"
  - "[[Rectified Linear Units (ReLU) for Deep Learning (2010–2012)|ReLU]]"
  - "[[Mish A Self Regularized Non-Monotonic Activation Function (2019)|Mish]]"
predecessors:
  - "[[Sigmoid]]"
  - "[[Rectified Linear Units (ReLU) for Deep Learning (2010–2012)|ReLU]]"
successors:
  - "[[EfficientNet Rethinking Model Scaling for Convolutional Neural Networks|EfficientNet (2019)]]"
  - "[[Attention Is All You Need (2017)|Transformer]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
This paper presents a new activation function, **Swish**, which was discovered by using an automated search technique based on reinforcement learning. The authors show that simply replacing the commonly used **ReLU** activation with Swish consistently improves the performance of deep learning models across a variety of challenging tasks and datasets, including ImageNet classification and machine translation. Its importance lies in demonstrating that even fundamental, human-designed components of neural networks can be improved through automated search. 🚀

---

# Key Idea
> Automatically searching for novel activation functions can yield a function that outperforms well-established, manually designed functions like ReLU.

---

# Method
- **Automated Search**: The core of the method is a **Neural Architecture Search (NAS)** approach. An RNN controller samples candidate activation functions from a predefined search space (e.g., combinations of unary/binary functions like `+, -, *, exp, sin, x, constant`).
- **Evaluation & Update**: Each sampled function is used to train a small "child" network on a simple task (like CIFAR-10). The resulting accuracy is used as a **reward signal** to update the controller via reinforcement learning (policy gradient), encouraging it to find better functions over time.
- **The Swish Function**: The search process discovered the function $f(x) = x \cdot \sigma(\beta x)$, where $\sigma(x)$ is the sigmoid function. The term $\beta$ can be a constant (typically 1) or a learnable parameter. The function is also known as the **Sigmoid-weighted Linear Unit (SiLU)**.

---

# Results
- **Benchmarks**: Swish was tested on large-scale benchmarks like ImageNet-1k for image classification and WMT 2014 En-Fr for machine translation.
- **Improvements vs Baselines**:
    - Replacing ReLUs with Swish improved the top-1 accuracy of **MobileNet** on ImageNet by **0.9%**.
    - It improved the accuracy of **Inception-ResNet-v2** by **0.6%**.
    - It consistently matched or outperformed ReLU across a wide range of model scales and types, demonstrating its robustness.
- **Notable Numbers**: The improvements, while seemingly small (~0.5-1%), are significant on challenging benchmarks like ImageNet where progress has often plateaued.

---

# Why it Mattered
- **Challenged ReLU's Dominance**: Before this paper, ReLU was the undisputed default activation function. Swish was one of the first functions to show consistent, widespread improvement over ReLU, prompting researchers to reconsider this choice.
- **Validated Automated Discovery**: It was a powerful demonstration that automated search techniques could discover novel, non-obvious, and highly effective components for neural networks.
- **Popularized Non-Monotonic Functions**: Unlike ReLU, Swish is **non-monotonic** (it dips slightly for small negative values). This property, once thought to be undesirable, was shown to be beneficial, helping to create a smoother optimization landscape.

---

# Architectural Pattern
- **Self-Gating**: Swish can be interpreted as a form of self-gating, where the function modulates the input ($x$) by a "gate" ($\sigma(\beta x)$) that is itself a function of the input. This self-gating mechanism is a powerful pattern that has appeared in other architectures like LSTMs and attention mechanisms.
- **Simplicity and Efficiency**: Despite its discovery through a complex search, the final function is simple and elegant, making it easy to implement and adopt in various models. It became a core component of later high-performance models like **EfficientNet**.

---

# Connections
- **Contemporaries**: The **GELU** (Gaussian Error Linear Unit) function was proposed around the same time and shares many of Swish's beneficial properties (smoothness, non-monotonicity). Both became popular choices in Transformer-based models.
- **Influence**: Swish's success directly influenced the design of many state-of-the-art models, most notably the **EfficientNet** family of models, which made Swish a standard component in high-performance computer vision architectures.

---

# Implementation Notes
- **Learnable Parameter ($\beta$)**: While $\beta$ can be a learnable parameter, the authors found that setting $\beta=1$ worked nearly as well in most cases. Modern frameworks like PyTorch implement it with a fixed $\beta=1$ in `torch.nn.SiLU`.
- **Computational Cost**: Swish is slightly more computationally expensive than ReLU due to the sigmoid calculation. However, this cost is often negligible compared to other operations in a deep network and is justified by the performance gain.
- **Numerical Stability**: When using automatic mixed precision (AMP), the implementation should be handled carefully to maintain stability.

---

# Critiques / Limitations
- **Expensive Discovery**: The search process itself is extremely computationally expensive, requiring thousands of GPU hours, making it inaccessible to most researchers.
- **Marginal Gains**: While consistent, the performance improvements can be incremental. In some applications, the extra computational cost may not be worth the small accuracy boost.
- **Superseded by later work?**: While still a very strong baseline, other functions like **Mish** have since claimed to outperform Swish in some contexts, though Swish/SiLU remains a very popular and effective choice.

---

# Repro / Resources
- **Paper link**: [https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)
- **Official code repo**: Minimal implementation available in the [TensorFlow models repo](https://github.com/tensorflow/models/blob/master/research/swnas/swish.py).
- **Framework Implementations**:
    - **PyTorch**: `torch.nn.SiLU`
    - **TensorFlow**: `tf.keras.activations.swish` or `tf.nn.swish`

---

# Educational Connections

## Undergraduate-Level Concepts
- **Calculus**: The derivative of Swish, $f'(x) = f(x) + \sigma(\beta x)(1 - \beta f(x))$, is essential for backpropagation. Its smooth, non-zero nature helps prevent the "dying ReLU" problem.
- **Probability & Statistics**: The sigmoid function $\sigma(x)$ is fundamental to logistic regression and represents a probability.
- **Optimization Basics**: Swish's smooth, non-convex shape can create a more favorable optimization landscape for gradient-based optimizers compared to the piecewise linear and non-differentiable-at-zero ReLU.

## Postgraduate-Level Concepts
- **Machine Learning Theory**: The properties of Swish—being **unbounded above** (avoids saturation) and **bounded below** (provides some regularization)—contribute to its strong performance. Its non-monotonicity is a key topic of analysis.
- **Neural Network Design**: A perfect case study in how a micro-level architectural choice (the activation function) can have a macro-level impact on model performance.
- **Research Methodology**: This paper is a landmark example of using **reinforcement learning for automated scientific discovery**, a paradigm that extends far beyond just finding activation functions.

---

# My Notes
- **How this connects to my projects**: Could I get a quick and easy performance boost in my current ResNet model by simply swapping all ReLU layers with `torch.nn.SiLU`? It's worth a quick experiment.
- **Open questions**: How sensitive is the optimal value of the learnable parameter $\beta$ to the network depth or width? Does it learn different values for different layers?
- **Possible extensions**: Could the same search methodology be adapted to find better normalization layers or pooling strategies? The search space would be more complex, but the potential payoff could be large.