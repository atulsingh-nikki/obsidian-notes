# Book: Deep Learning Architectures: Building Blocks, Tweaks, and Impact

## Introduction
- **The Philosophy of Architecture Design**
  - Understanding Inductive Biases: Why structure matters.
  - The tradeoff between flexibility (low bias) and sample efficiency (high bias).
  - Overview of the "Building Block" approach: Layers, Normalizations, Activations, and Connections.

---

## Part I: Spatial Architectures (The Evolution of Vision)

### Chapter 1: The Convolutional Foundation (From Filters to Hierarchies)
- **Core Building Block:** The Convolution Operation (Parameter sharing, Translation Invariance) & Pooling.
- **Key Architecture:** **AlexNet** & **VGG**.
- **Tweaks & Impact:**
  - *Tweak:* **ReLU (Rectified Linear Unit)** vs. Tanh/Sigmoid.
    - *Impact:* Solved vanishing gradients, enabled deeper training.
  - *Tweak:* **Dropout**.
    - *Impact:* Regularization, preventing overfitting in large parameter spaces.
  - *Tweak:* **Local Response Normalization (LRN)** (and why it faded).

### Chapter 2: Going Deeper (The Optimization Breakthroughs)
- **Core Building Block:** **Residual Connections (Skip Connections)**.
- **Key Architecture:** **ResNet**.
- **Tweaks & Impact:**
  - *Tweak:* **Identity Mappings**.
    - *Impact:* Allowed gradients to flow through hundreds of layers; solved degradation problem.
  - *Tweak:* **Bottleneck Blocks** (1x1 convolutions).
    - *Impact:* Reduced computational cost while increasing depth.
  - *Tweak:* **Batch Normalization**.
    - *Impact:* Stabilized training, allowed higher learning rates.

### Chapter 3: Multi-Scale and Width (Thinking Laterally)
- **Core Building Block:** **Inception Modules** (Split-Transform-Merge).
- **Key Architecture:** **Inception (GoogLeNet)** & **ResNeXt**.
- **Tweaks & Impact:**
  - *Tweak:* **Factorized Convolutions** (NxN -> 1xN + Nx1).
    - *Impact:* Efficiency without losing receptive field.
  - *Tweak:* **Grouped Convolutions** (Cardinality).
    - *Impact:* Increased capacity without proportional compute cost (ResNeXt).

### Chapter 4: Efficiency by Design (Mobile and Edge)
- **Core Building Block:** **Depthwise Separable Convolutions**.
- **Key Architecture:** **MobileNet** & **EfficientNet**.
- **Tweaks & Impact:**
  - *Tweak:* **Inverted Residuals & Linear Bottlenecks**.
    - *Impact:* Preserved information in low-dimensional manifolds (MobileNetV2).
  - *Tweak:* **Compound Scaling** (Width, Depth, Resolution).
    - *Impact:* Systematic way to scale up models for optimal performance (EfficientNet).
  - *Tweak:* **Squeeze-and-Excitation (SE) Blocks**.
    - *Impact:* Adaptive channel-wise feature recalibration.

### Chapter 5: The Spatial Attention Shift (Transformers in Vision)
- **Core Building Block:** **Patch Embeddings** & **Self-Attention**.
- **Key Architecture:** **ViT (Vision Transformer)** & **Swin Transformer**.
- **Tweaks & Impact:**
  - *Tweak:* **Shifted Windows (Swin)**.
    - *Impact:* Reintroduced locality and hierarchical processing to Transformers.
  - *Tweak:* **Positional Encodings** (Absolute vs. Relative).
    - *Impact:* How models understand "where" things are without convolution.

---

## Part II: Sequential Architectures (Time, Text, and Series)

### Chapter 6: Recurrent Foundations (Memory in Loops)
- **Core Building Block:** **Recurrence** (Hidden States).
- **Key Architecture:** **Vanilla RNNs**.
- **Tweaks & Impact:**
  - *Tweak:* **Backpropagation Through Time (BPTT)**.
    - *Impact:* Enabling learning over temporal sequences.
  - *Tweak:* **Gradient Clipping**.
    - *Impact:* Handling exploding gradients in long sequences.

### Chapter 7: Solving Short-Term Memory (Gating Mechanisms)
- **Core Building Block:** **Gating** (Sigmoid valves).
- **Key Architecture:** **LSTM (Long Short-Term Memory)** & **GRU**.
- **Tweaks & Impact:**
  - *Tweak:* **Forget Gate**.
    - *Impact:* Allowed the model to reset state, crucial for continuous learning.
  - *Tweak:* **Bidirectionality**.
    - *Impact:* Seeing the future context (crucial for NLP tasks like NER).

### Chapter 8: The Alignment Era (Seq2Seq and Attention)
- **Core Building Block:** **Encoder-Decoder** & **Attention Mechanism**.
- **Key Architecture:** **Seq2Seq with Bahdanau/Luong Attention**.
- **Tweaks & Impact:**
  - *Tweak:* **Context Vector vs. Attention Weights**.
    - *Impact:* Solved the "bottleneck" problem of fixed-length context vectors.
  - *Tweak:* **Beam Search Decoding**.
    - *Impact:* Better output generation than greedy search.

### Chapter 9: The Transformer Revolution (Attention is All You Need)
- **Core Building Block:** **Multi-Head Self-Attention (MHSA)** & **Feed-Forward Networks (FFN)**.
- **Key Architecture:** **Transformer (BERT, GPT)**.
- **Tweaks & Impact:**
  - *Tweak:* **Scaled Dot-Product Attention**.
    - *Impact:* Stabilized gradients in large dimension spaces.
  - *Tweak:* **Layer Normalization Placement** (Pre-LN vs. Post-LN).
    - *Impact:* Pre-LN allowed for stable training of very deep transformers (GPT-2/3).
  - *Tweak:* **Masked Attention** (Causal masking).
    - *Impact:* Enabled generative (autoregressive) modeling.

### Chapter 10: Beyond Quadratic Complexity (Modern Sequences)
- **Core Building Block:** **State Space Models (SSMs)** & **Linear Attention**.
- **Key Architecture:** **Mamba**, **RWKV**, **Longformer**.
- **Tweaks & Impact:**
  - *Tweak:* **Discretization (ZOH/Bilinear)**.
    - *Impact:* Mapping continuous ODEs to discrete recurrent steps (Mamba).
  - *Tweak:* **Parallel Associative Scan**.
    - *Impact:* Training RNN-like models in parallel like Transformers.
  - *Tweak:* **Sliding Window Attention**.
    - *Impact:* Linear complexity for long documents.

---

## Part III: Building Blocks Deep Dive (The Atomic Elements)

### Chapter 11: Normalization Layers
- **Blocks:** Batch Norm, Layer Norm, Instance Norm, Group Norm.
- **Comparison:** When to use which? (Batch size dependency, Sequence vs. Spatial).

### Chapter 12: Activation Functions
- **Blocks:** Sigmoid -> Tanh -> ReLU -> Leaky ReLU -> GELU -> Swish.
- **Impact:** The quest for smooth, non-monotonic, unbounded gradients.

### Chapter 13: Optimizers as Architecture Partners
- **Blocks:** SGD, Momentum, RMSProp, Adam, AdamW.
- **Impact:** How weight decay implementation (AdamW) changed transformer training stability.


