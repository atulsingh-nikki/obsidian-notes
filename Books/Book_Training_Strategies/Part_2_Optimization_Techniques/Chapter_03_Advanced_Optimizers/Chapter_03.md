# Chapter 3: Advanced Optimizers

## Introduction

While SGD with momentum is the gold standard for many computer vision tasks, adaptive optimizers that automatically adjust learning rates per parameter have revolutionized deep learning. These algorithms make training more robust to hyperparameter choices and often converge faster, especially for complex architectures like Transformers.

## 3.1 The Motivation for Adaptive Learning Rates

Different parameters need different learning rates:
- **Sparse features**: Parameters receiving infrequent gradients benefit from larger updates
- **Dense features**: Frequently updated parameters need smaller, more conservative steps
- **Different scales**: Activations and gradients can vary dramatically across layers

**Key insight**: Instead of a single global learning rate, adapt the rate per parameter based on gradient history.

## 3.2 AdaGrad: Adaptive Gradient Algorithm

AdaGrad (2011) was the first widely-adopted adaptive optimizer. It scales learning rates inversely proportional to the square root of accumulated squared gradients.

### Algorithm

$$g_t = \nabla_\theta \mathcal{L}(\theta_t)$$

$$G_t = G_{t-1} + g_t^2 \quad \text{(element-wise)}$$

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot g_t$$

Where:
- \(G_t\): accumulated squared gradients
- \(\epsilon\): small constant (typically 1e-8) for numerical stability
- \(\odot\): element-wise multiplication

```python
import torch

G = torch.zeros_like(parameters)
epsilon = 1e-8
lr = 0.01

for batch in data_loader:
    gradient = compute_gradient(batch)
    
    # Accumulate squared gradients
    G += gradient ** 2
    
    # Adaptive update
    parameters -= lr * gradient / (torch.sqrt(G) + epsilon)
```

### Intuition

- Parameters with **large accumulated gradients** get **smaller learning rates**
- Parameters with **small accumulated gradients** get **larger learning rates**
- Each parameter automatically gets its own adaptive learning rate

### Advantages
- ✅ No manual learning rate tuning per parameter
- ✅ Works well for sparse gradients (NLP, embeddings)
- ✅ Good for convex optimization problems

### Disadvantages
- ❌ Learning rate continually decreases (accumulation never forgets)
- ❌ Can stop learning too early in deep networks
- ❌ Rarely used for modern computer vision

**Verdict**: Historical importance, but superseded by RMSProp and Adam.

## 3.3 RMSProp: Root Mean Square Propagation

RMSProp (2012, Hinton) fixes AdaGrad's aggressive learning rate decay by using an **exponential moving average** of squared gradients instead of accumulation.

### Algorithm

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

Where \(\beta\) is typically 0.9 or 0.99.

```python
# PyTorch implementation
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99,      # decay rate (β in equations)
    eps=1e-8,
    weight_decay=0,
    momentum=0
)
```

### Key Difference from AdaGrad

**AdaGrad**: \(G_t = G_{t-1} + g_t^2\) → monotonic increase → vanishing learning rates

**RMSProp**: \(E[g^2]_t = 0.99 \cdot E[g^2]_{t-1} + 0.01 \cdot g_t^2\) → forgets old gradients → maintains learning

### Why It Works

The exponential moving average emphasizes recent gradients while gradually forgetting older ones. This allows the optimizer to:
- Adapt to changing loss landscapes during training
- Maintain non-zero learning rates throughout training
- Handle non-stationary objectives (common in deep learning)

### Typical Hyperparameters

- Learning rate: **0.001** (good starting point)
- Decay rate (β): **0.9 to 0.99**
- Epsilon: **1e-8**

**Use case**: Good general-purpose optimizer, especially for RNNs. Less common for CNNs where SGD or Adam dominate.

## 3.4 Adam: Adaptive Moment Estimation

Adam (2014, Kingma & Ba) combines the best of momentum and RMSProp. It's arguably the most popular optimizer in modern deep learning.

### Algorithm

Adam maintains two moving averages:
1. **First moment** (mean): Exponential moving average of gradients (like momentum)
2. **Second moment** (variance): Exponential moving average of squared gradients (like RMSProp)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(momentum)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(RMSProp)}$$

### Bias Correction

Early in training, \(m_t\) and \(v_t\) are biased toward zero. Adam corrects this:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

### Parameter Update

$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
# PyTorch implementation
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,           # α
    betas=(0.9, 0.999), # (β₁, β₂)
    eps=1e-8,
    weight_decay=0
)
```

### Why Adam is Popular

1. **Combines momentum and adaptive learning rates**: Best of both worlds
2. **Robust to hyperparameter choices**: Default settings work remarkably well
3. **Fast convergence**: Often reaches good performance quickly
4. **Works across domains**: CNNs, RNNs, Transformers

### Typical Hyperparameters

- Learning rate (α): **0.001** or **0.0001**
- β₁ (momentum): **0.9**
- β₂ (RMSProp decay): **0.999**
- ε: **1e-8**

## 3.5 AdamW: Adam with Decoupled Weight Decay

A critical discovery (Loshchilov & Hutter, 2017): The way Adam handles L2 regularization (weight decay) is flawed.

### The Problem with Adam + Weight Decay

Standard approach adds L2 penalty to loss:

$$\mathcal{L}_{total} = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2$$

This means weight decay is applied **before** the adaptive scaling, which interacts poorly with Adam's adaptive learning rates.

### AdamW Solution: Decouple Weight Decay

Apply weight decay **after** the adaptive update:

$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha\lambda\theta_t$$

The key: weight decay is a **separate term**, not part of the gradient.

```python
# PyTorch implementation
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # This is now proper weight decay!
)
```

### Impact

For Vision Transformers and modern architectures, **AdamW significantly outperforms Adam**:
- Better generalization (1-3% accuracy improvement)
- More stable training
- Better interaction with learning rate schedules

**Rule**: Always use **AdamW**, not Adam, especially for Transformers.

## 3.6 Choosing the Right Optimizer

### Decision Framework

```
                    Start
                      │
                      ├─ Training CNNs (ResNet, VGG)?
                      │  └─→ SGD + Momentum (0.9)
                      │     Learning rate schedule critical
                      │
                      ├─ Training Transformers (ViT, BERT)?
                      │  └─→ AdamW
                      │     lr = 1e-4, weight_decay = 0.01
                      │
                      ├─ Quick prototyping / experimentation?
                      │  └─→ Adam or AdamW
                      │     Fast convergence, forgiving
                      │
                      └─ Training RNNs / LSTMs?
                         └─→ Adam or RMSProp
                            Handle vanishing gradients well
```

### Performance Comparison (ImageNet Classification)

| Optimizer | Final Accuracy | Convergence Speed | Hyperparameter Sensitivity |
|-----------|----------------|-------------------|---------------------------|
| SGD + Momentum | **Best** (76.5%) | Slow (90 epochs) | High |
| Adam | Good (75.8%) | Fast (30 epochs) | Low |
| AdamW | **Best** (76.4%) | Fast (30 epochs) | Low |
| RMSProp | Good (75.2%) | Medium (50 epochs) | Medium |

**Note**: Results vary by architecture. For ResNets, SGD often wins. For Transformers, AdamW dominates.

## 3.7 Learning Rate for Adaptive Optimizers

Adaptive optimizers still need learning rate tuning, but the range is narrower.

### Recommended Starting Points

```python
# For Adam/AdamW on large-scale tasks (ImageNet, COCO)
lr = 1e-4  # or 0.0001

# For smaller datasets or fine-tuning
lr = 1e-5  # or 0.00001

# For very small datasets or later stages of training
lr = 1e-6  # or 0.000001
```

### Combining with Schedules

Even adaptive optimizers benefit from learning rate schedules:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Cosine annealing with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

# Warmup for first 5 epochs
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=5
)
```

## 3.8 Advanced Variants

### Lookahead Optimizer

Maintains two sets of weights: fast weights (inner loop) and slow weights (outer loop).

```python
# Using lookahead wrapper
from torch_optimizer import Lookahead

base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

**Use case**: Improved training stability, especially with aggressive learning rates.

### LAMB (Layer-wise Adaptive Moments)

Enables large-batch training (batch sizes > 32K) for Transformers:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\|\theta_t\|}{\|\hat{m}_t / \sqrt{\hat{v}_t}\|} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}}$$

**Use case**: Distributed training at massive scale (BERT, GPT pre-training).

## 3.9 Practical Recommendations

### For Computer Vision CNNs (ResNet, EfficientNet)

**First choice**: SGD + Momentum
```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200
)
```

**Alternative**: AdamW (faster convergence, slightly lower final accuracy)
```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=0.01
)
```

### For Vision Transformers (ViT, Swin)

**Standard choice**: AdamW
```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=0.05
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300, eta_min=1e-6
)
```

### For Object Detection (Faster R-CNN, YOLO)

**Typical setup**: SGD for backbone, higher LR for detection head
```python
optimizer = torch.optim.SGD([
    {'params': backbone.parameters(), 'lr': 0.01},
    {'params': detection_head.parameters(), 'lr': 0.02}
], momentum=0.9, weight_decay=1e-4)
```

## Key Takeaways

1. **Adaptive optimizers adjust learning rates per parameter**: No manual tuning needed
2. **Adam is versatile and forgiving**: Great for prototyping and most tasks
3. **AdamW is superior to Adam**: Always use decoupled weight decay
4. **SGD + momentum often wins for CNNs**: But requires careful tuning
5. **Different architectures prefer different optimizers**: Transformers → AdamW, CNNs → SGD
6. **All optimizers benefit from learning rate schedules**: Especially cosine annealing

## What's Next?

Optimization algorithms get you to a good minimum, but without regularization, your model may memorize the training set. Chapter 4 explores techniques to prevent overfitting and improve generalization.

---

**Next:** [Chapter 4: Preventing Overfitting](../../Part_3_Regularization/Chapter_04_Preventing_Overfitting/Chapter_04.md)

