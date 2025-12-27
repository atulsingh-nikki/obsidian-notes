# Chapter 2: Gradient Descent and Variants

## Introduction

Gradient descent is the workhorse of deep learning optimization. While the basic idea—follow the negative gradient to minimize loss—is simple, the practical variants make the difference between models that train in days versus weeks, and between models that converge versus those that oscillate indefinitely.

## 2.1 Batch Gradient Descent

The classical formulation computes the gradient over the **entire training set** before making a single update.

### Algorithm

```python
for epoch in range(num_epochs):
    # Compute gradient over ALL training data
    predictions = model(all_training_data)
    loss = loss_function(predictions, all_labels)
    gradients = compute_gradients(loss)
    
    # Single update per epoch
    parameters -= learning_rate * gradients
```

### Mathematical Formulation

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

Where:
- \(\theta_t\): parameters at iteration \(t\)
- \(\alpha\): learning rate
- \(\nabla_\theta \mathcal{L}\): gradient of loss with respect to parameters

### Advantages
- ✅ Exact gradient → deterministic convergence
- ✅ Stable updates → smooth training curves
- ✅ Guaranteed convergence to local minimum (convex case)

### Disadvantages
- ❌ Requires entire dataset in memory
- ❌ Very slow for large datasets (one update per epoch!)
- ❌ Cannot escape poor local minima
- ❌ Impractical for modern deep learning

**Verdict**: Rarely used in practice except for very small datasets.

## 2.2 Stochastic Gradient Descent (SGD)

SGD makes updates based on **one training example at a time**, introducing randomness but enabling much faster iteration.

### Algorithm

```python
for epoch in range(num_epochs):
    # Shuffle data every epoch
    for single_example, single_label in shuffled_dataset:
        prediction = model(single_example)
        loss = loss_function(prediction, single_label)
        gradients = compute_gradients(loss)
        
        # Update after every single example
        parameters -= learning_rate * gradients
```

### Mathematical Formulation

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t; x_i, y_i)$$

Where the gradient is computed on a single example \((x_i, y_i)\).

### Advantages
- ✅ Fast iteration → many updates per epoch
- ✅ Can escape shallow local minima (noise helps)
- ✅ Online learning possible
- ✅ Memory efficient

### Disadvantages
- ❌ Noisy gradients → unstable training
- ❌ Difficult to parallelize
- ❌ May oscillate around minimum without converging

**Verdict**: Fast but too noisy for modern deep learning. Mini-batch SGD strikes the right balance.

## 2.3 Mini-Batch Gradient Descent

The **goldilocks solution**: compute gradients over small batches (typically 32-256 examples), balancing update frequency and gradient quality.

### Algorithm

```python
for epoch in range(num_epochs):
    for batch in data_loader(batch_size=32):
        images, labels = batch
        predictions = model(images)
        loss = loss_function(predictions, labels)
        gradients = compute_gradients(loss)
        
        # Update after each mini-batch
        parameters -= learning_rate * gradients
```

### Mathematical Formulation

$$\theta_{t+1} = \theta_t - \alpha \frac{1}{m}\sum_{i=1}^{m} \nabla_\theta \mathcal{L}(\theta_t; x_i, y_i)$$

Where \(m\) is the mini-batch size.

### Why Mini-Batches Work

1. **Gradient Estimation**: The mini-batch gradient is an unbiased estimator of the true gradient
2. **Variance Reduction**: Averaging over \(m\) examples reduces gradient noise by factor of \(\sqrt{m}\)
3. **Hardware Efficiency**: Modern GPUs are optimized for batch operations
4. **Regularization**: Some noise helps generalization (implicit regularization)

### Choosing Batch Size

```
Batch Size    Use Case                     GPU Memory    Training Speed
────────────────────────────────────────────────────────────────────────
8-16          Very limited memory          Low           Slow
32-64         Standard for research        Medium        Good
128-256       Production, large models     High          Fast
512-1024      Distributed training         Very High     Very Fast
```

**Rule of thumb**: Use the largest batch size that fits in GPU memory, but don't exceed 512 without careful learning rate tuning.

## 2.4 Learning Rate: The Most Important Hyperparameter

The learning rate \(\alpha\) controls step size. Too small → slow training. Too large → divergence.

### Finding a Good Learning Rate

**Learning Rate Range Test**:
1. Start with very small LR (e.g., 1e-7)
2. Train for a few hundred iterations, increasing LR exponentially
3. Plot loss vs. learning rate
4. Choose LR where loss decreases fastest

```python
# Pseudocode for LR range test
lrs = np.logspace(-7, 0, 100)  # 1e-7 to 1
losses = []

for lr in lrs:
    optimizer.lr = lr
    loss = train_step()
    losses.append(loss)

plt.plot(lrs, losses)
plt.xscale('log')
# Choose LR at steepest descent, before loss explodes
```

### Typical Learning Rates by Optimizer

- **SGD**: 0.01 to 0.1 (with momentum: 0.01 to 0.3)
- **Adam**: 0.0001 to 0.001 (1e-4 to 1e-3)
- **RMSProp**: 0.001 to 0.01

## 2.5 Learning Rate Schedules

Static learning rates are rarely optimal. Schedules adapt LR during training for better convergence.

### Step Decay

Reduce LR by factor every \(N\) epochs:

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/N \rfloor}$$

```python
# Reduce LR by 0.1 every 30 epochs
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
```

**Use case**: Classic schedule for ResNets on ImageNet.

### Exponential Decay

Smooth exponential reduction:

$$\alpha_t = \alpha_0 \cdot e^{-\lambda t}$$

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)
```

**Use case**: When you want gradual, continuous decay.

### Cosine Annealing

Follows a cosine curve from initial LR to minimum:

$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)
```

**Use case**: Very popular for modern architectures (Vision Transformers, modern CNNs). Often gives 1-2% better accuracy than step decay.

### One Cycle Policy

Increase LR in first half of training, then decrease:

```
LR
 │    ╱╲
 │   ╱  ╲
 │  ╱    ╲
 │ ╱      ╲
 │╱        ╲
 └──────────── Iterations
```

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, total_steps=num_steps
)
```

**Use case**: Fast convergence when training time is limited. Popularized by fastai.

### Warmup

Start with very small LR and linearly increase to target:

```python
def linear_warmup(current_step, warmup_steps, base_lr):
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)
    return base_lr
```

**Why warmup helps**:
- Prevents early instability with large gradients
- Especially important for large batch training
- Critical for training Transformers

**Typical**: 5-10% of total training as warmup, then switch to cosine decay.

## 2.6 SGD with Momentum

Plain SGD can be slow and oscillate in ravines (dimensions with high curvature). Momentum smooths updates by accumulating gradient history.

### Algorithm

```python
velocity = 0
momentum = 0.9

for batch in data_loader:
    gradient = compute_gradient(batch)
    
    # Accumulate velocity
    velocity = momentum * velocity + gradient
    
    # Update with velocity, not raw gradient
    parameters -= learning_rate * velocity
```

### Mathematical Formulation

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - \alpha v_t$$

Where \(\beta\) is the momentum coefficient (typically 0.9).

### Why Momentum Works

Think of a ball rolling downhill:
- It accumulates speed in consistent directions
- It dampens oscillations in dimensions with high curvature
- It can roll through small bumps (escape shallow local minima)

### Momentum as Exponential Moving Average

Expanding the recursion:

$$v_t = \nabla_t + \beta\nabla_{t-1} + \beta^2\nabla_{t-2} + \cdots$$

Recent gradients have more weight, older gradients decay exponentially.

### Nesterov Momentum (Lookahead)

Computes gradient at the *lookahead* position:

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t - \alpha \beta v_{t-1})$$

$$\theta_{t+1} = \theta_t - \alpha v_t$$

**Intuition**: Look ahead where momentum will take you, then compute gradient there. Often gives slightly better convergence.

```python
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9, 
    nesterov=True
)
```

## 2.7 Practical Guidelines

### When to Use SGD

✅ **Use SGD (with momentum) when**:
- Training CNNs for computer vision (ResNet, VGG, etc.)
- You have time to tune learning rate carefully
- You want the best final generalization (often beats Adam)
- Training for many epochs (100+)

❌ **Avoid SGD when**:
- Quick prototyping is needed
- Limited time for hyperparameter tuning
- Training Transformers or RNNs (Adam is better)

### Hyperparameter Starting Points

**For ImageNet-scale training (ResNet)**:
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,           # Reduced by 10x at epochs 30, 60, 90
    momentum=0.9,
    weight_decay=1e-4
)
batch_size = 256
num_epochs = 120
```

**For smaller datasets**:
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,          # Lower LR for smaller datasets
    momentum=0.9,
    weight_decay=5e-4  # Stronger regularization
)
batch_size = 32
num_epochs = 200
```

## Key Takeaways

1. **Mini-batch SGD is the standard**: Balances update frequency and gradient quality
2. **Learning rate is critical**: Use LR range test to find good starting point
3. **Schedules improve final accuracy**: Cosine annealing and warmup are very effective
4. **Momentum accelerates convergence**: Almost always use momentum (β = 0.9)
5. **Batch size affects everything**: Scale LR when changing batch size

## What's Next?

SGD with momentum is powerful but requires careful tuning. Chapter 3 introduces adaptive optimizers like Adam that automatically adjust learning rates per parameter, making training more forgiving and often faster.

---

**Next:** [Chapter 3: Advanced Optimizers](../Chapter_03_Advanced_Optimizers/Chapter_03.md)

