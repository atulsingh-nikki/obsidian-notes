# Chapter 1: Understanding Model Training

## Introduction

At its core, training a neural network for computer vision involves showing the model examples (images and their corresponding labels or targets) and adjusting its internal parameters to minimize the difference between its predictions and the ground truth. This chapter establishes the foundational concepts that underpin all training strategies.

## 1.1 The Training Loop

The training loop is the heartbeat of deep learning. Understanding its components is essential for mastering more advanced techniques.

### Basic Structure

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward Pass
        images, labels = batch
        predictions = model(images)
        
        # 2. Compute Loss
        loss = loss_function(predictions, labels)
        
        # 3. Backward Pass
        loss.backward()
        
        # 4. Update Parameters
        optimizer.step()
        optimizer.zero_grad()
```

### The Four Essential Steps

1. **Forward Pass**: Data flows through the network layers, producing predictions
2. **Loss Computation**: Quantifies how far predictions are from ground truth
3. **Backward Pass**: Computes gradients of loss with respect to all parameters
4. **Parameter Update**: Adjusts parameters in the direction that reduces loss

## 1.2 Loss Functions for Vision Tasks

Different computer vision tasks require different loss functions. The choice of loss function fundamentally shapes what the model learns.

### Classification: Cross-Entropy Loss

For image classification, cross-entropy loss is the standard choice:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Where:
- \(C\) is the number of classes
- \(y_i\) is the ground truth label (one-hot encoded)
- \(\hat{y}_i\) is the predicted probability for class \(i\)

**Why it works**: Cross-entropy measures the difference between two probability distributions. It heavily penalizes confident wrong predictions, pushing the model toward correct classifications.

### Regression: Mean Squared Error

For tasks like depth estimation or pose prediction:

$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

**Alternative**: L1 loss (Mean Absolute Error) is more robust to outliers:

$$\mathcal{L}_{L1} = \frac{1}{N}\sum_{i=1}^{N} \mid y_i - \hat{y}_i \mid$$

### Segmentation: Pixel-wise Cross-Entropy

Semantic segmentation treats each pixel as a classification problem:

$$\mathcal{L}_{seg} = -\frac{1}{H \times W}\sum_{h=1}^{H}\sum_{w=1}^{W}\sum_{c=1}^{C} y_{h,w,c} \log(\hat{y}_{h,w,c})$$

**Advanced**: Dice Loss addresses class imbalance:

$$\mathcal{L}_{Dice} = 1 - \frac{2 \mid X \cap Y \mid}{\mid X \mid + \mid Y \mid}$$

### Object Detection: Multi-Component Loss

Object detection combines multiple objectives:

$$\mathcal{L}_{det} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{box}\mathcal{L}_{box} + \lambda_{obj}\mathcal{L}_{obj}$$

Where:
- \(\mathcal{L}_{cls}\): Classification loss for object categories
- \(\mathcal{L}_{box}\): Bounding box regression loss (often IoU-based)
- \(\mathcal{L}_{obj}\): Objectness confidence loss
- \(\lambda\) terms: Loss balancing weights

## 1.3 Gradient Computation and Backpropagation

Backpropagation is the algorithm that makes training neural networks tractable. It efficiently computes gradients using the chain rule.

### The Chain Rule in Practice

For a simple network \(y = f_3(f_2(f_1(x)))\), the gradient is:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial \theta_1}$$

Modern frameworks like PyTorch and TensorFlow automate this computation through **automatic differentiation**.

### Computational Graph

```
Input → Conv1 → ReLU → Conv2 → ReLU → FC → Softmax → Loss
  ↓       ↓       ↓       ↓       ↓      ↓      ↓        ↓
  θ1     ---     θ2     ---     θ3   ---   ---    ∂L/∂θ
```

During the backward pass, gradients flow from right to left, accumulating at each parameter.

### Gradient Flow Challenges

**Vanishing Gradients**: In deep networks, repeated multiplication of small gradients can make early layers learn very slowly.

**Solution**: 
- Residual connections (ResNet)
- Batch Normalization
- Proper weight initialization (He, Xavier)

**Exploding Gradients**: Gradients can grow exponentially, causing training instability.

**Solution**:
- Gradient clipping: \(\theta \leftarrow \theta - \alpha \cdot \min(1, \frac{\tau}{\|\nabla\|}) \nabla\)
- Batch Normalization
- Layer Normalization

## 1.4 Training vs. Validation vs. Test Sets

Proper data splitting is critical for honest evaluation and preventing overfitting.

### The Three-Way Split

```
Total Dataset (100%)
├── Training Set (70-80%)    → Used to update model parameters
├── Validation Set (10-15%)  → Used for hyperparameter tuning & early stopping
└── Test Set (10-15%)        → Used only for final evaluation
```

### Why Three Sets?

**Training Set**: The model "sees" and learns from this data. Gradients are computed and parameters updated based on training data.

**Validation Set**: Used to:
- Monitor overfitting during training
- Tune hyperparameters (learning rate, batch size, regularization strength)
- Decide when to stop training (early stopping)

**Test Set**: The "held-out" set that provides an unbiased estimate of final model performance. Should **never** influence any training decisions.

### Common Mistakes

❌ **Tuning on test set**: Using test accuracy to choose hyperparameters leads to overfitting to the test set

❌ **No validation set**: Using only train/test makes it impossible to tune hyperparameters without "peeking" at test performance

✅ **Proper workflow**: Train → Validate → Tune → Repeat → Final test evaluation once

### Cross-Validation for Small Datasets

When data is limited, k-fold cross-validation provides more reliable estimates:

```
Fold 1: [Test] [Train Train Train Train]
Fold 2: [Train] [Test] [Train Train Train]
Fold 3: [Train Train] [Test] [Train Train]
Fold 4: [Train Train Train] [Test] [Train]
Fold 5: [Train Train Train Train] [Test]
```

Each fold serves as test set once, and results are averaged.

## 1.5 Monitoring Training Progress

### Key Metrics to Track

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease, but may plateau or increase (overfitting signal)
3. **Learning Rate**: Important to track when using schedules
4. **Gradient Norms**: Detect vanishing/exploding gradients
5. **Task-Specific Metrics**: Accuracy, mAP, IoU, etc.

### Interpreting Training Curves

```
Loss
  │
  │  Train Loss ↘
  │              ↘
  │                ↘
  │                  ───  (plateau)
  │  Val Loss   ↘
  │              ↘
  │                ↗  (overfitting!)
  └────────────────────── Epochs
```

**Healthy Training**: Both curves decrease, validation slightly above training

**Underfitting**: Both curves high and not decreasing → Model too simple or learning rate too low

**Overfitting**: Training loss decreases but validation loss increases → Model memorizing training data

## 1.6 Batch Size Considerations

Batch size significantly impacts training dynamics and final performance.

### Small Batches (8-32)
- ✅ More frequent updates → faster early training
- ✅ Better generalization (noise acts as regularization)
- ❌ Noisy gradients → unstable training
- ❌ Slower wall-clock time (less parallelization)

### Large Batches (128-512+)
- ✅ Stable gradients → smooth training
- ✅ Better GPU utilization → faster wall-clock time
- ❌ May generalize worse (sharp minima hypothesis)
- ❌ Requires careful learning rate scaling

### Linear Scaling Rule

When increasing batch size by factor \(k\), scale learning rate by \(k\):

$$\text{lr}_{new} = k \times \text{lr}_{base}$$

**Intuition**: Larger batches provide more stable gradient estimates, allowing larger steps.

## Key Takeaways

1. **Training is iterative**: Forward pass → Loss → Backward pass → Update
2. **Loss function defines the objective**: Choose carefully based on your task
3. **Backpropagation computes gradients**: Automatic differentiation makes this seamless
4. **Data splitting is critical**: Train/Val/Test prevents overfitting and enables honest evaluation
5. **Monitor everything**: Training curves reveal problems early
6. **Batch size matters**: It affects both training dynamics and final performance

## What's Next?

Now that we understand the fundamentals, we'll explore optimization techniques that go beyond basic gradient descent. Chapter 2 introduces momentum, adaptive learning rates, and modern optimizers that power state-of-the-art training.

---

**Next:** [Chapter 2: Gradient Descent and Variants](../../Part_2_Optimization_Techniques/Chapter_02_Gradient_Descent_and_Variants/Chapter_02.md)

