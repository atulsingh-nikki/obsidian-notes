# Chapter 4: Preventing Overfitting

## Introduction

A model that achieves 99% accuracy on training data but only 70% on test data has failed at its primary objective: generalization. Overfitting—when a model memorizes training data rather than learning underlying patterns—is one of the most fundamental challenges in machine learning. This chapter explores techniques to combat it.

## 4.1 Understanding Overfitting

### The Generalization Gap

$$\text{Generalization Gap} = \text{Training Error} - \text{Test Error}$$

A large gap indicates overfitting. Your model has learned the noise, not the signal.

### Why Overfitting Happens

1. **Model capacity exceeds data complexity**: Too many parameters relative to training examples
2. **Training too long**: Model starts memorizing instead of generalizing
3. **Insufficient regularization**: No constraints on parameter values
4. **Data quality issues**: Noisy labels, outliers, or insufficient diversity

## 4.2 L2 Regularization (Weight Decay)

Add a penalty term that discourages large parameter values.

### Mathematical Formulation

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{\lambda}{2}\sum_i \theta_i^2$$

Where \(\lambda\) controls regularization strength.

### Implementation

```python
# Method 1: Built into optimizer
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    weight_decay=1e-4  # This is λ
)

# Method 2: Manual in loss
l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
total_loss = task_loss + 0.0001 * l2_penalty
```

### Why It Works

- Smaller weights → smoother functions → better generalization
- Prevents any single feature from dominating
- Equivalent to MAP estimation with Gaussian prior

### Typical Values

- **Strong regularization**: λ = 1e-3 (small datasets)
- **Standard**: λ = 1e-4 (ImageNet scale)
- **Light**: λ = 1e-5 (very large datasets)

## 4.3 L1 Regularization (Lasso)

Penalizes the absolute value of parameters, promoting sparsity.

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda\sum_i \mid\theta_i\mid$$

### Effect: Sparse Models

L1 drives many parameters to exactly zero, effectively performing feature selection.

```python
l1_penalty = sum(p.abs().sum() for p in model.parameters())
total_loss = task_loss + 0.0001 * l1_penalty
```

**Use case**: When you want to identify which features are most important. Less common in deep learning than L2.

## 4.4 Dropout: Randomly Disabling Neurons

Dropout (Hinton et al., 2012) randomly sets neuron activations to zero during training.

### Algorithm

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # probability of dropping
    
    def forward(self, x):
        if self.training:
            # Create binary mask
            mask = (torch.rand_like(x) > self.p).float()
            # Scale by 1/(1-p) to maintain expected value
            return x * mask / (1 - self.p)
        else:
            # No dropout during inference
            return x
```

### Why Dropout Works

- **Ensemble effect**: Training \(2^n\) different subnetworks (where \(n\) is number of neurons)
- **Prevents co-adaptation**: Forces neurons to learn robust features independently
- **Reduces overfitting**: Acts as strong regularizer

### Typical Dropout Rates

- **Fully-connected layers**: p = 0.5 (drop 50%)
- **Convolutional layers**: p = 0.1 to 0.3 (CNNs less sensitive)
- **Recurrent connections**: p = 0.2 to 0.5 (careful with RNNs)

### Modern Usage

```python
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),      # After activation
    nn.Linear(256, 10)
)
```

**Note**: Modern architectures (ResNets, Transformers) use dropout sparingly. Batch normalization often makes aggressive dropout unnecessary.

## 4.5 Spatial Dropout for CNNs

Standard dropout drops individual values. **Spatial dropout** drops entire feature maps.

```python
class SpatialDropout2D(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        if self.training:
            # Drop entire channels
            mask = (torch.rand(x.size(0), x.size(1), 1, 1, device=x.device) > self.p).float()
            return x * mask / (1 - self.p)
        return x
```

**Why better for CNNs**: Convolutional features are spatially correlated. Dropping individual pixels is less effective than dropping entire feature maps.

## 4.6 Batch Normalization

Batch Norm (Ioffe & Szegedy, 2015) normalizes layer inputs, which has a regularizing effect.

### Algorithm

For a mini-batch \(\mathcal{B} = \{x_1, \ldots, x_m\}\):

$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} x_i$$

$$\sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

Where \(\gamma\) and \(\beta\) are learnable parameters.

### Implementation

```python
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3),
    nn.BatchNorm2d(64),  # After conv, before activation
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.BatchNorm2d(128),
    nn.ReLU(),
)
```

### Why It Regularizes

- Adds noise from batch statistics (similar to dropout)
- Reduces internal covariate shift
- Allows higher learning rates → faster convergence

### Batch Norm vs. Dropout

**Batch Norm**:
- ✅ Accelerates training
- ✅ Reduces need for dropout
- ❌ Introduces batch-dependent noise

**Dropout**:
- ✅ Strong regularizer
- ❌ Can slow convergence
- ❌ Less effective with batch norm

**Modern practice**: Use Batch Norm, add dropout only if still overfitting.

## 4.7 Early Stopping

Stop training when validation performance stops improving.

### Algorithm

```python
best_val_loss = float('inf')
patience = 10  # Number of epochs to wait
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model
model.load_state_dict(torch.load('best_checkpoint.pth'))
```

### Why It Works

Training loss always decreases, but validation loss may start increasing (overfitting signal). Early stopping prevents training too long.

### Choosing Patience

- **Small patience (5-10 epochs)**: Fast training, risk of stopping too early
- **Large patience (20-30 epochs)**: More thorough training, slower
- **Adaptive**: Increase patience as training progresses

## 4.8 Data-Based Regularization

### Train/Val/Test Split

Proper data splitting is the first line of defense:

```python
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train_val, test = train_test_split(data, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.176, random_state=42) # 0.176 * 0.85 ≈ 0.15
```

### Cross-Validation for Small Datasets

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    train_data = data[train_idx]
    val_data = data[val_idx]
    
    model = create_fresh_model()
    train_model(model, train_data, val_data)
    evaluate_model(model, val_data)
```

### Label Smoothing

Soften hard labels to prevent overconfidence:

$$y_{smooth} = (1-\epsilon)y + \frac{\epsilon}{K}$$

Where \(K\) is number of classes and \(\epsilon \approx 0.1\).

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        # Apply smoothing
        smooth_target = one_hot * (1 - self.epsilon) + self.epsilon / n_classes
        # Compute loss
        log_probs = F.log_softmax(pred, dim=-1)
        return -(smooth_target * log_probs).sum(dim=-1).mean()
```

**Effect**: Prevents model from being overconfident, improves calibration.

## 4.9 Practical Guidelines

### Progressive Regularization Strategy

```python
# Phase 1: Minimal regularization (find model capacity)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0)

# Phase 2: If overfitting, add weight decay
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Phase 3: If still overfitting, add dropout
model.add_dropout(p=0.3)

# Phase 4: If still overfitting, use stronger augmentation
train_transform = StrongAugmentation()

# Phase 5: Early stopping as last resort
use_early_stopping(patience=15)
```

### Regularization Checklist

1. ✅ **Always**: Proper train/val/test split
2. ✅ **Standard**: Weight decay (L2) in optimizer
3. ✅ **For CNNs**: Batch normalization
4. ✅ **If overfitting**: Dropout (0.3-0.5 for FC layers)
5. ✅ **For small datasets**: Strong data augmentation + early stopping
6. ✅ **Final safeguard**: Early stopping

## Key Takeaways

1. **Overfitting is inevitable without regularization**: Modern networks have millions of parameters
2. **L2 regularization (weight decay) is universal**: Use in every optimizer
3. **Batch Norm is both accelerator and regularizer**: Essential for modern architectures
4. **Dropout is powerful but use sparingly**: Modern architectures need less dropout
5. **Early stopping is a last line of defense**: Monitor validation loss religiously
6. **Combine multiple techniques**: Regularization works best in combination

## What's Next?

Regularization prevents overfitting by constraining the model. But you can also make your model more robust by increasing effective dataset size through data augmentation—the focus of Chapter 5.

---

**Next:** [Chapter 5: Data Augmentation](../Chapter_05_Data_Augmentation/Chapter_05.md)

