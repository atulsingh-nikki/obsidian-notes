# Chapter 5: Data Augmentation

## Introduction

The most effective way to prevent overfitting is to have more training data. But collecting and labeling data is expensive. Data augmentation generates new training examples by applying transformations to existing data, effectively multiplying dataset size without additional labeling costs.

## 5.1 Why Data Augmentation Works

### Implicit Regularization

Each augmented image is slightly different, forcing the model to learn invariant features rather than memorizing specific examples.

### Effective Dataset Size

With 50,000 training images and 10 random augmentations per epoch:
- **Effective dataset size**: 500,000 unique training examples
- The model never sees the exact same image twice

### Invariance and Robustness

Augmentation teaches the model that certain transformations (rotation, scaling, color shifts) shouldn't change predictions.

## 5.2 Geometric Transformations

### Random Crop and Resize

The most fundamental augmentation for computer vision.

```python
import torchvision.transforms as T

transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
    T.ToTensor(),
])
```

**Effect**: Forces model to recognize objects at different scales and positions.

### Horizontal Flip

```python
transform = T.RandomHorizontalFlip(p=0.5)
```

**Use case**: Natural images (cats, cars, landscapes)  
**Avoid for**: Text, digits (flipped "6" becomes "9"), medical images with anatomical orientation

### Rotation

```python
transform = T.RandomRotation(degrees=30)
```

**Typical range**: ±15° to ±45°  
**Use case**: Objects that can appear at various orientations (aerial imagery, flowers)

### Perspective and Affine Transforms

```python
transform = T.RandomPerspective(distortion_scale=0.2, p=0.5)
```

**Effect**: Simulates viewing angle changes, improving 3D understanding.

## 5.3 Color Space Augmentations

### Brightness, Contrast, Saturation

```python
transform = T.ColorJitter(
    brightness=0.2,  # ±20% brightness
    contrast=0.2,    # ±20% contrast
    saturation=0.2,  # ±20% saturation
    hue=0.1          # ±0.1 hue shift (max 0.5)
)
```

**Why it works**: Makes model robust to lighting conditions and camera differences.

### Grayscale Conversion

```python
transform = T.RandomGrayscale(p=0.1)
```

**Effect**: Forces model to not rely solely on color (useful when texture is more important).

### Advanced: AutoAugment

AutoAugment (Cubuk et al., 2019) uses reinforcement learning to find optimal augmentation policies.

```python
from torchvision.transforms.autoaugment import AutoAugmentPolicy

transform = T.AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
```

**Performance**: Can improve accuracy by 1-2% on ImageNet, especially for smaller models.

## 5.4 Mixup and CutMix

These techniques create new training examples by mixing pairs of images.

### Mixup

Linearly interpolate between two images and their labels:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$

$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Where \(\lambda \sim \text{Beta}(\alpha, \alpha)\), typically \(\alpha = 0.2\).

```python
def mixup(images, labels, alpha=0.2):
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    
    # Random permutation
    indices = torch.randperm(batch_size)
    
    # Mix images and labels
    mixed_images = lam * images + (1 - lam) * images[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    return mixed_images, mixed_labels
```

**Effect**:
- Encourages linear behavior between training examples
- Improves model calibration
- Typical improvement: 0.5-1.5% accuracy

### CutMix

Cuts a patch from one image and pastes it into another:

```python
def cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)
    
    # Generate random bounding box
    H, W = images.shape[2:]
    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    return images, mixed_labels
```

**Advantages over Mixup**:
- Maintains natural image statistics (no blending)
- Encourages localization (model must find objects in patches)
- Often gives slightly better results than Mixup

## 5.5 Domain-Specific Augmentations

### For Medical Imaging

```python
transform = T.Compose([
    T.RandomRotation(180),  # Scans can be at any orientation
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # Elastic deformation for tissue-like transforms
    ElasticTransform(alpha=50, sigma=5),
])
```

### For Satellite/Aerial Imagery

```python
transform = T.Compose([
    T.RandomRotation(180),  # No "up" direction
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
])
```

### For Object Detection

Must also transform bounding boxes:

```python
import albumentations as A

transform = A.Compose([
    A.RandomSizedBBoxSafeCrop(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```

## 5.6 Practical Guidelines

### Standard Augmentation Pipeline

For **ImageNet-style classification**:

```python
# Training
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation (no augmentation, just resize and center crop)
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Augmentation Strength vs. Dataset Size

```
Dataset Size       Augmentation Strength
─────────────────────────────────────────
< 1,000 samples    Very strong (mixup, cutmix, autoaugment)
1K - 10K           Strong (all geometric + color jitter)
10K - 100K         Moderate (standard augmentation)
> 100K             Light (crop + flip may suffice)
```

### Test-Time Augmentation (TTA)

Apply augmentations during inference and average predictions:

```python
def predict_with_tta(model, image, num_augmentations=5):
    predictions = []
    
    for _ in range(num_augmentations):
        augmented = augment(image)
        pred = model(augmented)
        predictions.append(pred)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)
```

**Effect**: Typically 0.5-1% accuracy improvement, at cost of slower inference.

## 5.7 Avoiding Common Pitfalls

### ❌ Over-Augmentation

Too aggressive augmentation can hurt performance:
- Rotating text images upside-down
- Extreme color shifts that change object identity
- Crops that completely remove the object

**Rule**: Augmented images should still be recognizable to humans.

### ❌ Validation Set Augmentation

```python
# WRONG: Augmenting validation set
val_loader = DataLoader(dataset, transform=train_transform)

# CORRECT: Clean validation set
val_loader = DataLoader(dataset, transform=val_transform)
```

**Why**: Validation set should represent real-world data (no augmentation).

### ❌ Inconsistent Normalization

```python
# WRONG: Different normalization for train/val
train_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
val_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# CORRECT: Same normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
train_transform = T.Normalize(mean=MEAN, std=STD)
val_transform = T.Normalize(mean=MEAN, std=STD)
```

## 5.8 Advanced Techniques

### RandAugment

Simplifies AutoAugment with just two hyperparameters:

```python
from torchvision.transforms import RandAugment

transform = RandAugment(num_ops=2, magnitude=9)
```

**Parameters**:
- `num_ops`: Number of augmentations to apply (typically 1-3)
- `magnitude`: Strength of augmentations (0-30, typically 9-15)

### AugMax

Adversarially selects hardest augmentations during training, forcing model to learn more robust features.

## Key Takeaways

1. **Augmentation is free performance**: Essential for small datasets, beneficial for all
2. **Geometric transforms are universal**: Crop, flip, rotation work for almost everything
3. **Color augmentation adds robustness**: Especially important for real-world deployment
4. **Mixup and CutMix are powerful**: Can add 1-2% accuracy with minimal cost
5. **Match augmentation to dataset size**: Stronger augmentation for smaller datasets
6. **Never augment validation/test sets**: They should reflect real-world data
7. **Test-Time Augmentation (TTA) boosts accuracy**: At cost of slower inference

## What's Next?

We've covered optimization and regularization—training a model from scratch. But what if you could start with a model that already understands visual features? Chapter 6 explores transfer learning and foundation models.

---

**Next:** [Chapter 6: Foundation Models](../../Part_4_Transfer_Learning/Chapter_06_Foundation_Models/Chapter_06.md)

