---
layout: post
title: "Building Contrast Prediction Models with Unsupervised Learning: When Ground Truth Doesn't Exist"
description: "Learn how to approach the challenge of building ML models for contrast prediction using unsupervised learning techniques when ground truth labels are unavailable. Covers feature engineering, clustering, dimensionality reduction, autoencoders, and self-supervised methods for contrast assessment."
tags: [machine-learning, unsupervised-learning, computer-vision, contrast, image-quality, autoencoders, self-supervised]
---

Imagine you have thousands of images and need to build a system that can predict their contrast characteristics. The catch? **You have no labeled ground truth**. No "correct" contrast values to train on. No expert annotations. Just raw images and your knowledge of contrast metrics.

This scenario is more common than you might think: dataset curation pipelines, automated quality control systems, and content recommendation engines all face this challenge. Traditional supervised learning is off the table. So how do we proceed?

> **Note**: This post builds on the contrast measurement framework developed in [Understanding Contrast in Images]({{ "/2025/12/27/understanding-image-contrast.html" | relative_url }}), [Understanding Contrast in Color Images]({{ "/2025/12/27/understanding-color-contrast.html" | relative_url }}), [Measuring Contrast Between Images]({{ "/2025/12/28/measuring-contrast-between-images.html" | relative_url }}), and [Comparing Contrast Across Different Images]({{ "/2025/12/29/comparing-contrast-across-different-images.html" | relative_url }}). If you're new to contrast metrics, I recommend reading those posts first.

## Table of Contents

1. [The Ground Truth Problem](#the-ground-truth-problem)
2. [Why Unsupervised Learning for Contrast?](#why-unsupervised-learning-for-contrast)
3. [The Feature Engineering Foundation](#the-feature-engineering-foundation)
4. [Unsupervised Learning Approaches](#unsupervised-learning-approaches)
   - [4.1. Clustering-Based Methods](#41-clustering-based-methods)
   - [4.2. Dimensionality Reduction](#42-dimensionality-reduction)
   - [4.3. Autoencoder-Based Approaches](#43-autoencoder-based-approaches)
   - [4.4. Self-Supervised Learning](#44-self-supervised-learning)
5. [Pseudo-Labeling: Bridging Unsupervised and Supervised](#pseudo-labeling-bridging-unsupervised-and-supervised)
6. [Validation Without Ground Truth](#validation-without-ground-truth)
7. [Practical Architecture Design](#practical-architecture-design)
8. [Deployment Considerations](#deployment-considerations)
9. [Limitations and When to Use Supervised Instead](#limitations-and-when-to-use-supervised-instead)
10. [Conclusion](#conclusion)

## The Ground Truth Problem

What does "ground truth" even mean for contrast? Unlike object detection (where bounding boxes are objectively defined) or classification (where labels are categorical), contrast is:

**Multi-Dimensional**: Luminance contrast vs. chromatic contrast vs. local contrast vs. global contrast—which one is the "true" contrast?

**Context-Dependent**: A low-contrast portrait may be perfectly fine (soft lighting), but a low-contrast X-ray might be diagnostically useless.

**Subjectively Perceived**: Two observers might disagree on whether an image is "high contrast" or "medium contrast."

**Task-Specific**: Contrast requirements differ wildly between applications—web accessibility needs different metrics than HDR photography.

Even if we compute RMS contrast or any other metric, is that value "ground truth"? Not really—it's a **proxy measurement** that correlates with contrast but doesn't capture the full perceptual story.

This is the core challenge: **We want to predict contrast characteristics, but we don't have an objective, universal ground truth to train against.**

## Why Unsupervised Learning for Contrast?

Given the ground truth problem, unsupervised learning offers several compelling advantages:

**1. No Annotation Burden**
- Labeling thousands of images with "correct" contrast values is expensive and subjective
- Different annotators may disagree significantly
- Unsupervised methods work directly on raw images

**2. Discovery of Natural Groupings**
- Images naturally cluster by contrast characteristics (high-contrast cityscapes, low-contrast foggy scenes, etc.)
- Unsupervised learning can discover these patterns automatically

**3. Generalization Across Domains**
- Supervised models trained on specific contrast labels may not generalize
- Unsupervised representations can capture universal contrast properties

**4. Scalability**
- Can leverage massive unlabeled datasets
- Self-supervised pretraining can bootstrap better representations

**5. Robustness to Subjectivity**
- Instead of forcing a single "correct" contrast value, learn a **contrast embedding space**
- Similar images cluster together, dissimilar images separate

## The Feature Engineering Foundation

Before diving into ML models, we need to engineer features that capture contrast properties. Drawing from our previous posts, here's a comprehensive feature set:

### Low-Level Statistical Features

```python
def extract_statistical_features(image_linear):
    """Extract basic statistical contrast features."""
    features = {}
    
    # 1. Global RMS Contrast
    features['C_RMS'] = np.std(image_linear) / (np.mean(image_linear) + 1e-8)
    
    # 2. Dynamic Range Utilization
    features['U_DR'] = (np.percentile(image_linear, 99) - 
                        np.percentile(image_linear, 1)) / 255.0
    
    # 3. Histogram Entropy
    hist, _ = np.histogram(image_linear, bins=256, density=True)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist))
    
    # 4. Histogram Moments
    features['mean'] = np.mean(image_linear)
    features['std'] = np.std(image_linear)
    features['skewness'] = scipy.stats.skew(image_linear.ravel())
    features['kurtosis'] = scipy.stats.kurtosis(image_linear.ravel())
    
    return features
```

### Spatial Frequency Features

```python
def extract_frequency_features(image_linear):
    """Extract frequency-domain contrast features."""
    # FFT
    f_transform = np.fft.fft2(image_linear)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Radial power spectrum
    h, w = image_linear.shape
    center = (h // 2, w // 2)
    
    # Compute radial average
    # ... (implementation details)
    
    features = {
        'high_freq_energy': ...,
        'low_freq_energy': ...,
        'freq_ratio': ...
    }
    return features
```

### Local Contrast Features

```python
def extract_local_contrast_features(image_linear, window_size=11):
    """Extract local contrast statistics."""
    # Compute local contrast map
    local_mean = uniform_filter(image_linear, size=window_size)
    local_std = generic_filter(image_linear, np.std, size=window_size)
    local_contrast = local_std / (local_mean + 1e-8)
    
    features = {
        'mean_local_contrast': np.mean(local_contrast),
        'std_local_contrast': np.std(local_contrast),
        'max_local_contrast': np.percentile(local_contrast, 99)
    }
    return features
```

### Color Contrast Features

```python
def extract_color_features(image_rgb):
    """Extract chromatic contrast features."""
    # Convert to Lab
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    
    # Chromatic contrast
    chroma = np.sqrt(a**2 + b**2)
    
    features = {
        'luminance_contrast': np.std(L) / (np.mean(L) + 1e-8),
        'chroma_contrast': np.std(chroma) / (np.mean(chroma) + 1e-8),
        'chroma_mean': np.mean(chroma),
        'hue_diversity': ...,  # Circular variance of hue
    }
    return features
```

**Key Insight**: These engineered features are our **contrast proxies**. While not perfect, they provide a rich, multi-dimensional representation of contrast characteristics.

## Unsupervised Learning Approaches

### 4.1. Clustering-Based Methods

**Core Idea**: Group images by similar contrast characteristics, then use cluster assignments as pseudo-labels or for anomaly detection.

#### K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_images_by_contrast(image_features, n_clusters=5):
    """
    Cluster images into contrast groups.
    
    Args:
        image_features: Array of shape (N, D) where N is number of images,
                       D is feature dimension
        n_clusters: Number of contrast groups
    
    Returns:
        cluster_labels: Cluster assignment for each image
        cluster_centers: Prototype contrast profiles
    """
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(image_features)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_normalized)
    
    # Cluster centers represent "typical" contrast profiles
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return cluster_labels, cluster_centers
```

**Interpretation**:
- Cluster 0: Low-contrast, foggy images
- Cluster 1: High-contrast, high-frequency images (cityscapes)
- Cluster 2: Medium contrast, smooth gradients (portraits)
- Cluster 3: High dynamic range, bimodal histograms (sunset/sunrise)
- Cluster 4: Low dynamic range, narrow histograms (overexposed)

**Use Case**: Once clustered, you can:
- Label each cluster manually (one-time effort for 5 clusters vs. thousands of images)
- Use cluster centroids as reference contrast profiles
- Detect anomalies (images far from any cluster center)

#### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

def hierarchical_contrast_clustering(features):
    """Build a hierarchy of contrast groups."""
    # Compute linkage
    linkage = sch.linkage(features, method='ward')
    
    # Cut dendrogram at different levels
    clusters_3 = sch.fcluster(linkage, 3, criterion='maxclust')
    clusters_5 = sch.fcluster(linkage, 5, criterion='maxclust')
    
    return {'coarse': clusters_3, 'fine': clusters_5, 'linkage': linkage}
```

**Advantage**: Provides multi-resolution contrast taxonomy—coarse groups (high/medium/low) and fine-grained subgroups.

### 4.2. Dimensionality Reduction

High-dimensional feature spaces are hard to interpret. Dimensionality reduction reveals underlying structure.

#### PCA for Contrast

```python
from sklearn.decomposition import PCA

def pca_contrast_embedding(features, n_components=3):
    """
    Learn low-dimensional contrast embedding.
    
    First 2-3 principal components often capture:
    - PC1: Overall contrast magnitude (global dynamic range)
    - PC2: Spatial distribution (local vs. global)
    - PC3: Color vs. luminance contrast
    """
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(features_normalized)
    
    # Explained variance tells us information content
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    
    return embedding, pca
```

**Visualization**: Plot images in 2D PCA space—similar contrast characteristics cluster together.

#### t-SNE for Contrast Visualization

```python
from sklearn.manifold import TSNE

def tsne_contrast_visualization(features):
    """Create 2D visualization of contrast space."""
    # t-SNE preserves local structure better than PCA
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding_2d = tsne.fit_transform(features)
    
    # Visualize with scatter plot, color by computed RMS contrast
    import matplotlib.pyplot as plt
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                c=features[:, 0],  # Color by first feature (e.g., C_RMS)
                cmap='viridis')
    plt.colorbar(label='RMS Contrast')
    plt.title('t-SNE Contrast Embedding')
    plt.show()
```

**Use Case**: Interactive exploration—click on points to see representative images from different contrast regions.

### 4.3. Autoencoder-Based Approaches

Autoencoders learn compressed representations by reconstructing input images. The bottleneck embedding captures essential contrast information.

#### Variational Autoencoder (VAE)

```python
import torch
import torch.nn as nn

class ContrastVAE(nn.Module):
    """
    Variational Autoencoder for learning contrast embeddings.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(256 * 32 * 32, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 32, 32)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```

**Training Objective**:

$$ \mathcal{L} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL} $$

where:
- $\mathcal{L}_{recon}$ is reconstruction loss (MSE or perceptual loss)
- $\mathcal{L}_{KL}$ is KL divergence between latent distribution and prior
- $\beta$ controls regularization (higher = more disentangled)

**Key Insight**: The latent code $z$ encodes contrast information. Images with similar contrast have similar $z$ vectors.

**Usage**:
```python
# Train VAE
vae = ContrastVAE(latent_dim=32)
# ... training loop ...

# Extract contrast embeddings
with torch.no_grad():
    mu, _ = vae.encode(images)
    contrast_embeddings = mu.cpu().numpy()

# Use embeddings for downstream tasks
# - Clustering
# - Similarity search
# - Anomaly detection (images with unusual contrast)
```

#### Contrast-Specific Reconstruction Loss

Standard VAE uses pixel-wise MSE, which doesn't specifically emphasize contrast. We can design a custom loss:

```python
def contrast_aware_loss(original, reconstructed):
    """
    Custom loss that emphasizes contrast preservation.
    """
    # 1. Pixel-wise MSE (standard)
    mse_loss = torch.mean((original - reconstructed) ** 2)
    
    # 2. Gradient magnitude loss (preserves edges/contrast)
    grad_x_orig = original[:, :, :, 1:] - original[:, :, :, :-1]
    grad_x_recon = reconstructed[:, :, :, 1:] - reconstructed[:, :, :, :-1]
    grad_loss = torch.mean((grad_x_orig - grad_x_recon) ** 2)
    
    # 3. Histogram loss (preserves tonal distribution)
    hist_orig = compute_histogram(original)
    hist_recon = compute_histogram(reconstructed)
    hist_loss = torch.mean((hist_orig - hist_recon) ** 2)
    
    # Weighted combination
    total_loss = mse_loss + 0.5 * grad_loss + 0.3 * hist_loss
    
    return total_loss
```

### 4.4. Self-Supervised Learning

Self-supervised learning creates proxy tasks from unlabeled data to learn useful representations.

#### Contrastive Learning for Contrast

```python
import torch.nn.functional as F

class ContrastContrastiveLearning:
    """
    Learn contrast representations via contrastive learning.
    
    Key idea: Images with similar contrast should have similar embeddings,
    images with different contrast should have dissimilar embeddings.
    """
    
    def create_positive_pairs(self, image):
        """
        Create augmented versions with preserved contrast.
        """
        # Augmentations that preserve contrast:
        # - Random crops
        # - Horizontal flips
        # - Color jitter (hue/saturation, NOT brightness)
        
        aug1 = self.safe_augment(image)
        aug2 = self.safe_augment(image)
        
        return aug1, aug2
    
    def create_negative_pairs(self, image):
        """
        Create augmented versions with ALTERED contrast.
        """
        # Augmentations that change contrast:
        # - Histogram equalization
        # - Gamma correction
        # - Local contrast enhancement
        
        contrast_altered = self.alter_contrast(image)
        
        return contrast_altered
    
    def contrastive_loss(self, anchor, positive, negatives, temperature=0.07):
        """
        SimCLR-style contrastive loss.
        
        Pulls together embeddings of same-contrast images,
        pushes apart embeddings of different-contrast images.
        """
        # Compute embeddings
        z_anchor = self.encoder(anchor)
        z_positive = self.encoder(positive)
        z_negatives = [self.encoder(neg) for neg in negatives]
        
        # Similarity scores
        pos_sim = F.cosine_similarity(z_anchor, z_positive)
        neg_sims = [F.cosine_similarity(z_anchor, z_neg) for z_neg in z_negatives]
        
        # NT-Xent loss
        pos_logit = torch.exp(pos_sim / temperature)
        neg_logits = torch.sum(torch.stack([torch.exp(s / temperature) 
                                             for s in neg_sims]))
        
        loss = -torch.log(pos_logit / (pos_logit + neg_logits))
        
        return loss.mean()
```

**Training Strategy**:
1. For each image, create:
   - Positive pair: augmentations preserving contrast
   - Negative pairs: augmentations altering contrast OR different images
2. Train encoder to distinguish contrast similarity
3. Resulting embeddings capture contrast characteristics

#### Pretext Task: Contrast Classification

```python
def generate_contrast_pretext_labels(image):
    """
    Create synthetic contrast labels as pretext task.
    
    Bins:
    - Low: C_RMS < 0.3
    - Medium: 0.3 <= C_RMS < 0.6
    - High: C_RMS >= 0.6
    """
    C_RMS = compute_rms_contrast(image)
    
    if C_RMS < 0.3:
        return 0  # Low
    elif C_RMS < 0.6:
        return 1  # Medium
    else:
        return 2  # High

# Train a classifier on these bins
model = ContrastClassifier()
optimizer = torch.optim.Adam(model.parameters())

for image in dataset:
    label = generate_contrast_pretext_labels(image)
    
    output = model(image)
    loss = F.cross_entropy(output, label)
    
    loss.backward()
    optimizer.step()
```

**Key Insight**: Even though these labels are synthetic (derived from metrics, not human annotation), they provide useful training signal. The model learns features predictive of contrast.

## Pseudo-Labeling: Bridging Unsupervised and Supervised

Once you have unsupervised embeddings or clusters, you can **pseudo-label** a small subset manually, then use semi-supervised learning:

```python
def pseudo_labeling_pipeline(images, features):
    """
    1. Cluster images
    2. Select representative images from each cluster
    3. Manually label representatives
    4. Train supervised model on labeled subset
    5. Predict labels for entire dataset
    """
    # Step 1: Cluster
    clusters = KMeans(n_clusters=5).fit_predict(features)
    
    # Step 2: Select representatives (closest to cluster centers)
    representatives = []
    for i in range(5):
        cluster_members = images[clusters == i]
        cluster_features = features[clusters == i]
        
        # Find image closest to centroid
        centroid = np.mean(cluster_features, axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        rep_idx = np.argmin(distances)
        
        representatives.append(cluster_members[rep_idx])
    
    # Step 3: Manual labeling (5 images instead of thousands!)
    labels = manual_labeling_interface(representatives)
    
    # Step 4: Train supervised model
    model = train_supervised_model(representatives, labels)
    
    # Step 5: Predict for all
    predictions = model.predict(images)
    
    return predictions
```

**Advantage**: Combines unsupervised discovery with minimal supervision—best of both worlds.

## Validation Without Ground Truth

How do you evaluate a model when you don't have ground truth? Several strategies:

### 1. Consistency with Handcrafted Metrics

```python
def validate_against_metrics(model_predictions, images):
    """
    Check if model predictions correlate with computed metrics.
    """
    # Compute traditional metrics
    C_RMS_values = [compute_rms_contrast(img) for img in images]
    
    # Correlate with model predictions
    correlation = np.corrcoef(model_predictions, C_RMS_values)[0, 1]
    
    print(f"Correlation with C_RMS: {correlation:.3f}")
    
    # Should be high (>0.7) if model learned meaningful contrast
```

### 2. Human Evaluation on Sample

```python
def human_evaluation(model, test_images, n_samples=100):
    """
    Sample images and ask humans to rank by contrast.
    Compare with model rankings.
    """
    sample = random.sample(test_images, n_samples)
    
    # Model predictions
    model_scores = model.predict(sample)
    model_ranking = np.argsort(model_scores)
    
    # Human ranking (crowdsourcing or expert)
    human_ranking = collect_human_rankings(sample)
    
    # Spearman rank correlation
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(model_ranking, human_ranking)
    
    print(f"Rank correlation: {correlation:.3f} (p={p_value:.3e})")
```

### 3. Clustering Quality Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering_quality(features, cluster_labels):
    """
    Intrinsic clustering metrics (no ground truth needed).
    """
    # Silhouette score: [-1, 1], higher is better
    # Measures how similar objects are to their cluster vs. other clusters
    sil_score = silhouette_score(features, cluster_labels)
    
    # Davies-Bouldin index: lower is better
    # Ratio of within-cluster to between-cluster distances
    db_score = davies_bouldin_score(features, cluster_labels)
    
    print(f"Silhouette: {sil_score:.3f}")
    print(f"Davies-Bouldin: {db_score:.3f}")
```

### 4. Application-Specific Validation

```python
def validate_on_downstream_task(contrast_model, task_dataset):
    """
    Validate by performance on downstream task.
    
    Example: If contrast predictions help with image quality assessment,
    that's evidence the model learned something meaningful.
    """
    # Use contrast predictions as features for downstream task
    contrast_features = contrast_model.predict(task_dataset.images)
    
    # Train downstream model
    downstream_model = train_downstream(contrast_features, task_dataset.labels)
    
    # If downstream performance improves, contrast model is useful
    accuracy = downstream_model.evaluate(test_set)
    
    print(f"Downstream accuracy: {accuracy:.3f}")
```

## Practical Architecture Design

### Hybrid Architecture: Handcrafted + Learned

```python
class HybridContrastModel(nn.Module):
    """
    Combines handcrafted features with learned representations.
    """
    def __init__(self, num_handcrafted=20, latent_dim=64):
        super().__init__()
        
        # CNN for learning spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Combine learned + handcrafted
        self.fusion = nn.Sequential(
            nn.Linear(64 + num_handcrafted, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, image, handcrafted_features):
        cnn_features = self.cnn(image)
        combined = torch.cat([cnn_features, handcrafted_features], dim=1)
        embedding = self.fusion(combined)
        return embedding
```

**Advantage**: Leverages domain knowledge (handcrafted features) while allowing model to learn complementary patterns.

## Deployment Considerations

### Efficiency vs. Accuracy Trade-offs

```python
# Option 1: Full deep model (slow, accurate)
contrast_score = deep_model.predict(image)

# Option 2: Handcrafted features only (fast, less accurate)
contrast_score = compute_rms_contrast(image)

# Option 3: Hybrid (balanced)
quick_features = extract_fast_features(image)
contrast_score = lightweight_model.predict(quick_features)
```

### Online Learning and Adaptation

```python
class OnlineContrastModel:
    """
    Continuously adapt to new data distribution.
    """
    def __init__(self, base_model):
        self.model = base_model
        self.feature_buffer = []
    
    def predict_and_update(self, image):
        # Predict
        prediction = self.model.predict(image)
        
        # Extract features
        features = extract_features(image)
        self.feature_buffer.append(features)
        
        # Periodically retrain on accumulated data
        if len(self.feature_buffer) >= 1000:
            self.retrain()
        
        return prediction
    
    def retrain(self):
        # Re-cluster or update embeddings
        new_features = np.array(self.feature_buffer)
        self.model.update_clusters(new_features)
        self.feature_buffer = []
```

## Limitations and When to Use Supervised Instead

**Use Unsupervised When**:
- ✅ No labeled data available
- ✅ Task is exploratory (discovering contrast patterns)
- ✅ Objective ground truth doesn't exist
- ✅ Dataset is large and diverse

**Use Supervised When**:
- ✅ You have high-quality labeled data
- ✅ Task has clear, objective definition
- ✅ Prediction accuracy is critical
- ✅ Regulations require interpretable decisions

**Hybrid Approach**:
1. Start with unsupervised pretraining on large unlabeled dataset
2. Fine-tune with small labeled dataset
3. Best of both worlds: broad representation + task-specific adaptation

## Conclusion

Building ML models for contrast prediction without ground truth is challenging but feasible through unsupervised learning. The key principles are:

**1. Feature Engineering is Critical**
- Start with domain knowledge (RMS contrast, dynamic range, local statistics)
- Let models discover complementary patterns

**2. Unsupervised Learning as Discovery**
- Clustering reveals natural contrast groupings
- Autoencoders learn compressed contrast representations
- Self-supervised methods create training signal from data itself

**3. Validation Without Ground Truth**
- Consistency with handcrafted metrics
- Human evaluation on samples
- Downstream task performance

**4. Practical Deployment**
- Hybrid models (handcrafted + learned) balance accuracy and interpretability
- Pseudo-labeling bridges unsupervised and supervised paradigms
- Online adaptation handles distribution shift

**The Future**: As multimodal models (CLIP, DALL-E) mature, we may see contrast embeddings emerge naturally from vision-language pretraining—"high-contrast cityscape" as a prompt already encodes contrast semantics. The next frontier is leveraging these foundation models for zero-shot contrast assessment.

The journey from raw pixels to meaningful contrast predictions without ground truth labels demonstrates the power of unsupervised learning: structure exists in data, waiting to be discovered. Sometimes the best way forward isn't to collect more labels—it's to let the data speak for itself.

---

*Have you tackled unsupervised learning problems in image quality or contrast assessment? What techniques worked best for your use case? Share your experiences in the comments!*

