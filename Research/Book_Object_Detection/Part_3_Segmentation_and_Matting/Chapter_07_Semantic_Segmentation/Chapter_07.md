# Chapter 7: Pixel-Perfect Understanding: Semantic Segmentation

## 7.1 Introduction: From Boxes to Masks

In the previous part of this book, we explored the evolution of object detection—the task of drawing a bounding box around every object in an image. While this is a powerful form of understanding, the bounding box is a coarse approximation. It cannot capture the true, detailed shape of an object, and it cannot distinguish between different object classes within the box (e.g., a person on a horse is just one box).

To achieve a more detailed, pixel-level understanding of a scene, we must turn to **semantic segmentation**. The goal of semantic segmentation is to assign a class label to *every single pixel* in an image. The output is not a set of boxes, but a dense, pixel-wise "mask" that carves the image into its constituent parts: this group of pixels is a "person," this group is a "horse," and this group is "sky."

This is a fundamentally different and more challenging task. The key question that unlocked the power of deep learning for this problem was: how can we take a network that is brilliant at image-level classification and transform it into a network that can make a dense, pixel-level prediction? The answer, and the paper that started the revolution, was the Fully Convolutional Network.

## 7.2 Fully Convolutional Networks (FCN): End-to-End Dense Prediction

The 2015 paper "Fully Convolutional Networks for Semantic Segmentation" by Long, Shelhamer, and Darrell is the foundational work of modern, deep-learning-based segmentation [1]. It provided a simple, powerful, and elegant solution to the problem of adapting classification networks for dense prediction.

### 7.2.1 The Core Insight: Convolutions All the Way Down

The core insight of the FCN paper is that the fully-connected layers at the end of a standard classification network (like VGG) are the main barrier to dense prediction. These layers discard all spatial information, flattening the rich, 2D feature map into a 1D vector to make a single prediction. The authors' brilliant idea was to **replace these fully-connected layers with convolutional layers**, transforming the entire network into a "fully convolutional" architecture that could operate on images of any size and produce a spatial output.

#### A Deeper Look: How to "Convolutionalize" a Fully-Connected Layer
A fully-connected (FC) layer can be viewed as a convolution with a kernel that has the exact same spatial dimensions as its input feature map. For example, consider the first FC layer in VGG, which takes a 7x7x512 feature map as input and produces a 4096-dimensional vector. This layer has a weight matrix of size `[4096, 25088]` (where 25088 = 7*7*512). This exact same transformation can be achieved by a convolutional layer with 4096 kernels, each of size 7x7x512. The subsequent FC layers can then be mimicked by 1x1 convolutions. This "convolutionalization" produces a network that is mathematically identical to the original classifier but is now spatially aware and can produce a coarse heatmap as its output.

![Figure 7.1: The FCN-8s architecture. The diagram shows how a classification network (VGG) is transformed into a fully convolutional network. Skip connections combine coarse, deep semantic features with fine, shallow spatial features at multiple scales to produce a precise, dense segmentation. Image Source: Long, Shelhamer, & Darrell, 2015 [1].](../../images/ch07_fig01_fcn_architecture.png)

### 7.2.2 The FCN Pipeline: Upsampling and Skip Connections

This fully-convolutional approach produces a coarse, low-resolution heatmap (e.g., at 1/32nd of the input image resolution). To get a dense, pixel-wise prediction, two more key ideas were needed.

#### A Deeper Look: In-Network Upsampling
To restore the output to the original image resolution, the network must learn to upsample its coarse feature map. The FCN paper introduced the idea of **in-network upsampling** using a layer that is often called a **deconvolution** or, more accurately, a **transposed convolution**. For a detailed mathematical and visual guide to this operation, the technical report "A guide to convolution arithmetic for deep learning" by Dumoulin and Visin is the canonical reference [2]. This is not a true mathematical inverse of the convolution operation. Rather, it is a learnable layer that reverses the geometric input-output mapping of a standard convolution. While a standard convolution maps a patch of inputs to a single output (downsampling), a transposed convolution maps a single input to a larger patch of outputs (upsampling). Crucially, the weights of this upsampling kernel are learned during training, allowing the network to learn how to intelligently reconstruct the high-resolution details.

#### Fusing Features with Skip Connections
The predictions from the final, deep layers of the network are semantically rich (they know *what* is in the image) but spatially coarse (they don't know exactly *where*). This is because the repeated pooling operations have discarded precise location information. The feature maps from the earlier, shallower layers, in contrast, are spatially detailed but semantically weak. To get the best of both worlds, FCN introduced **skip connections**.

These connections provide a shortcut for the fine-grained, high-resolution information from the early layers to bypass the deep, downsampled part of the network and be combined directly with the upsampled output. This fusion (done via element-wise addition) allows the network to use the precise local information from the early layers to refine the coarse, semantic predictions from the deep layers.

The authors created three variants of their architecture based on this idea:
*   **FCN-32s:** This is the simplest version. The output from the final layer is upsampled by a factor of 32 in a single step to get back to the full image resolution. It is semantically strong but spatially very coarse.
*   **FCN-16s:** The output of the final layer is upsampled by a factor of 2 and then fused (added) with the feature map from the preceding pooling layer (`pool4` in VGG). This combined feature map is then upsampled by a factor of 16. This adds more spatial detail.
*   **FCN-8s:** This process is repeated one more time. The result from the FCN-16s path is upsampled by 2 and fused with the feature map from an even earlier layer (`pool3`). This is then upsampled by a factor of 8.

Each successive fusion step adds more fine-grained detail from shallower layers, resulting in progressively sharper and more accurate segmentation masks.

### 7.2.3 Training FCNs

The final piece of the puzzle is how the network is trained. The FCN is trained **end-to-end** with a dense, per-pixel loss function. The final output of the network is a heatmap with C channels, where C is the number of classes. This heatmap is then compared to the ground-truth segmentation mask (which is a simple image with integer class labels for each pixel). A standard **per-pixel cross-entropy loss** is calculated over all the pixels in the image, and the gradients are backpropagated through the entire network. This allows the network to learn how to make correct, pixel-wise predictions for the entire image at once.

### 7.2.4 Key Contributions & Impact

The FCN paper was a monumental achievement that created the entire field of modern semantic segmentation.
*   It provided a simple and elegant **end-to-end, pixels-to-pixels** framework for training a deep network for dense prediction.
*   It introduced the core architectural components that would become standard in almost all future segmentation models: **in-network upsampling (transposed convolution)** and **skip connections** for multi-scale feature fusion.
*   It achieved state-of-the-art results on all major segmentation benchmarks, proving the power of this new approach and setting the stage for years of subsequent research.

## 7.3 U-Net: The Encoder-Decoder for Precise Segmentation

Happening in parallel to the development of FCN, another hugely influential paper was published in 2015 that proposed a similar but distinct fully convolutional architecture. The paper, "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger, Fischer, and Brox, was designed to solve a specific and challenging problem: the segmentation of biomedical images [2].

### 7.3.1 Motivation: The Challenges of Biomedical Imaging

Biomedical image segmentation presents a unique set of challenges. First, the task often requires extremely precise localization. For example, accurately segmenting the boundary of a single cell in a microscopy image is crucial. Second, in the medical field, large, annotated datasets are often a luxury. The authors needed an architecture that could be trained effectively on a very small number of training images and still produce highly precise results.

### 7.3.2 The Core Insight: A Symmetric, U-Shaped Architecture

The solution proposed by the authors was an elegant, fully symmetric, U-shaped architecture that became known as the **U-Net**. Like FCN, it consisted of a downsampling path to capture context and an upsampling path to produce a full-resolution output. The key innovation, however, was in how these two paths were connected.

![Figure 7.2: The U-Net architecture. The symmetric, U-shaped design consists of a contracting path (encoder) on the left and an expansive path (decoder) on the right. Powerful skip connections concatenate high-resolution feature maps from the encoder to the decoder at every level, enabling precise localization. Image Source: Ronneberger, Fischer, & Brox, 2015 [3].](../../images/ch07_fig02_unet_architecture.png)

### 7.3.3 A Deeper Look: The U-Net Architecture

The U-Net consists of two main parts:

1.  **The Contracting Path (Encoder):** This is a standard, downsampling convolutional network. It consists of repeating blocks of two 3x3 convolutions followed by a 2x2 max pooling operation. At each downsampling step, the spatial dimensions are halved, and the number of feature channels is doubled. The purpose of this path is to capture the "what" of the image—the high-level, semantic context.

2.  **The Expansive Path (Decoder):** This path is a symmetric, upsampling network. At each step, it upsamples the feature map using a 2x2 transposed convolution, which halves the number of feature channels. It then **concatenates** the upsampled feature map with the corresponding feature map from the contracting path and applies two 3x3 convolutions.

#### The Power of U-Net's Skip Connections
The **skip connections** are the most important part of the U-Net architecture and the key to its success. Unlike the FCN, which used simple addition to combine coarse features from only a few layers, the U-Net's skip connections are much more powerful. By **concatenating** the full, high-resolution feature maps from the encoder path at every level of the decoder, the U-Net provides a massive amount of high-resolution, local information directly to the upsampling path. This allows the decoder to reconstruct a very precise, high-resolution output, using the rich contextual information from the downsampling path to inform the fine-grained localization.

#### Cropping and Alignment in the Original U-Net
The original U-Net used "valid" 3x3 convolutions (no padding) in the encoder, which shrink the spatial dimensions at each layer. As a result, when features are upsampled in the decoder, the corresponding encoder feature maps are slightly larger. To concatenate them channel-wise, the encoder features are **cropped** to match the decoder feature map size before concatenation. Near image borders, U-Net uses mirror padding so that convolutions have valid context. These implementation details are crucial for exact alignment and are documented in the original paper [3].

### 7.3.4 Other Innovations: The Overlap-Tile Strategy

To handle very large images that would not fit into GPU memory, the authors also proposed a clever **overlap-tile strategy**. The large image is broken down into smaller, overlapping patches. The U-Net makes a prediction for each patch, but only the central region of the prediction is kept. These central regions are then stitched back together to form the final, full-resolution segmentation. The overlap ensures that the predictions at the edges of each patch, which can be less reliable, are discarded.

#### Weighted Loss for Touching Object Boundaries
Biomedical images often contain many small, touching objects (e.g., adjacent cells). U-Net addresses this by using a **weighted per-pixel loss** that increases the penalty on boundary pixels between touching instances, improving separation of adjacent objects [3].

#### Heavy Data Augmentation for Small Datasets
Because annotated biomedical datasets are typically small, U-Net relies on strong augmentation (random elastic deformations, rotations, shifts, and scalings) to improve generalization from limited data [3].

### 7.3.5 Loss Functions and Metrics
- **Losses:** Per-pixel cross-entropy remains a standard choice. In highly imbalanced settings (foreground vs. background), practitioners often adopt class weighting, boundary-aware weighting (as in U-Net), or Dice-style losses. The Dice loss optimizes directly for the Dice coefficient by minimizing 1 − Dice.
- **Metrics:** Common evaluation metrics include **Intersection-over-Union (IoU/Jaccard)** and the **Dice coefficient (F1 score on pixels)**. Boundary-sensitive metrics (e.g., boundary F1) are also used when precise edges are critical.

### 7.3.6 Key Contributions & Impact

U-Net was a landmark paper that has had a profound and lasting impact on the field.
*   It introduced a simple, elegant, and highly effective **symmetric encoder-decoder architecture** that became a standard for segmentation tasks.
*   It demonstrated the power of **concatenating skip connections** to combine coarse and fine features for precise localization.
*   It became the **de facto standard for biomedical image segmentation** and remains an incredibly strong baseline for this task today.
*   Its core architectural ideas have been adapted and extended to a huge variety of other computer vision tasks, from satellite image analysis to image restoration.

## 7.4 DeepLab: Dilated Convolutions and Context

### 7.4.1 Motivation: Preserve Resolution, Enlarge Context
A central challenge in segmentation is reconciling two needs: large receptive fields for context and high spatial resolution for precise boundaries. Pooling and striding enlarge the receptive field but destroy resolution. DeepLab’s insight is to recover context without further downsampling by using **atrous (dilated) convolutions** and to refine edges with a CRF.

### 7.4.2 Atrous/Dilated Convolutions: A Mathematical View
Given an input feature map \(X\) and a kernel \(K\) of size \((2r+1)\times(2r+1)\), a 2D dilated convolution with dilation rate \(d\) computes
$$
Y(i,j) = \sum_{m=-r}^{r} \sum_{n=-r}^{r} X(i + d\,m,\; j + d\,n)\;K(m,n)
$$
- **Effect**: Increases the effective receptive field by inserting \(d-1\) zeros between kernel elements, capturing wider context without additional pooling [4].
- **Use in DeepLab**: Replace some strided/pooling layers with atrous convolutions to maintain a higher output stride (e.g., 8 or 16) while preserving dense predictions [5, 6, 7].

### 7.4.3 ASPP: Multi-Scale Context in Parallel
DeepLab v2/v3 introduce **Atrous Spatial Pyramid Pooling (ASPP)**: several parallel atrous convolutions with different dilation rates, plus image-level pooling, whose outputs are concatenated. This captures objects at multiple scales in a parameter-efficient way [5, 6].

![Figure 7.3: The Atrous Spatial Pyramid Pooling (ASPP) module from DeepLabv3. It uses multiple parallel atrous convolutions with different rates to probe the incoming feature map at multiple scales, capturing rich multi-scale context. Image Source: Chen et al., 2017 [6].](/images/ch07_fig03_aspp_module.png)

### 7.4.4 CRF Post-Processing: Edge-Aligned Masks
Early DeepLab variants pair the CNN output with a **fully connected Conditional Random Field (CRF)** to sharpen boundaries and respect low-level edges. Dense pairwise terms with Gaussian kernels enable efficient mean-field inference and align predictions to image gradients [5, 8]. Later variants rely less on CRFs as decoders improve (v3+).

### 7.4.5 Evolution and Practical Trade-offs
- **DeepLab v1**: Atrous conv + fully connected CRF [5].
- **DeepLab v2**: Adds ASPP for robust multi-scale context [5].
- **DeepLab v3**: Refines atrous design, stronger ASPP; often drops CRF [6].
- **DeepLab v3+**: Adds a lightweight decoder to recover detail, with atrous separable convolutions for efficiency [7].
- **Pros**: Strong balance of context and detail; flexible output stride; excellent accuracy.
- **Cons**: Dilated convs can be memory-heavy; careless dilation choices may create gridding artifacts; training schedules matter.

## 7.5 Datasets, Metrics, and Evaluation

### 7.5.1 Common Benchmarks
- **PASCAL VOC 2012 (Segmentation)**: 21 classes (incl. background), moderate-scale images; widely used for ablations and classic comparisons [11].
- **Cityscapes**: Urban street scenes; fine annotations for 19 classes; emphasizes thin structures and boundaries; high-resolution images (1024×2048) [9].
- **ADE20K**: Diverse scene parsing with 150 classes; broad variability; challenging long-tail distribution [10].

### 7.5.2 IoU and mIoU for Segmentation
As we saw with bounding boxes in Chapter 4, Intersection over Union (IoU) is the standard for measuring overlap. For segmentation, the same principle is applied at the pixel level. The pixel-wise IoU for a given class is the ratio of correctly classified pixels of that class to the total number of pixels that are in either the prediction or the ground truth for that class.

Formally, for a class \(c\):
$$
\text{IoU}_c = \frac{\text{True Positives}_c}{\text{True Positives}_c + \text{False Positives}_c + \text{False Negatives}_c}
$$
The primary metric for evaluating a segmentation model's performance across an entire dataset is the **mean Intersection over Union (mIoU)**, which is simply the IoU score averaged over all classes:
$$
\text{mIoU} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{IoU}_c
$$
- **Notes**: mIoU treats all classes equally, which means performance on rare or small object classes can significantly impact the overall score. This makes it a comprehensive but sometimes challenging metric for datasets with severe class imbalance.

## 7.6 A Deeper Dive into Advanced Loss Functions

While the per-pixel cross-entropy loss is a powerful and standard baseline, it treats every pixel as an independent classification problem. This can be suboptimal for segmentation, especially in the face of extreme class imbalance (e.g., a few small foreground objects on a vast background) or when the precise structure of the output mask is critical. Several advanced loss functions have been developed to address these issues by optimizing metrics that are more closely related to the final segmentation quality.

### 7.6.1 Dice Loss: Directly Optimizing for Overlap

For tasks with severe class imbalance, the **Dice Loss** is one of the most popular and effective alternatives to cross-entropy. It is derived from the Dice Coefficient, a metric we discussed earlier that directly measures the overlap between the predicted and ground-truth masks. By framing the loss as a measure of non-overlap, the network is trained to directly maximize the segmentation overlap.

#### A Mathematical View
The Dice Coefficient (DC) for a single class between a predicted probability map \(P\) and a ground-truth binary mask \(G\) can be approximated in a differentiable way as:
$$
\text{DC} = \frac{2 \sum_{i} p_i g_i}{\sum_{i} p_i^2 + \sum_{i} g_i^2}
$$
where the sum is over all pixels $i$, $p_i$ is the predicted probability, and $g_i$ is the ground-truth label (0 or 1). The Dice Loss is then simply:
$$
\mathcal{L}_{\text{Dice}} = 1 - \text{DC}
$$
This formulation is numerically stable and backpropagates smoothly, forcing the network to focus on the agreement between the prediction and the ground truth, making it highly robust to class imbalance. It was popularized for deep learning-based segmentation in the V-Net architecture [12].

### 7.6.2 Lovász-Softmax: Directly Optimizing for mIoU

While Dice Loss optimizes for overlap, the final evaluation metric is often mean Intersection over Union (mIoU). The **Lovász-Softmax** loss provides a brilliant way to directly optimize a surrogate of the mIoU metric itself [13]. It is based on the Lovász extension of the Jaccard index (IoU), which allows the notoriously non-differentiable IoU metric to be relaxed into a loss function that can be optimized with gradient descent. While its mathematical formulation is complex, its practical impact is simple: it allows the network to be trained with a loss function that is much more closely aligned with the final evaluation metric, which can often lead to a significant boost in performance, especially on datasets with many classes.

### 7.6.3 Boundary-Aware Losses: Sharpening the Edges

A final class of specialized losses aims to solve a common failure mode of segmentation models: blurry or imprecise boundaries. Standard region-based losses like cross-entropy or Dice Loss treat all pixels equally. This means that a mistake on a thin boundary between two large regions is penalized the same as a mistake in the interior of a region. As a result, the network has little incentive to learn the precise, high-frequency details required for sharp edges, often leading to smoothed or "blobby" predictions.

Boundary-aware losses directly counteract this by explicitly increasing the penalty for errors on or near object boundaries. There are two primary families of these losses.

#### 1. Distance Map Weighting
This is the most common approach. Before training, a pre-processing step is used to compute a distance map from the ground-truth segmentation boundaries. This map is then used to assign a high weight to pixels that are close to an edge and a lower weight to pixels in the interior of a region. This weight map is then multiplied with a standard loss function like cross-entropy. The paper by Kervadec et al. that we previously cited uses this strategy to great effect, particularly for highly unbalanced medical imaging tasks where the foreground is small and the boundary is critical [14].

***A Mathematical View***

A weighted cross-entropy loss can be formulated as:
$$
\mathcal{L}_{\text{WeightedCE}} = - \frac{1}{N} \sum_{i=1}^{N} w(i) \cdot \left[ g_i \log(p_i) + (1-g_i)\log(1-p_i) \right]
$$
where \(w(i)\) is the pre-computed weight for pixel \(i\), which is high if the pixel is near a boundary and low otherwise.

#### 2. Gradient-Based Losses
Another approach is to directly penalize the difference between the spatial gradients of the predicted and ground-truth masks. A **Boundary Loss** can be formulated to compute the integral over the difference in the probability contours between the prediction and the ground truth. Other methods compute a gradient map (e.g., using a Sobel filter) for both the prediction and the ground truth and add a loss term (like L2 distance) that penalizes any mismatch. This forces the network to produce sharp probability changes at the same locations as the ground-truth edges [18].

These specialized losses are often combined with a region-based loss (e.g., $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Dice}} + \lambda \mathcal{L}_{\text{Boundary}}$) to ensure both high region overlap and crisp, accurate boundaries.

## 7.7 Practical Considerations and Inference Techniques

Beyond the core architecture and loss function, several practical techniques are crucial for achieving state-of-the-art performance and handling real-world data with a segmentation model.

### 7.7.1 Inference on Large Images: Sliding Window vs. Full-Image

Modern segmentation models are often trained on fixed-size patches (e.g., 512x512) for memory efficiency. At inference time, however, we must process a full, high-resolution image. There are two main strategies for this:
1.  **Full-Image Inference:** The simplest approach is to resize the entire image to a size the network can handle, run a single forward pass, and then resize the output mask back to the original image dimensions. This is very fast but can severely degrade the performance for small objects and fine details.
2.  **Sliding Window (Patch-Based) Inference:** A more accurate but slower approach is to slide a window across the high-resolution image, extracting patches of the training size. The model predicts a mask for each patch, and these masks are then stitched together to form the final full-resolution output. To avoid artifacts at the patch boundaries, it is common to use overlapping patches and blend the predictions in the overlapping regions (e.g., with a Gaussian weighting that gives more importance to the center of each patch). This is the same core idea as the "overlap-tile" strategy introduced by U-Net [3].

### 7.7.2 Test-Time Augmentation (TTA)

To further boost performance, a powerful technique is **Test-Time Augmentation (TTA)**. Instead of running inference on just the original image, the model also makes predictions on several augmented versions of the image (e.g., horizontally flipped, multi-scale versions). These augmented predictions are then reverse-transformed back to the original image's coordinate space and averaged together to produce the final, more robust segmentation mask. This ensemble-like approach can often provide a significant performance boost at the cost of increased inference time. The practice was popularized in early classification models like AlexNet (which averaged predictions from 10 crops) and used to achieve state-of-the-art results in models like ResNet [15, 16].

### 7.7.3 Mitigating Upsampling Artifacts

A common visual artifact associated with transposed convolutions is the "checkerboard pattern," which can arise from an uneven overlap in the upsampling operation. A simple and highly effective way to mitigate this is to avoid transposed convolutions altogether. A common alternative is to first use a simple, non-learnable upsampling method like **bilinear interpolation** to increase the feature map's resolution, and then apply a standard 3x3 convolution to refine the upsampled features. This "upsample-then-convolve" approach often leads to smoother and higher-quality masks and has become a standard design pattern in many modern segmentation architectures. For a definitive visual explanation of this phenomenon, see the Distill article on deconvolution and checkerboard artifacts [17].

## 7.8 Key Takeaways

- **FCN reframing**: Replacing fully-connected layers with convolutions yields a fully convolutional network that produces spatial outputs and enables dense, per-pixel prediction [1].
- **Learned upsampling**: Transposed convolution provides a learnable mechanism to recover resolution; it reverses the geometry of standard convolution but is not an inverse operator [2].
- **Multi-scale fusion**: Skip connections fuse semantic depth with spatial detail. FCN uses additive fusion at selected depths (32s/16s/8s), while U-Net concatenates encoder features at every scale for sharper masks [1, 3].
- **U-Net design**: Symmetric encoder-decoder with 2×2 transposed convolutions, concatenation-based skips, and careful cropping/alignment due to valid 3×3 convolutions; mirror padding helps border handling [3].
- **Large images, small data**: Overlap-tile inference handles large inputs; strong augmentation and boundary-weighted losses help when datasets are small and objects touch [3].
- **Losses and metrics**: Per-pixel cross-entropy is standard; class imbalance often benefits from weighting or Dice-style losses. Evaluate with IoU (Jaccard) and Dice; consider boundary metrics when edges matter.
- **Practice tips**: Verify alignment when mixing padding schemes, monitor checkerboard artifacts in upsampling, and balance resolution vs. memory. See Chapter 2 for convolution arithmetic and Chapter 7 references for upsampling specifics [2].

---
## References
1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 3431-3440).
2. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical image computing and computer-assisted intervention* (pp. 234-241). Springer, Cham.
4. Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. In *International Conference on Learning Representations (ICLR)*.
5. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(4), 834–848.
6. Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. *arXiv preprint arXiv:1706.05587*.
7. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In *European Conference on Computer Vision (ECCV)*.
8. Krähenbühl, P., & Koltun, V. (2011). Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials. In *Advances in Neural Information Processing Systems (NeurIPS)*.
9. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., ... & Schiele, B. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 3213–3223).
10. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., & Torralba, A. (2017). Scene Parsing through ADE20K Dataset. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 633–641).
11. Everingham, M., Van Gool, L., Williams, C. K., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision*, 88(2), 303–338.
12. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In *2016 Fourth International Conference on 3D Vision (3DV)* (pp. 565-571).
13. Berman, M., Rannen Triki, A., & Blaschko, M. B. (2018). The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4413-4421).
14. Kervadec, H., Bouchareb, J., Wolf, D., Joskowicz, L., & Montagnon, E. (2019). Boundary loss for highly unbalanced segmentation. In *International Conference on Medical Imaging with Deep Learning* (pp. 285-296).
15. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
16. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. In *Advances in Neural Information Processing Systems 25* (pp. 1097–1105).
17. Odena, A., Dumoulin, V., & Olah, C. (2016). Deconvolution and Checkerboard Artifacts. *Distill*. https://distill.pub/2016/deconv-checkerboard/
18. Hayder, Z., He, X., & Salzmann, M. (2017). Boundary-aware instance segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 5636-5644).
