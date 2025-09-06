# Chapter 6: Beyond Bounding Boxes: Modern Detection Architectures

## 6.1 Introduction: The Problem with Anchors

The object detectors we have seen so far, from the two-stage Faster R-CNN to the one-stage RetinaNet, all rely on a single, core concept: **anchor boxes**. These pre-defined reference boxes of various sizes and aspect ratios are a powerful tool, but they also have significant drawbacks. They are a "hyperparameter," meaning the designer must carefully choose the right set of scales and aspect ratios for a given dataset. They also contribute to the extreme class imbalance problem, as the vast majority of the ~100,000 anchors are negatives.

A new wave of research began to ask a fundamental question: are anchor boxes really necessary? This led to the development of **anchor-free** detectors, a diverse and powerful set of modern architectures. In this chapter, we will explore the first major family of these: keypoint-based detectors.

## 6.2 CornerNet: Detecting Objects as Paired Keypoints

The first influential paper to propose a truly competitive anchor-free detector was the 2018 paper "CornerNet: Detecting Objects as Paired Keypoints" by Law and Deng [1]. The core idea was to completely reframe the problem. Instead of classifying and refining thousands of anchor boxes, CornerNet proposed a much simpler idea: **an object is just a pair of keypoints**—its top-left corner and its bottom-right corner.

### 6.2.1 The Core Insight: A Keypoint Detection Problem

By reframing object detection as a keypoint detection problem, CornerNet was able to eliminate the need for anchor boxes entirely. This simplified the design of the network and removed the complex hyperparameter tuning associated with anchors.

![Figure 6.1: The CornerNet Architecture. An input image is passed to a ConvNet, which produces heatmaps for the top-left and bottom-right corners, as well as an embedding vector for each detected corner. The network is trained to produce similar embeddings for corners that belong to the same object, allowing them to be grouped together. Image Source: Law and Deng, 2018 [1].](../../images/ch06_fig01_cornernet_architecture.png)

### 6.2.2 The CornerNet Pipeline: A Step-by-Step Guide

The CornerNet architecture works as follows:
1.  **Predict Corner Heatmaps:** The network takes an image and produces two separate **heatmaps**: one that predicts the locations of all top-left corners in the image, and another that predicts the locations of all bottom-right corners. These heatmaps are trained with a focal-loss-like penalty to handle the imbalance between corner locations and empty space.

2.  **Predict Embedding Vectors:** This is the key to grouping the corners. For each detected corner, the network also predicts a 1-dimensional **embedding vector**. The network is trained with a loss function that encourages the embeddings for a pair of corners that belong to the same object to be very similar, and the embeddings for corners from different objects to be dissimilar.

3.  **Group Corners:** After the network has made its predictions, a simple post-processing step is used. The system looks at all the detected top-left corners and all the detected bottom-right corners. It then calculates the distance between their embedding vectors. If the distance is below a certain threshold, the pair of corners is grouped together to form a final bounding box.

### 6.2.3 A Deeper Look: Corner Pooling

A key challenge in this approach is that corners are often located outside the object (e.g., the top-left corner of a bounding box for a person may be in the empty space above their head). This makes it hard for a standard pooling layer to gather the necessary information to localize the corner.

To solve this, the authors introduced a new, specialized pooling layer called **corner pooling**. For a top-left corner, for example, the corner pooling layer takes the feature map and, at each location, it max-pools all the features to the right of it and all the features below it. This gives the network the information it needs to find the top-most and left-most boundaries of the object, which is exactly what is needed to localize a top-left corner.

![Figure 6.2: Corner pooling. For each channel, we take the maximum values (red dots) in two directions (red lines), each from a separate feature map, and add the two maximums together (blue dot). Image Source: Law and Deng, 2018 [1].](../../images/ch06_fig02_corner_pooling.png)

### 6.2.4 Key Contributions & Impact

CornerNet was a major conceptual breakthrough.
*   It was the **first truly competitive anchor-free detector**, proving that this new paradigm could achieve state-of-the-art results.
*   It introduced a **new way of thinking about object detection**, as a keypoint detection and grouping problem, which inspired a huge wave of follow-up research into other anchor-free methods.
*   It showed that clever, task-specific network modules like **corner pooling** could significantly improve performance.

### 6.2.5 Architectural Details: The Hourglass Backbone

A key to CornerNet's success was its choice of a powerful backbone network. The authors used a **Stacked Hourglass Network**, an architecture originally designed for the task of human pose estimation [4].

The choice was deliberate and gets to the heart of the challenge of a keypoint-based approach. The top-left corner of a bounding box is an abstract concept, not a physical part of the object. For example, the top-left corner of a bounding box around a person is often empty space, located above their head and to the left of their shoulder. The local features at that exact pixel location are therefore often meaningless or even misleading.

To correctly identify that empty location as a "top-left corner," the network must look for evidence *away* from that point. It needs to find two things simultaneously: the top-most boundary of the object (e.g., the top of the person's head) and the left-most boundary of the object (e.g., the person's shoulder). This requires the network to combine fine-grained local evidence with a more global, contextual understanding of the object's shape.

This is exactly what the Hourglass network and the corner pooling layer are designed to do.
*   **The Hourglass Network** has a symmetric, encoder-decoder structure. It first downsamples the feature maps to capture coarse, global, semantic features about the entire scene. It then upsamples the feature map back to its original resolution, using skip connections to combine the rich, global information with the fine-grained, local information. This structure allows the final prediction for each pixel to be informed by both immediate local features and the global context of the entire image.
*   **Corner Pooling**, as we discussed, explicitly searches for these distant boundaries by max-pooling features horizontally and vertically.

This combination of a powerful, multi-scale backbone and a task-specific pooling strategy is precisely what is needed for accurate corner detection.

While it was later surpassed in performance by other anchor-free methods, CornerNet's role in challenging the dominance of anchor-based detectors was a pivotal moment in the history of the field.

## 6.3 CenterNet: Detecting Objects as Points

While CornerNet was a brilliant new paradigm, the process of grouping corner pairs with embedding vectors was a complex, post-processing step that could be a source of errors. A 2019 paper, "Objects as Points" by Zhou et al., proposed an even simpler and more direct anchor-free approach that became highly influential: **CenterNet** [2].

### 6.3.1 The Core Insight: An Object is a Single Point

The core insight of CenterNet is to model an object not as a box, and not as a pair of keypoints, but as a single point: its center. This simplifies object detection to its absolute essence: find the center of the object, and then predict its size.

*(Placeholder for a diagram of the CenterNet concept: an image of a cat with a single heatmap highlighting the center of the cat, and arrows indicating the regression of the width and height from that center point)*

### 6.3.2 The CenterNet Pipeline: A Step-by-Step Guide

The CenterNet pipeline is a model of simplicity and elegance. It uses a standard backbone network (like ResNet) to produce a high-level feature map, typically at 1/4 the resolution of the input image. From this single feature map, it predicts three separate outputs.

**1. Predict a Center Heatmap**
The primary output is a heatmap with C channels, where C is the number of object classes. This heatmap is trained to predict the locations of the centers of all objects in the image.
*   **Ground Truth:** To create the training target, a "ground truth" heatmap is generated for each class. This is a black image where a Gaussian kernel is "splatted" onto the location of each object's center for that class.
*   **Training:** The network is trained to produce a heatmap that matches this ground truth. Because the vast majority of the heatmap is zero (background), a variant of the **Focal Loss** [3] is used to focus the training on the rare positive locations (the object centers).

**2. Regress Object Properties**
In parallel to the heatmap, two other regression heads are trained to predict the other necessary properties of the object at each location.
*   **Object Size:** A 2-channel output predicts the width (w) and height (h) of the bounding box for each object. This is trained with a standard L1 loss. Crucially, this loss is only applied at the locations of the ground-truth object centers.
*   **Center Offset:** A 2-channel output predicts a small, local offset `(Δx, Δy)`. This is a clever trick to recover the precision that was lost when the image was downsampled by a factor of 4. This allows the model to place the object center with much higher accuracy than the coarse grid of the feature map would otherwise allow.

**3. Generate Detections from Peaks**
The inference process is remarkably simple and clean:
1.  The network performs a single forward pass to produce the heatmap, the size map, and the offset map.
2.  The peaks in the heatmap are identified. A simple 3x3 max pooling operation is used to ensure that only one peak is detected per object, which is a much faster and simpler alternative to the standard Non-Maximum Suppression (NMS) used in other detectors.
3.  For each detected peak, the corresponding values for the object's size (w, h) and the center offset (Δx, Δy) are extracted from the other two output maps at the same spatial location.
4.  These values are combined to produce the final, high-precision bounding box for each detected object.

### 6.3.3 Extensibility: Objects as Points is a General Framework
A key advantage of the CenterNet framework is its extensibility. The same simple idea of finding a keypoint and regressing its properties can be used for many other tasks. For example, to perform human pose estimation, one can simply add another regression head to predict the locations of the human joints relative to the center point. This makes CenterNet a highly flexible and powerful general-purpose perception algorithm.

### 6.3.4 Key Contributions & Impact

CenterNet offered a compelling new direction for object detection.
*   **Simplicity and Elegance:** It is one of the simplest and cleanest modern object detector designs, with no need for complex concepts like anchors or corner grouping.
*   **Speed and Accuracy:** It provided an excellent trade-off between speed and accuracy, outperforming many anchor-based one-stage detectors at the time.
*   **Extensibility:** The "objects as points" framework is highly extensible and can be applied to many other perception tasks beyond simple object detection.

CenterNet, along with CornerNet, firmly established anchor-free methods as a powerful and viable alternative to the anchor-based paradigm, and its simple and extensible design has had a lasting influence on the field.

## 6.4 FCOS: A Fully Convolutional Approach

While keypoint-based methods like CornerNet and CenterNet offered one path away from anchors, a parallel line of research explored a different, perhaps more intuitive, anchor-free paradigm: treating object detection as a dense, per-pixel prediction problem, much like semantic segmentation. The most influential paper in this area was the 2019 work "FCOS: Fully Convolutional One-Stage Object Detection" by Tian et al. [6].

### 6.4.1 The Core Insight: Per-Pixel Bounding Box Regression

The core insight of FCOS is to predict a bounding box directly from every single foreground pixel on a feature map. Instead of using pre-defined anchor boxes, FCOS simply predicts a 4D vector `(l, t, r, b)` for each pixel. This vector represents the distances from that pixel's location to the four sides of the bounding box that encloses it: **l**eft, **t**op, **r**ight, and **b**ottom.

### 6.4.2 The FCOS Architecture and Pipeline

FCOS uses a standard backbone with a Feature Pyramid Network (FPN) to produce a multi-scale pyramid of feature maps. For each pixel location on each level of the pyramid, the prediction head outputs:
1.  **A Classification Score:** A C-dimensional vector of class probabilities.
2.  **A 4D Bounding Box Vector:** The `(l, t, r, b)` distances.
3.  **A "Center-ness" Score:** This is a clever innovation to solve a key problem. A standard per-pixel regression approach creates a large number of low-quality bounding boxes predicted from pixels far from the center of an object. To solve this, FCOS adds a single-channel head that predicts a "center-ness" score for each pixel. This score is trained to be high for pixels near the center of the object and low for pixels far away. During inference, this center-ness score is multiplied by the classification score, which effectively down-weights the scores of these low-quality, off-center boxes, allowing them to be easily filtered out by NMS.

This per-pixel prediction approach elegantly avoids all the complex hyperparameter tuning related to anchor boxes and provides a simple and powerful alternative to the keypoint-based methods.

## 6.5 DETR: Object Detection as a Set Prediction Problem

The final modern architecture we will discuss in this chapter represents another radical rethinking of the entire object detection problem. So far, every detector we have seen relies on a significant amount of hand-designed components and post-processing steps. They use anchors, grids, or keypoint grouping rules to deal with the inherent duplicates in their raw predictions, and they all rely on a final **Non-Maximum Suppression (NMS)** step to produce the final, unique set of detections.

The 2020 paper "End-to-End Object Detection with Transformers" by Carion et al. from Facebook AI Research introduced the **DEtection TRansformer**, or **DETR**, which aimed to eliminate all of this complexity [5].

### 6.5.1 The Core Insight: A Direct Set Prediction Problem

The core insight of DETR is to treat object detection as a direct **set prediction problem**. Instead of producing a massive number of intermediate predictions that need to be filtered and refined, the model should simply output the final, correct, and unique set of objects directly.

To achieve this, the authors turned to the powerful **Transformer architecture**, which we introduced as a foundational concept in Chapter 2. By leveraging the Transformer's global self-attention mechanism, the authors were able to create a model that could reason about the entire image and the relationships between objects at once, eliminating the need for hand-designed heuristics like NMS.

*(Placeholder for a diagram of the DETR architecture, showing the CNN backbone, the Transformer encoder-decoder, the object queries, and the final feed-forward networks)*

### 6.5.2 The DETR Pipeline: A Step-by-Step Guide

The DETR architecture is a true end-to-end system that replaces the hand-designed components of previous detectors with a powerful, general-purpose Transformer.

1.  **Step 1: CNN Backbone and Positional Encodings:** The image is first passed through a standard CNN backbone (like ResNet) to extract a rich, spatial feature map. Because the Transformer architecture that follows is permutation-invariant (it does not have a built-in understanding of spatial relationships), this feature map is then supplemented with a **fixed positional encoding**, which is added to the feature map to give each pixel a unique spatial signature.

2.  **Step 2: The Transformer Encoder:** The feature map is flattened into a sequence and fed into a **Transformer encoder**. The encoder is a stack of self-attention layers. The **self-attention** mechanism allows every pixel in the feature map to "look at" every other pixel, calculating a similarity score and building a new representation for each pixel that is a weighted sum of all other pixels. This process is repeated through multiple layers, allowing the network to build a rich, global, and contextual understanding of the entire scene.

3.  **Step 3: Object Queries and the Transformer Decoder:** A fixed number, N, of **object queries** are then passed to the **Transformer decoder**. These are not just "empty slots," but **learnable embeddings** that are part of the model's parameters. Each query learns to specialize in finding a certain type of object or a certain region of the image. The decoder is also a stack of layers, but its attention mechanism is more complex:
    *   **Self-Attention:** The object queries first pass through a self-attention layer. This allows them to communicate with each other. This is a crucial step that helps the model learn to avoid making duplicate predictions for the same object, as the queries can learn to "claim" different objects.
    *   **Cross-Attention:** The queries then attend to the output of the encoder. This **cross-attention** mechanism allows each query to "look at" the entire, context-rich feature map of the image and gather the information it needs to make its prediction.

4.  **Step 4: Prediction Heads:** The final output of the decoder is a set of N transformed embeddings. Each of these is passed to a shared Feed-Forward Network (FFN) that acts as the prediction head. This FFN has two branches: one that predicts the bounding box coordinates (as 4 numbers) and another that predicts the class label (using a Softmax over the C classes plus a special "no object" class).

### 6.5.3 A Deeper Look: The Bipartite Matching Loss

The final piece of the puzzle is the loss function, which is what enforces the uniqueness of the predictions and eliminates the need for NMS. DETR uses a **bipartite matching loss**.
*   During training, it finds the optimal one-to-one matching between the N predicted boxes and the M ground-truth boxes in the image.
*   The loss is then calculated based on this optimal pairing. The N-M predictions that are not matched to any ground-truth object are assigned the "no object" class.

This clever loss function forces the network to learn to assign a single, unique prediction to each object in the image, making the duplicate-removal step of NMS completely unnecessary.

### 6.5.4 Key Contributions & Impact

DETR introduced a completely new and powerful paradigm for object detection.
*   **Truly End-to-End:** It was the first architecture to be truly end-to-end, with no hand-designed components like anchors or NMS.
*   **Transformer for Vision:** It successfully demonstrated how the powerful Transformer architecture, with its global self-attention mechanism, could be applied to a core computer vision problem.
*   **A New Research Direction:** While the original DETR was slow to train and had some issues with small objects, its conceptual simplicity and power inspired a massive wave of follow-up research into Transformer-based detectors, which have since become the state-of-the-art in the field.

## 6.6 Pushing Efficiency and Scale: Modern Backbones

While much of the research we have discussed focuses on the "head" of the detector (the part that makes the final predictions), the performance of any detector is fundamentally dependent on the quality of its "backbone"—the deep convolutional network that extracts the features from the input image. In the final section of this chapter, we will look at two of the most influential modern backbone architectures that pushed the boundaries of both accuracy and efficiency.

### 6.6.1 EfficientNet: Rethinking Model Scaling

For years, the standard way to improve a network's accuracy was to make it bigger. This was typically done along one of three dimensions:
-   **Depth:** Making the network deeper by adding more layers (e.g., ResNet-50 vs. ResNet-101).
-   **Width:** Making the network wider by increasing the number of channels in each layer.
-   **Resolution:** Feeding the network a higher-resolution input image.

![Figure 6.3: Model Scaling. (a) is a baseline network example; (b)-(d) are conventional scaling that only increases one dimension of network width, depth, or resolution. (e) is our proposed compound scaling method that uniformly scales all three dimensions with a fixed ratio. Image Source: Tan and Le, 2019 [7].](../../images/ch06_fig_efficientnet_scaling.png)

The 2019 paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" by Tan and Le argued that this one-dimensional, brute-force scaling was suboptimal [7]. They showed through careful empirical study that these three dimensions are not independent; scaling one provides diminishing returns, but scaling them together in a balanced way can yield significant gains in both accuracy and efficiency.

#### The Core Insight: Compound Scaling
The authors proposed a principled **compound scaling** method. The idea is to use a single compound coefficient, $\phi$, to uniformly scale the network's width, depth, and resolution together.
$$
\begin{align*}
\text{depth: } d &= \alpha^\phi \\
\text{width: } w &= \beta^\phi \\
\text{resolution: } r &= \gamma^\phi
\end{align*}
$$
Here, $\alpha, \beta, \gamma$ are constants determined by a small grid search on the baseline network. The user can then simply scale the model up or down by choosing a value for $\phi$. This ensures that as the model gets bigger, it gets proportionally deeper, wider, and receives a larger input image.

#### The EfficientNet-B0 Baseline and Architecture Family
To apply this scaling rule, they first developed a highly efficient, mobile-optimized baseline architecture, **EfficientNet-B0**, which was found through a neural architecture search. This baseline is built on the **mobile inverted bottleneck block (MBConv)** from MobileNetV2 [9], which is a highly efficient building block that uses depthwise separable convolutions.

They then used their compound scaling rule with the B0 baseline to create a family of models, from the tiny `EfficientNet-B0` (by setting $\phi=0$) to the massive `EfficientNet-B7`. This family of models achieved new state-of-the-art accuracy on ImageNet while being significantly smaller and faster than previous models like ResNet or Inception.

#### Impact and Legacy
EfficientNet had a profound impact on the field. It demonstrated that model scaling was not just about making things bigger, but about doing so intelligently. The work proved that efficiency and accuracy were not necessarily a trade-off, but could be achieved simultaneously through principled design. EfficientNets and their successors quickly became a popular and powerful backbone choice for object detectors and many other vision tasks.

### 6.6.2 ConvNeXt: The ConvNet Strikes Back

By the early 2020s, the Vision Transformer (ViT) and its successors had begun to challenge the long-held dominance of ConvNets. A 2022 paper, "A ConvNet for the 2020s" by Liu et al., asked a fascinating question: was the performance of Transformers due to their superior architecture, or was it due to the modern, sophisticated training techniques that had been developed alongside them? [8]

The authors of **ConvNeXt** performed a methodical study to answer this. They started with a standard ResNet-50 and progressively "modernized" it, borrowing design ideas directly from modern Transformers like the Swin Transformer [10].

#### A Recipe for Modernizing a ConvNet
The key architectural changes, applied sequentially, included:
-   **Changing the stage compute ratio:** The distribution of blocks in a ResNet (e.g., 3, 4, 6, 3 blocks in the four stages of ResNet-50) was changed to the 1:1:3:1 ratio used in Swin Transformers, which better allocates parameters for performance.
-   **Using a "Patchify" Stem:** The initial stem of the ResNet (a large 7x7 convolution with a stride of 2) was replaced with a much smaller 4x4 non-overlapping convolution with a stride of 4, directly mimicking the patch embedding layer of a ViT.
-   **Using Depthwise Convolutions:** Inspired by the efficient ResNeXt block, the standard convolutions were replaced with depthwise separable convolutions, a highly efficient variant that processes spatial and channel information separately. This also had the effect of mimicking the weighted sum operation in self-attention, which operates on a per-channel basis.
-   **Inverting the Bottleneck Block:** The classic ResNet bottleneck block has a "wide-narrow-wide" shape. The ConvNeXt authors inverted this to the "narrow-wide-narrow" shape of the MBConv block, which proved to be more efficient.
-   **Using Large Kernel Sizes:** To increase the receptive field and mimic the global attention of Transformers, the small 3x3 kernels were moved up in the network and replaced with large 7x7 depthwise convolutions.
-   **Modernizing the Guts:** Finally, a series of micro-design changes were made, replacing classic components with their modern equivalents: **GELU** replaced ReLU, **Layer Normalization** replaced Batch Normalization, and fewer activation functions and normalization layers were used overall, again mirroring Transformer design.

#### The Result and Its Impact
The authors paired this modernized ConvNet architecture with the latest training recipes (like the AdamW optimizer and advanced data augmentation). The result was a pure ConvNet that not only matched but in some cases surpassed the performance of the state-of-the-art Transformer models on major benchmarks. This work was highly influential, proving that the core convolutional architecture remains incredibly powerful and that many of the gains seen in recent years were due to a combination of both architectural improvements and better training methodologies.

## 6.7 Key Takeaways

- **Anchor-free shift**: Modern detectors can localize objects without anchors, avoiding heavy hyperparameter tuning and class imbalance concerns inherent to anchor sets [1, 2].
- **CornerNet**: Frames detection as paired keypoint prediction; uses corner embeddings for grouping and specialized corner pooling with a stacked hourglass backbone for multi-scale context [1, 4].
- **CenterNet**: Reduces detection to center heatmap + size + offset; trains with focal-style loss for sparse positives, uses simple local-maximum peak selection instead of NMS, and extends naturally to related tasks [2, 3].
- **FCOS**: Treats detection as dense per-pixel regression; predicts classification, center-ness, and bounding box for each pixel; uses FPN for multi-scale features, and NMS for final filtering [6].
- **DETR**: Recasts detection as direct set prediction with object queries and bipartite matching; eliminates NMS and hand-crafted priors but requires longer training and may struggle with small objects; see Chapter 2 for Transformer fundamentals [5].
- **Modern Backbones**: Performance is deeply tied to the backbone; **EfficientNet** showed the power of principled, compound scaling for efficiency and accuracy [7], while **ConvNeXt** demonstrated that modernized ConvNets can match Transformer performance [8].
- **Choosing a method**: Use keypoint/center-based models for simplicity and speed with strong backbones; consider DETR-style models for clean end-to-end pipelines and global reasoning when you can afford training time.
- **Practical notes**: For keypoint methods, tune Gaussian radii and peak suppression; resolution and backbone choice drive small-object performance. For Transformer-based methods, monitor convergence schedules and data augmentation; hybrid backbones often help.

---
## References
1. Law, H., & Deng, J. (2018). Cornernet: Detecting objects as paired keypoints. In *Proceedings of the European conference on computer vision (ECCV)* (pp. 734-750).
2. Zhou, X., Wang, D., & Krähenbühl, P. (2019). Objects as points. *arXiv preprint arXiv:1904.07850*.
3. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision* (pp. 2980-2988).
4. Newell, A., Yang, K., & Deng, J. (2016). Stacked hourglass networks for human pose estimation. In *European conference on computer vision* (pp. 483-499). Springer, Cham.
5. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. In *European conference on computer vision* (pp. 213-229). Springer, Cham.
6. Tian, Z., Shen, C., Chen, H., & He, T. (2019). Fcos: Fully convolutional one-stage object detection. In *Proceedings of the IEEE/CVF international conference on computer vision* (pp. 9627-9636).
7. Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. In *International conference on machine learning* (pp. 6105-6114). PMLR.
8. Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A convnet for the 2020s. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 11976-11986).
9. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510-4520).
10. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF international conference on computer vision* (pp. 10012-10022).
