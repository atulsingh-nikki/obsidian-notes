# Chapter 14: Stepping into the Third Dimension

## 14.1 Introduction: Beyond the Flat Image

Our journey so far has largely been confined to the 2D world of images and videos. While this is a powerful domain, the real world is three-dimensional. To build machines that can perceive, navigate, and interact with their environment — robots, autonomous vehicles, AR/VR systems — we must equip them with the ability to understand and synthesize 3D structure. The past decade has seen an explosion of research on exactly this, driven less by new loss functions than by a more basic question: *how should a 3D scene be represented so that a neural network can learn on it?*

That question matters because 3D has no single canonical format, and each choice carries baggage:

- **Voxels** — a 3D grid of occupancy or features. Intuitive and CNN-friendly (just use 3D convolutions), but memory grows as the cube of resolution, so detail is expensive.
- **Meshes** — vertices and faces. Compact and standard in graphics, but the connectivity is irregular and hard for a fixed-shape network to predict.
- **Point clouds** — an unordered set of (X, Y, Z) coordinates, exactly what a LiDAR or depth sensor returns. The most direct representation, but standard CNNs cannot consume an unordered set.
- **Implicit fields** — represent the scene as a *function* (of position, or position and view direction) rather than as explicit geometry. Continuous and resolution-free, but you must render the function to see it.

This chapter follows three landmark ideas, each of which took one of these representations seriously and unlocked a wave of research: **PointNet**, which made raw point clouds learnable; **NeRF**, which represented whole scenes as a neural implicit field; and **3D Gaussian Splatting**, which returned to an explicit representation to make that quality real-time. The throughline is the theme that has recurred all book: the right *representation* often matters more than the network on top of it.

## 14.2 Working with Raw 3D Data: PointNet

The most raw 3D representation — the point cloud — is also the most awkward for deep learning. A point cloud is a *set*: it has no grid, no fixed ordering, and no notion of a neighbor built in. A CNN's whole machinery assumes a regular grid; feeding it shuffled coordinates is a category error.

The 2017 paper "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" by Qi et al. gave the first effective architecture for learning directly on raw point sets [1].

### 14.2.1 The Core Insight: Learning on Unordered Sets

The defining property PointNet must respect is **permutation invariance**: if the input is a set, the output must not change when the points are reordered. PointNet achieves this with a strikingly simple recipe:

1.  **Point-wise feature learning.** Each point is passed *independently* through a shared multi-layer perceptron $h(\cdot)$, lifting it from three coordinates to a high-dimensional feature. Because the same MLP is applied to every point, order is irrelevant at this stage.
2.  **Symmetric aggregation.** A single symmetric function — max pooling — collapses the $n$ per-point features into one global descriptor. Max is permutation-invariant by construction: the maximum of a set does not depend on the order you read it in.
3.  **Prediction head.** A final MLP $g(\cdot)$ maps the global descriptor to an output (a class score, or, by concatenating global and per-point features, a per-point segmentation label).

The authors formalize *why* this suffices: any continuous set function can be approximated arbitrarily well in the form

$$
f(\{x_1, \dots, x_n\}) \approx g\!\left(\underset{i=1,\dots,n}{\text{MAX}}\; h(x_i)\right)
$$

where the MAX is taken element-wise over the per-point feature vectors. In words: a shared per-point map followed by a symmetric aggregator is not just a convenient hack — it is a universal approximator for functions on sets. This is the theoretical heart of the paper.

![Figure 14.1: PointNet. Each point is mapped independently by a shared MLP; a symmetric max-pool over all points yields an order-invariant global feature that a final MLP turns into a class or per-point segmentation. The identity f({xᵢ}) ≈ g(MAXᵢ h(xᵢ)) is what guarantees permutation invariance. Schematic; after Qi et al., 2017 [1].](Books/Book_Object_Detection/images/ch14_fig01_pointnet.svg)

PointNet also inserts small learned **alignment networks** (T-Nets) that predict a rigid transform to canonicalize the input points and, later, the features — nudging the network toward invariance to rigid motion of the whole object.

### 14.2.2 Critical Analysis and Legacy

PointNet's elegance is also its limitation. Because features are computed *per point* and then pooled *globally*, the architecture has **no notion of local neighborhoods** — it cannot, by itself, capture that a cluster of nearby points forms a wing or a wheel. This is the exact analogue of trying to classify an image from a single global pooling of independent pixels, with none of the local structure that made CNNs work in the first place (Chapter 2).

The immediate successor, **PointNet++** [2], fixed this by applying PointNet hierarchically over local neighborhoods — recovering the multi-scale, local-to-global processing that convolutions gave 2D vision. The lineage continued into point-based detectors and, later, transformer-based point models. PointNet's lasting contribution is conceptual: it showed that respecting the *symmetry of the data* (here, permutation invariance) is the key design principle for a new modality — the same lesson that motivated translation equivariance in CNNs and would later motivate the token treatment in ViTs (Chapter 13).

## 14.3 Novel View Synthesis: Neural Radiance Fields (NeRF)

PointNet asked how to *understand* existing 3D data. A different, more audacious line of work asked how to *create* a photorealistic 3D representation of a scene from nothing but a handful of ordinary 2D photographs. The 2020 paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" by Mildenhall et al. answered it with startling quality [3].

### 14.3.1 The Core Insight: A Scene as a Continuous Function

NeRF's radical move is to store the scene not as geometry but as the *weights of a small MLP*. That network represents a continuous **5D radiance field**: given a 3D location $(x, y, z)$ and a viewing direction $(\theta, \phi)$, it outputs a **volume density** $\sigma$ (how much this point blocks light — roughly, how "solid" it is) and a **view-dependent color** $c = (r, g, b)$. Making color depend on viewing direction is what lets NeRF reproduce specular highlights and reflections that change as you move.

### 14.3.2 The Rendering Process

To produce one pixel of a novel view, NeRF borrows classical **volume rendering**:

1.  Cast a camera ray into the scene through that pixel.
2.  Sample many points along the ray; query the MLP at each to get $(\sigma_i, c_i)$.
3.  Composite those samples front-to-back into a single color. In discrete form,

$$
\hat{C}(r) = \sum_{i=1}^{N} T_i \,\bigl(1 - \exp(-\sigma_i \delta_i)\bigr)\, c_i, \qquad T_i = \exp\!\left(-\sum_{j < i} \sigma_j \delta_j\right)
$$

where $\delta_i$ is the distance between adjacent samples and $T_i$ is the **transmittance** — the fraction of light that survives from the camera to sample $i$ without being absorbed earlier. A point contributes color only if it is dense ($\sigma_i$ high) *and* still visible ($T_i$ high). This rendering step is fully differentiable, so the whole system trains end-to-end by minimizing the photometric error between rendered and ground-truth pixels. No 3D supervision is needed — only posed images.

![Figure 14.2: NeRF renders a pixel by sampling points along its camera ray, querying an MLP for density and view-dependent color at each, and integrating them via volume rendering. Training minimizes the difference between rendered and real pixels. Schematic; after Mildenhall et al., 2020 [3].](Books/Book_Object_Detection/images/ch14_fig02_nerf_rendering.svg)

### 14.3.3 Two Ideas That Made It Work

NeRF's quality hinged on two details worth calling out:

- **Positional encoding.** A plain MLP fed raw coordinates produces blurry, low-frequency output — networks are biased toward smooth functions. NeRF maps each input coordinate through a bank of sinusoids of geometrically increasing frequency,
$$
\gamma(p) = \bigl(\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\bigr),
$$
letting the same small MLP represent sharp, high-frequency detail. (This is the same Fourier-feature idea that underlies positional information in Transformers.)
- **Hierarchical sampling.** A first "coarse" network finds where matter is along each ray, so a second "fine" network can concentrate its samples there instead of wasting them on empty space.

### 14.3.4 Critical Analysis

NeRF produced view synthesis of unprecedented fidelity, but with sharp caveats. It is **per-scene**: each NeRF is optimized from scratch for one scene and does not generalize to others. Training is slow (hours), and — most consequentially — **rendering is slow**, because every pixel of every frame requires hundreds of MLP evaluations. A torrent of follow-up work attacked these costs: Mip-NeRF addressed aliasing across scales [4], and Instant-NGP used a multi-resolution hash grid to cut training from hours to seconds [5]. But the fundamental tension — an implicit, neural representation is compact and continuous yet expensive to query — set the stage for the next idea.

## 14.4 Real-Time Rendering: 3D Gaussian Splatting

If NeRF's bottleneck is that the scene is trapped inside an MLP that must be queried point by point, the fix is to make the representation **explicit** again. The 2023 paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering" by Kerbl et al. did exactly that, matching or beating NeRF quality while rendering at real-time frame rates [6].

### 14.4.1 The Core Insight: An Explicit Cloud of Gaussians

The scene is represented directly as a large collection of **3D Gaussians** — soft, fuzzy ellipsoidal blobs. Each Gaussian carries a position (mean) $\mu$, a covariance $\Sigma$ describing its shape and orientation, an opacity, and a view-dependent color (stored as spherical-harmonic coefficients):

$$
G(x) = \exp\!\left(-\tfrac{1}{2}(x - \mu)^{\top} \Sigma^{-1} (x - \mu)\right), \qquad \Sigma = R S S^{\top} R^{\top}
$$

The covariance is factored into a rotation $R$ and a scale $S$ so it stays a valid (positive semi-definite) covariance throughout optimization while remaining easy to differentiate.

### 14.4.2 Splatting: Rasterization, Not Ray Marching

Rendering — "splatting" — is a rasterization process, the same family of operations GPUs were built for. The 3D Gaussians are projected onto the image plane as 2D Gaussians, sorted by depth, and **alpha-composited** front-to-back per tile. There is no per-pixel network query; the work is a highly parallel projection-and-blend that maps beautifully onto GPU pipelines, which is why it hits real-time. The representation is still learned by differentiable rendering against the input photos, with an **adaptive densification** step that clones or splits Gaussians where detail is missing and prunes those that are transparent or redundant.

### 14.4.3 NeRF vs. 3D Gaussian Splatting

| Property | NeRF | 3D Gaussian Splatting |
|---|---|---|
| Representation | Implicit (MLP weights) | Explicit (set of 3D Gaussians) |
| Rendering | Ray marching + MLP queries | Tile-based rasterization ("splatting") |
| Render speed | Slow (seconds/frame) | Real-time (interactive) |
| Memory footprint | Small (one MLP) | Larger (millions of Gaussians) |
| Editability | Hard (baked into weights) | Easier (Gaussians are explicit primitives) |

The two are best read as a pendulum swing between implicit compactness and explicit speed — the same accuracy-vs-efficiency trade-off that has governed detection (Chapter 5) and every other topic in this book.

## 14.5 Why 3D Belongs in a Perception Book

These techniques may look like a detour from detection, segmentation, and tracking, but they are increasingly the substrate those tasks run on. Autonomous driving and robotics perceive in **3D detection** and **occupancy** over LiDAR and multi-camera rigs — direct descendants of the point-set and volumetric ideas here. Reconstructed radiance fields provide geometry for **embodied agents** that must plan and act, not just label pixels. And the field is converging: promptable and foundation-model thinking (Chapters 8, 12, 13) is moving into 3D, aiming for models that reconstruct or segment 3D scenes as readily as SAM segments an image. Perception is stepping off the image plane, and the representations in this chapter are how it gets there.

## 14.6 Key Takeaways

*   **Representation is the crux of 3D.** Voxels, meshes, point clouds, and implicit fields each trade off memory, regularity, and renderability; the landmark methods are defined by which representation they embrace.
*   **PointNet respects the symmetry of sets.** A shared per-point MLP plus a symmetric aggregator (max) gives permutation invariance and is a universal approximator for set functions — but pooling only globally ignores local structure, which PointNet++ restored hierarchically.
*   **NeRF stores a scene in an MLP.** A continuous 5D radiance field, rendered by a differentiable volume-rendering integral and trained only on posed photos, achieves photorealistic novel views — with positional encoding and hierarchical sampling doing the heavy lifting — at the cost of slow rendering.
*   **3D Gaussian Splatting makes it real-time.** Returning to an explicit representation (a cloud of learned Gaussians) and GPU-friendly rasterization delivers NeRF-level quality at interactive speed, trading memory for latency.
*   **3D is becoming core perception.** Point-based, volumetric, and radiance-field ideas underpin 3D detection, occupancy, and embodied AI — where perception increasingly happens.

---
## References

1.  Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR.
2.  Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.* NeurIPS.
3.  Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.* ECCV.
4.  Barron, J. T., et al. (2021). *Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields.* ICCV.
5.  Müller, T., Evans, A., Schied, C., & Keller, A. (2022). *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (Instant-NGP).* ACM TOG.
6.  Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG, 42(4).
