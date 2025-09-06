---
title: "SPLATNet: Sparse Lattice Networks for Point Cloud Processing (2018)"
aliases: 
  - SPLATNet
  - Sparse Lattice Networks
authors:
  - Hang Su
  - Varun Jampani
  - Deqing Sun
  - Subhransu Maji
  - Evangelos Kalogerakis
  - Ming-Hsuan Yang
  - Jan Kautz
year: 2018
venue: "CVPR"
doi: "10.1109/CVPR.2018.00253"
arxiv: "https://arxiv.org/abs/1802.08275"
code: "https://github.com/NVlabs/splatnet"
citations: 600+
dataset:
  - ShapeNet
  - RueMonge2014 (3D urban dataset)
  - Stanford 3D Indoor Dataset
tags:
  - paper
  - point-clouds
  - 3d
  - lattice-networks
fields:
  - vision
  - graphics
  - robotics
related:
  - "[[PointNet (2017)]]"
  - "[[PointNet++ (2017)]]"
predecessors:
  - "[[Bilateral Convolution Layers (BCL)]]"
successors:
  - "[[KPConv (2019)]]"
  - "[[SparseConvNet]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

# Summary
SPLATNet introduced **Sparse Lattice Networks** for efficient **point cloud processing**, leveraging **Bilateral Convolutional Layers (BCLs)** to project unordered points into high-dimensional lattices. This enabled effective **point cloud segmentation and classification** with scalability advantages.

# Key Idea
> Use **sparse lattice structures** and **bilateral convolutions** to process point clouds efficiently, preserving geometric structure while being computationally tractable.

# Method
- Projects point clouds into a **sparse permutohedral lattice**.  
- Performs **bilateral convolution** (convolution in a high-dimensional feature lattice).  
- Allows joint reasoning over **3D point clouds and 2D images** by projecting both into the lattice.  
- Architectures: SPLATNet3D (point clouds only) and SPLATNet2D-3D (joint 2D–3D).  

# Results
- Achieved competitive results on **ShapeNet part segmentation** and **RueMonge urban reconstruction**.  
- Outperformed PointNet on some tasks, especially in leveraging 2D + 3D data.  
- Demonstrated efficiency for large-scale point cloud processing.  

# Why it Mattered
- Pioneered **sparse lattice-based deep learning** for point clouds.  
- Provided a unified framework for combining 2D images and 3D point clouds.  
- Inspired later work on sparse convolutions and lattice-based representations.  

# Architectural Pattern
- Input: point cloud (and optionally 2D image).  
- Projection into sparse lattice.  
- Bilateral convolution for feature extraction.  
- Task-specific decoder for segmentation/classification.  

# Connections
- **Contemporaries**: PointNet++, voxel CNNs.  
- **Influence**: KPConv (kernel point convolution), sparse convolution frameworks.  

# Implementation Notes
- Lattice projection adds overhead, but convolution efficient in sparse domain.  
- Works well when fusing 2D + 3D modalities.  
- Requires careful tuning of lattice resolution.  

# Critiques / Limitations
- More complex than PointNet-style approaches.  
- Performance gains limited for small-scale tasks.  
- Replaced by more direct sparse convolution approaches (SparseConvNet, MinkowskiNet).  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1802.08275)  
- [Official code (PyTorch, Caffe2)](https://github.com/NVlabs/splatnet)  
- [Related Bilateral Convolution Layer repo](https://github.com/NVlabs/bilateralNN)  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: Sparse tensor representations.  
- **Probability & Statistics**: High-dimensional filtering.  
- **Geometry**: Point cloud structures, projections.  
- **Optimization Basics**: Training segmentation/classification networks.  

## Postgraduate-Level Concepts
- **Neural Network Design**: Bilateral convolutions, sparse lattices.  
- **Computer Vision**: Joint 2D–3D reasoning.  
- **Research Methodology**: Benchmarks in segmentation and reconstruction.  
- **Advanced Optimization**: Handling sparsity efficiently in GPU pipelines.  

---

# My Notes
- Relevant for **video editing with 3D scene understanding**.  
- Open question: Can lattice-based processing scale to **temporal 3D point sequences**?  
- Possible extension: Combine SPLATNet with **transformers on sparse lattices** for multimodal 2D+3D editing.  

---
