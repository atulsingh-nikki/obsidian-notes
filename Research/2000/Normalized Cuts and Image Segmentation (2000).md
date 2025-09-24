---
title: "Normalized Cuts and Image Segmentation (2000)"
aliases:
  - Ncut
  - Normalized CutsNormalized Cuts and Image Segmentation (2000)
  - Shi & Malik 2000
authors:
  - Jianbo Shi
  - Jitendra Malik
year: 2000
venue: "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)"
doi: "10.1109/34.868688"
citations: 30,000+
tags:
  - paper
  - image-segmentation
  - graph-cuts
  - spectral-clustering
  - computer-vision
fields:
  - vision
  - clustering
  - segmentation
  - graph-theory
related:
  - "[[Graph-Based Image Segmentation (Felzenszwalb & Huttenlocher, 2004)]]"
  - "[[Spectral Clustering in Vision (Ng, Jordan, Weiss, 2002)]]"
predecessors:
  - graph partitioning methods (min-cut, ratio cut)
successors:
  - spectral clustering methods in CV
  - graph-based segmentation pipelines
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**Normalized Cuts (Shi & Malik, 2000)** reframed image segmentation as a **graph partitioning problem**. Pixels (or regions) are nodes; edges encode similarity (color, intensity, texture, spatial proximity). Instead of minimizing raw cut cost (which biases toward small isolated sets), the paper introduced the **Normalized Cut (Ncut) criterion**, balancing cut cost with the association within each segment.

# Key Idea
> Partition a graph such that inter-cluster dissimilarity is minimized **relative to intra-cluster similarity**.  
$$
Normalized Cut (Ncut) =  
Ncut(A,B) = \frac{cut(A,B)}{assoc(A,V)} + \frac{cut(A,B)}{assoc(B,V)}
$$  
where $cut(A,B)$ = weight of edges between A and B, and $assoc(A,V)$ = total connection from A to all nodes.

# Method
- Represent image as weighted undirected graph.  
- Edge weights encode similarity in intensity, color, texture, position.  
- Partition objective = minimize Ncut.  
- Leads to solving a **generalized eigenvalue problem** for the graph Laplacian.  
- Use eigenvectors to find optimal bipartition; recursively split for multi-segment.  

# Results
- Produced perceptually meaningful segmentations on natural images.  
- Outperformed min-cut or ratio-cut which favored small clusters.  
- Demonstrated applicability to texture + motion segmentation.  

# Why it Mattered
- Brought **spectral graph theory** into computer vision.  
- First strong formalization of segmentation as **balanced partitioning**.  
- Inspired **spectral clustering**, **graph-based segmentation**, and modern deep clustering approaches.  

# Architectural Pattern
- Graph representation of image.  
- Laplacian eigenvector computation → segmentation cues.  
- Recursive bipartitioning.  

# Connections
- Related to spectral clustering (Ng et al. 2002).  
- Predecessor to graph-based image segmentation (Felzenszwalb & Huttenlocher 2004).  
- Conceptual ancestor of deep graph segmentation / GNNs in vision.  

# Implementation Notes
- Requires solving large eigenvalue problems → computationally heavy.  
- Approximations (e.g. multiscale, sparse graphs) used in practice.  
- Sensitive to choice of similarity function & parameters.  

# Critiques / Limitations
- Computationally expensive for large images.  
- No semantic understanding — purely low-level grouping.  
- Recursive splitting may propagate early errors.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Graph theory basics: nodes, edges, weights.  
- Min-cut vs normalized cut.  
- Segmentation = grouping pixels into regions.  

## Postgraduate-Level Concepts
- Spectral graph theory: eigenvectors of Laplacian.  
- Generalized eigenvalue problems in clustering.  
- Relationship to normalized Laplacian in spectral clustering.  
- Extensions: multi-way cuts, approximation algorithms.  

---

# My Notes
- **Ncut = balance**: avoids trivial small segments.  
- Heavy computation, but conceptually elegant.  
- Still taught as a cornerstone of segmentation & clustering.  
- Later simplified/accelerated by Felzenszwalb-Huttenlocher (2004).  

---
