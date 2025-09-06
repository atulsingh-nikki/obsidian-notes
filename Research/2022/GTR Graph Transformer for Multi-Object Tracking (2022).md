---
title: "GTR: Graph Transformer for Multi-Object Tracking (2022)"
aliases:
  - GTR
  - Graph Transformer MOT
authors:
  - Yunbo Zhang
  - Jiayi Liu
  - Siyuan Qiao
  - Alan Yuille
  - Wenbo Li
  - et al.
year: 2022
venue: "arXiv preprint"
doi: "10.48550/arXiv.2207.06863"
arxiv: "https://arxiv.org/abs/2207.06863"
code: "https://github.com/MCG-NJU/GTR"
citations: 150+
dataset:
  - MOT17
  - MOT20
  - DanceTrack
tags:
  - paper
  - tracking
  - multi-object-tracking
  - mot
  - transformers
  - graph-networks
fields:
  - vision
  - tracking
  - autonomous-driving
related:
  - "[[MOTR (2022)]]"
  - "[[MeMOT (2022)]]"
  - "[[TransTrack (2021)]]"
predecessors:
  - "[[Graph Neural Networks for MOT (2019+)]]"
  - "[[Transformer-based MOT (TransTrack 2021, MOTR 2022)]]"
successors:
  - "[[Next-Gen Hybrid MOT (2023+)]]"
impact: ⭐⭐⭐⭐☆
status: "read"

---

# Summary
**GTR** introduced a **graph-transformer hybrid architecture** for multi-object tracking (MOT). It modeled **detections and tracks as graph nodes** and used transformers to reason about both **spatial relationships (within-frame)** and **temporal relationships (across frames)**, combining the strengths of graph neural networks (GNNs) and transformers.

# Key Idea
> Treat detections and tracks as nodes in a **spatio-temporal graph**, and apply a **graph–transformer hybrid** to learn associations jointly across space and time.

# Method
- **Graph construction**:  
  - Nodes = detections/tracks.  
  - Edges = potential associations (spatial + temporal).  
- **Graph transformer**:  
  - Transformer layers reason about node/edge features.  
  - Attention encodes relationships between detections and tracks.  
- **Output**: Object detections linked into tracklets.  
- **Training**: Supervised with ground-truth trajectories.  

# Results
- Competitive with MOTR, MeMOT on MOT17, MOT20, DanceTrack.  
- More robust in crowded scenes due to relational reasoning.  
- Achieved strong identity consistency and reduced ID switches.  

# Why it Mattered
- Unified **graph reasoning + transformer attention** in MOT.  
- Provided a structured way to model relationships between multiple objects.  
- Advanced understanding of **spatio-temporal association learning**.  

# Architectural Pattern
- Spatio-temporal graph representation.  
- Graph + transformer hybrid network.  

# Connections
- Related to **graph neural networks (GNNs)** in tracking (2018–2019).  
- Extended **transformer MOT (TransTrack, MOTR, MeMOT)** with explicit relational modeling.  
- Inspired later hybrid architectures for spatio-temporal tracking.  

# Implementation Notes
- More complex pipeline than MOTR/ByteTrack.  
- Requires careful graph construction.  
- PyTorch implementation open-sourced.  

# Critiques / Limitations
- Computationally heavy compared to ByteTrack.  
- Graph construction step may limit scalability.  
- Not yet real-time.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What a graph is (nodes and edges).  
- How objects (detections) can be modeled as graph nodes.  
- Transformers as attention mechanisms.  
- Applications: linking objects in video sequences.  

## Postgraduate-Level Concepts
- Graph neural networks (GNNs) and their use in MOT.  
- Combining GNNs with transformers for spatio-temporal reasoning.  
- Trade-offs: explicit graph modeling vs implicit transformer associations.  
- Potential for multimodal tracking (camera + LiDAR graphs).  

---

# My Notes
- GTR is a **clever marriage of GNNs and transformers** for MOT.  
- Explicitly modeling graphs = more structured than pure query persistence.  
- Open question: Will future MOT be **graph-structured or purely transformer-based**?  
- Possible extension: Apply GTR-style hybrid to **3D MOT with LiDAR point clouds**.  

---
