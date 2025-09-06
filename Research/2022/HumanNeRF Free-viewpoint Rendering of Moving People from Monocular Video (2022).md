---
title: "HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video (2022)"
aliases:
  - HumanNeRF
  - Monocular Human NeRF
authors:
  - Ziyan Weng
  - Sida Peng
  - Zhen Xu
  - Yuanqing Zhang
  - Qianqian Wang
  - Hujun Bao
  - Xiaowei Zhou
year: 2022
venue: "CVPR (Oral)"
doi: "10.1109/CVPR52688.2022.00502"
arxiv: "https://arxiv.org/abs/2201.04127"
code: "https://github.com/zju3dv/HumanNeRF"
citations: 700+
dataset:
  - Monocular human performance video (in-the-wild)
  - ZJU-MoCap dataset
tags:
  - paper
  - nerf
  - human-performance
  - monocular
  - free-viewpoint
fields:
  - vision
  - graphics
  - 3d-human
related:
  - "[[Animatable NeRF (2021)]]"
  - "[[Neural Body (2021)]]"
  - "[[Avatar Methods (2022+)]]"
predecessors:
  - "[[Neural Body (2021)]]"
successors:
  - "[[Avatar-friendly NeRFs (2022+)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**HumanNeRF** addressed the challenge of **free-viewpoint rendering of moving humans** from **monocular videos**, extending prior NeRF-based methods that required multi-view setups. It designed a pose-driven canonical representation that generalizes across identities and poses, enabling novel view synthesis of humans from single-camera captures.

# Key Idea
> Learn a **pose-conditioned canonical NeRF** from monocular video, enabling generalization across time and poses while synthesizing novel views of humans.

# Method
- **Canonical representation**: Human represented in a canonical pose space.  
- **Pose conditioning**: Uses SMPL parametric model for pose alignment.  
- **NeRF backbone**: Canonical NeRF deformed into posed space per frame.  
- **Regularization**: Temporal consistency constraints across frames.  
- **Input**: Only monocular RGB video sequences (no multi-view).  

# Results
- Enabled high-quality **free-viewpoint rendering** from monocular human videos.  
- Outperformed Animatable NeRF and Neural Body in monocular settings.  
- Generated temporally consistent novel views for dynamic human performances.  

# Why it Mattered
- Brought NeRF-based human modeling closer to **consumer-level monocular capture**.  
- Pushed animatable NeRFs toward generalization across **identities and poses**.  
- Widely adopted in avatar research and digital human modeling.  

# Architectural Pattern
- Pose-conditioned canonical NeRF.  
- Deformation field guided by SMPL skeleton.  
- Temporal consistency regularization.  

# Connections
- Successor to **Animatable NeRF (2021)** and **Neural Body (2021)**.  
- Predecessor to avatar-focused NeRFs and real-time human rendering approaches.  
- Related to monocular human capture and avatar synthesis.  

# Implementation Notes
- Trains from monocular videos, unlike prior multi-view datasets.  
- Needs SMPL pose extraction for alignment.  
- Code and pretrained models released (ZJU).  

# Critiques / Limitations
- Struggles with extreme occlusion or fast motion blur.  
- Requires accurate SMPL pose estimation.  
- Still slower than real-time requirements.  

---

# Educational Connections

## Undergraduate-Level Concepts
- Why monocular human capture is harder than multi-view.  
- Canonical vs posed space for human modeling.  
- Basics of free-viewpoint rendering.  
- Role of skeletons (SMPL) in providing pose priors.  

## Postgraduate-Level Concepts
- Deformation fields mapping canonical NeRF to posed space.  
- Temporal consistency losses for monocular dynamic NeRFs.  
- Comparison of Neural Body vs HumanNeRF for generalization.  
- Implications for consumer-level avatar systems.  

---

# My Notes
- HumanNeRF is the **key milestone**: NeRF humans from just monocular video.  
- Moves the field from studio captures toward **in-the-wild usability**.  
- Open question: How to make this **real-time** for avatars and VR?  
- Possible extension: Merge HumanNeRF pose-driven design with **Gaussian Splatting** for fast, animatable human avatars.  

---
