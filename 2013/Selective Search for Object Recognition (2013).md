---
title: Selective Search for Object Recognition (2013)
aliases:
  - Selective Search
  - Region Proposals Selective Search
authors:
  - Jasper R. R. Uijlings
  - Koen E. A. van de Sande
  - Theo Gevers
  - Arnold W. M. Smeulders
year: 2013
venue: IJCV
doi: 10.1007/s11263-013-0620-5
arxiv: https://arxiv.org/abs/1306.1189
code: https://github.com/sergeyk/selective_search
citations: 20,000+
dataset:
  - PASCAL VOC 2007/2010/2012
tags:
  - paper
  - object-detection
  - region-proposals
  - segmentation
fields:
  - vision
  - detection
related:
  - "[[Recognition Using Regions (2009)]]"
  - "[[R-CNN (2014)]]"
predecessors:
  - "[[Recognition Using Regions (2009)]]"
successors:
  - "[[EdgeBoxes (2014)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
Selective Search introduced a **segmentation-based algorithm for generating region proposals**. Instead of sliding windows, it produced a small set of candidate object regions (~2k per image) that significantly improved efficiency for object detection pipelines. It became the **de facto standard proposal method** used in R-CNN and subsequent detectors until replaced by learned proposal networks.

# Key Idea
> Use **hierarchical grouping of superpixels** to propose object candidate regions efficiently, balancing recall (cover all objects) with fewer proposals than exhaustive search.

# Method
- **Initial segmentation**: Over-segment the image into superpixels using Felzenszwalb-Huttenlocher algorithm.  
- **Region merging**: Iteratively merge similar regions using cues:  
  - Color similarity.  
  - Texture similarity.  
  - Size compatibility.  
  - Fill (how well regions cover an object).  
- **Multi-strategy**: Run selective search with multiple parameter settings to capture objects of different sizes and shapes.  
- Produces ~2000 region proposals per image.  

# Results
- Achieved **high object recall with relatively few proposals**.  
- Enabled practical region-based object detectors (R-CNN).  
- Outperformed sliding-window approaches in efficiency and coverage.  

# Why it Mattered
- Became the **standard proposal algorithm** pre-deep-learning detectors used.  
- Made region-based CNN detection feasible (R-CNN → Fast/Faster R-CNN).  
- Shifted object detection from sliding-window CNNs to **proposal + classification pipelines**.  

# Architectural Pattern
- Bottom-up segmentation → hierarchical grouping → proposal set.  

# Connections
- **Contemporaries**: DPM (HOG-based detectors), OverFeat (sliding window CNN).  
- **Influence**: R-CNN (2014), Fast R-CNN, Faster R-CNN (RPN replaced proposals).  

# Implementation Notes
- Typically produces ~2k proposals/image.  
- Designed to maximize recall (near 100%) at the expense of precision.  
- Not learning-based; purely hand-engineered heuristic pipeline.  

# Critiques / Limitations
- Computationally expensive (seconds per image).  
- Many redundant proposals.  
- Later replaced by learned Region Proposal Networks (RPNs) in Faster R-CNN.  

# Repro / Resources
- [Paper link](https://arxiv.org/abs/1306.1189)  
- [Original code (MATLAB)](https://github.com/sergeyk/selective_search)  
- Widely reimplemented in OpenCV, Python.  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Geometry**: Segmentation and region grouping.  
- **Probability & Statistics**: Similarity metrics between regions.  
- **Algorithms**: Greedy hierarchical merging.  

## Postgraduate-Level Concepts
- **Computer Vision**: Object proposals vs sliding windows.  
- **Research Methodology**: Balancing recall/efficiency.  
- **Advanced Optimization**: Proposal pruning and merging strategies.  

---

# My Notes
- Critical bridge: made **R-CNN possible** by giving CNNs a manageable set of regions.  
- Illustrates how **clever pre-deep heuristics** can bootstrap learning.  
- Open question: can **diffusion-based region proposals** outperform both heuristic and RPN methods for structured object discovery?  
- Possible extension: Use selective-search-like region priors for **guided video editing with object masks**.  

---
