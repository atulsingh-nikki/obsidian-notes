---
title: Rich feature hierarchies for accurate object detection and semantic segmentation (2014)
aliases:
  - R-CNN
  - Girshick et al. 2014
  - R-CNN (2014)
authors:
  - Ross Girshick
  - Jeff Donahue
  - Trevor Darrell
  - Jitendra Malik
year: 2014
venue: CVPR
doi: 10.48550/arXiv.1311.2524
arxiv: https://arxiv.org/abs/1311.2524
code: https://github.com/rbgirshick/rcnn
citations: 70,000+
dataset:
  - PASCAL VOC
  - ILSVRC2013
tags:
  - paper
  - deep-learning
  - computer-vision
  - object-detection
  - semantic-segmentation
fields:
  - vision
  - object-detection
  - segmentation
related:
  - "[[Fast R-CNN (2015)|Fast R-CNN]]"
  - "[[Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks|Faster R-CNN]]"
predecessors:
  - "[[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]]"
successors:
  - "[[Fast R-CNN (2015)]]"
  - "[[Faster R-CNN (2015)]]"
impact: ⭐⭐⭐⭐⭐
status: read
---

# Summary
This paper introduces **R-CNN (Regions with CNN features)**, the first deep learning–based method to achieve significant improvements in object detection accuracy. It combines region proposals with CNN feature extraction and class-specific linear SVMs. It also extends to semantic segmentation.

# Key Idea
> Use deep convolutional networks to extract features from bottom-up region proposals, drastically improving object detection accuracy.

# Method
- Generate ~2000 region proposals with Selective Search.  
- Warp each proposal to a fixed size and feed into a CNN (pre-trained on ImageNet, fine-tuned for detection).  
- Extract features (fc7 layer) and classify using class-specific SVMs.  
- Apply bounding box regression to refine localization.  
- For segmentation, regions are classified and merged.

# Results
- Achieved **mean Average Precision (mAP)** of 53.7% on PASCAL VOC 2010 (previous SOTA ~35%).  
- Strong improvement on ILSVRC2013 detection challenge.  
- Demonstrated that fine-tuning ImageNet-pretrained CNNs outperforms traditional hand-crafted features (HOG, DPM).

# Why it Mattered
- Marked the shift from hand-crafted features to deep CNN-based features for detection.  
- Sparked the R-CNN family (Fast R-CNN, Faster R-CNN, Mask R-CNN).  
- Established the paradigm of transfer learning from ImageNet to detection tasks.

# Architectural Pattern
- **Region-based detection pipeline**: region proposal → CNN feature extraction → classification/regression.  
- Pre-trained ImageNet CNN as backbone.  
- Modular design (separate proposal, feature extraction, classification).  

# Connections
- **Contemporaries**: OverFeat (2013), Deformable Part Models (DPM, 2008–2014).  
- **Influence**: Inspired entire family of two-stage detectors; foundation for semantic segmentation (Mask R-CNN).

# Implementation Notes
- CNN: [[AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|AlexNet (2012)]] pre-trained on ImageNet.  
- Region warping to fixed input size was computationally heavy (thousands of CNN passes per image).  
- Fine-tuning crucial for performance.  
- Linear SVMs used instead of softmax classifier.  

# Critiques / Limitations
- Extremely **slow** at inference (47s per image on GPU).  
- Relies on external region proposals (Selective Search).  
- Large disk storage for features (hundreds of GB).  
- Superseded by Fast/Faster R-CNN and later single-stage detectors (YOLO, SSD).  

# Repro / Resources
- Paper: https://arxiv.org/abs/1311.2524  
- Official code: https://github.com/rbgirshick/rcnn  
- Datasets: PASCAL VOC, ILSVRC2013  
- Tutorials: Stanford CS231n and PyTorch re-implementations.  

---

# Educational Connections

## Undergraduate-Level Concepts
- **Linear Algebra**: convolution operations, matrix multiplications in CNNs.  
- **Probability & Statistics**: classification probabilities, evaluation metrics (mAP).  
- **Calculus**: backpropagation for fine-tuning.  
- **Signals & Systems**: filtering intuition via convolutional kernels.  
- **Data Structures**: tensors, region proposals.  
- **Optimization Basics**: stochastic gradient descent.  

## Postgraduate-Level Concepts
- **Transfer Learning**: fine-tuning ImageNet CNN for detection.  
- **Advanced Optimization**: bounding box regression training.  
- **ML Theory**: why deep hierarchical features generalize better than hand-crafted ones.  
- **Computer Vision**: detection vs segmentation pipelines.  
- **Neural Network Design**: adapting classification CNNs to detection.  
- **Research Methodology**: ablations on fine-tuning, pretraining vs from-scratch.  

---

# My Notes
- **Connection to my projects**: Foundation for object selection and segmentation workflows (relevant for Smart Masking, Object Removal, Upscale).  
- **Open questions**: What happens if region proposals are learned end-to-end? → Answered by Faster R-CNN.  
- **Possible extensions**: Replace external proposal generator with RPN (done later); unify classifier and regressor into single network.
- SIFT and HOG are blockwise orientation histograms,
- Previously Localized was framed as Regression Problem.  However , [[Deep Neural Networks for Object Detection (2013)]] showed that Regression may not fare well. 
