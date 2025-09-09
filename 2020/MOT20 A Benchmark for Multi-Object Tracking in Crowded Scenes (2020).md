---
title: "MOT20: A Benchmark for Multi-Object Tracking in Crowded Scenes (2020)"
aliases:
  - MOT20
  - MOT20 Dataset
authors:
  - Patrick Dendorfer
  - Hamid Rezatofighi
  - Anton Milan
  - Laura Leal-Taixé
  - Ian Reid
  - Stefan Roth
  - Konrad Schindler
  - et al.
year: 2020
venue: "arXiv preprint / ECCV Workshops"
doi: "10.48550/arXiv.2003.09003"
arxiv: "https://arxiv.org/abs/2003.09003"
code: "https://motchallenge.net/data/MOT20/"
citations: 700+
dataset:
  - MOT20 (benchmark dataset)
tags:
  - dataset
  - tracking
  - multi-object-tracking
  - mot
  - crowded-scenes
fields:
  - vision
  - tracking
  - datasets
related:
  - "[[MOT17 Dataset]]"
  - "[[DanceTrack (2022)]]"
  - "[[ByteTrack (2022)]]"
  - "[[OC-SORT (2023)]]"
  - "[[BoT-SORT (2023)]]"
predecessors:
  - "[[MOT17 Dataset]]"
successors:
  - "[[DanceTrack (2022)]]"
impact: ⭐⭐⭐⭐⭐
status: "read"

---

# Summary
**MOT20** is a dataset and benchmark for **multi-object tracking (MOT)**, designed to evaluate trackers in **extremely dense pedestrian crowds**. It extended the popular **MOTChallenge** series (MOT16/17), pushing MOT research toward handling **severe occlusions and overlapping trajectories**.

# Key Idea
> Provide a dataset where pedestrian density is **much higher than MOT16/17**, exposing the limitations of trackers that work in sparser urban environments.

# Dataset Details
- **Content**: 8 sequences captured in European cities.  
- **Annotations**: 2D bounding boxes with consistent identities.  
- **Scale**: >2 million boxes annotated across thousands of frames.  
- **Density**: Up to **246 pedestrians per frame**, far denser than MOT17.  
- **Challenges**:  
  - Long occlusions.  
  - Heavy overlaps.  
  - Small-scale pedestrians in cluttered scenes.  

# Results
- Many trackers that performed well on MOT17 failed badly on MOT20.  
- Sparked methods like **ByteTrack** and **BoT-SORT** to improve occlusion handling.  
- Still a standard benchmark alongside MOT17 and DanceTrack.  

# Why it Mattered
- First **extreme density MOT dataset**, showing real-world limits.  
- Essential for evaluating trackers in **crowded public spaces** (stations, malls, events).  
- Complemented MOT17 (sparser) and paved the way for DanceTrack.  

# Connections
- Direct successor to **MOT17 Dataset**.  
- Complementary to **DanceTrack (2022)** (crowded but dynamic motion).  
- Strongly influenced the **SORT/DeepSORT → ByteTrack/OC-SORT/BoT-SORT** trajectory.  

# Critiques / Limitations
- Limited diversity (only pedestrian crowds, European cities).  
- Still purely **2D bounding-box based**.  
- Small number of sequences compared to modern large-scale datasets.  

---

# Educational Connections

## Undergraduate-Level Concepts
- What dataset benchmarks are and why they matter.  
- Why dense scenes are harder for tracking than sparse ones.  
- Examples: crowded streets vs suburban sidewalks.  

## Postgraduate-Level Concepts
- Impact of dataset density on MOT algorithm design.  
- Failure cases: why Kalman filter-based trackers fail in MOT20.  
- Implications for deploying MOT in real-world dense environments (e.g., transport hubs).  
- How MOT20 + DanceTrack shaped **occlusion-focused tracking research**.  

---

# My Notes
- MOT20 = **the stress test for MOT**: dense, cluttered, brutal for trackers.  
- Showed that high MOT17 scores ≠ robustness in real-world crowds.  
- Open question: Should future MOT datasets emphasize **density, dynamics, or diversity**?  
- Possible extension: A **3D MOT20-style dataset with LiDAR + video** for autonomous navigation in crowds.  

---
