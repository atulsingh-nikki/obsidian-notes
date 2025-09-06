# Chapter 12: New Frontiers in Tracking

## 12.1 Introduction: Beyond Tracking-by-Detection

The tracking-by-detection paradigm, while powerful and popular, has inherent limitations. By design, it separates the "what" (detection) from the "who" (association), which is not always optimal. This separation prevents the tracker from influencing the detector and makes the system vulnerable to detection failures.

In this chapter, we explore the new frontiers of tracking that seek to break down this separation, leading to more unified, efficient, and powerful models. We will look at two major paradigm shifts: joint detection and tracking, and the truly end-to-end approach of Transformer-based models.

## 12.2 Joint Detection and Tracking: CenterTrack

The 2020 paper "Tracking Objects as Points" by Zhou et al. presented **CenterTrack**, an elegant and effective method for joint detection and tracking [1]. It builds directly on the simple "objects as points" philosophy of the CenterNet detector.

The core insight of CenterTrack is to use the previous frame as a hint. The network takes both the current and previous frames as input and, in addition to detecting objects in the current frame, it also predicts a **tracking offset**—the 2D displacement of each object from its position in the previous frame. This allows a simple, greedy matching algorithm to link objects across frames, effectively learning a motion model implicitly and removing the need for a separate Kalman Filter and Hungarian Algorithm.

## 12.3 End-to-End Tracking with Transformers

The final and most recent paradigm shift in tracking seeks to eliminate all hand-designed components, creating a truly **end-to-end** system. To achieve this, researchers turned to the powerful **Transformer architecture**. The core idea of these new models is to treat tracking not as a frame-by-frame matching problem, but as a **set prediction problem over time**.

### 12.3.1 TrackFormer: Tracking with Persistent Queries
One of the most influential early works in this area was **TrackFormer** [2]. It builds directly upon the DETR object detector. The core insight is brilliantly simple: if DETR uses "object queries" to find objects in an image, a tracker can be built by allowing these queries to **persist across frames**. A query becomes the representation of the track itself. A query that finds a person in frame 1 is passed to frame 2 to find that same person. The self-attention mechanism in the Transformer decoder handles the data association implicitly.

### 12.3.2 MOTRv3: Scalable End-to-End Tracking
More recent models like **MOTRv3** have further refined this paradigm [3]. They have focused on improving the scalability and training of these Transformer-based trackers, further pushing their performance and demonstrating that the end-to-end, query-based approach is a powerful and general framework for state-of-the-art multi-object tracking.

## 12.4 Key Takeaways
-   **Joint Tracking (CenterTrack):** Unifies detection and tracking by using the previous frame as a hint and learning a **tracking offset**. This removes the need for classical components like the Kalman Filter [1].
-   **End-to-End Transformer Tracking:** The newest paradigm, which treats tracking as a set prediction problem over time. It eliminates all hand-designed components.
-   **Persistent Queries (TrackFormer):** The core idea is to have object queries from a DETR-like detector persist across frames, becoming the representation of the object's track. Association is handled implicitly by self-attention [2].
-   **State-of-the-Art:** Models like MOTRv3 demonstrate that the Transformer-based, end-to-end paradigm is the current state-of-the-art for multi-object tracking [3].

---
## References
1. Zhou, X., Wang, D., & Krähenbühl, P. (2020). Tracking objects as points. In *European conference on computer vision*.
2. Meinhardt, T., Kirillov, A., Leal-Taixé, L., & Feichtenhofer, C. (2022). TrackFormer: Multi-Object Tracking with Transformers. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
3. Yu, F., Li, Y., Wang, T., & Wang, Y. (2024). MOTRv3: A Scalable End-to-End Transformer-based Multi-Object Tracker. *arXiv preprint arXiv:2402.04323*.
