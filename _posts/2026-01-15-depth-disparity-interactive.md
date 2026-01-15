---
layout: post
title: "Interactive Depth vs Disparity Visualization"
description: "An interactive visualization exploring the inverse relationship between depth and disparity in stereo vision, demonstrating how baseline, focal length, and disparity affect 3D depth estimation."
tags: [computer-vision, stereo-vision, depth-estimation, interactive, visualization, 3d-geometry]
---

*This interactive tool lets you explore the fundamental relationship between depth and disparity in stereo vision systems. Experiment with camera parameters to build intuition about how stereo cameras perceive 3D structure.*

## The Depth-Disparity Relationship

In stereo vision, depth ($Z$) is inversely related to disparity ($d$) through the equation:

$$
Z = \frac{b \cdot f}{d}
$$

where:
- $Z$ = depth (distance to the scene point)
- $b$ = baseline (distance between camera centers)
- $f$ = focal length (in pixels)
- $d$ = disparity (pixel difference between left and right images)

This seemingly simple equation has profound implications for stereo camera design and depth estimation accuracy.

## Interactive Visualization

<div id="depth-disparity-viz"></div>
<script src="{{ site.baseurl }}/assets/js/depth-disparity-interactive.js"></script>

## How to Use This Visualization

### Camera Parameters (Left Panel)

1. **Baseline (b)**: Adjust from 0.05m to 1.0m
   - Controls the distance between the stereo cameras
   - Larger baseline → larger disparity → better depth resolution at distance
   - Trade-off: larger baseline → harder to match nearby objects

2. **Focal Length (f)**: Adjust from 300 to 1500 pixels
   - Simulates different camera lenses (wide-angle vs telephoto)
   - Larger focal length → larger disparity → better depth precision
   - Trade-off: narrower field of view

3. **Disparity (d)**: Adjust from 1 to 200 pixels
   - The pixel offset between corresponding points in left/right images
   - Small disparity → far objects
   - Large disparity → near objects

### Visualizations (Right Panel)

1. **3D Stereo Camera Geometry**
   - Isometric view showing both cameras and the 3D point
   - Cameras are positioned at the baseline distance
   - Purple rectangles represent image planes
   - Red dot shows the 3D point in space
   - Blue lines show sight lines from cameras to the point

2. **Stereo Camera Setup (Top View)**
   - Bird's-eye view of the stereo rig
   - Shows baseline, cameras, and 3D point position
   - Illustrates the triangulation geometry

3. **Depth vs Disparity Graph**
   - Plots the hyperbolic relationship
   - Red dot shows current depth/disparity values
   - Demonstrates why depth uncertainty grows with distance

### Key Insights Box

The insights update dynamically based on your parameter choices, helping you understand:
- The inverse relationship between depth and disparity
- Depth uncertainty factor ($\sigma_Z / Z^2$)
- Practical implications for camera design

## Experiments to Try

### 1. Understanding Depth Uncertainty

**Try this:**
- Set disparity to 100 pixels (close object)
- Note the computed depth
- Now set disparity to 10 pixels (far object)
- Compare the depth values

**What you'll learn:** Depth grows rapidly (hyperbolically) as disparity decreases. This means distant objects have much higher depth uncertainty than nearby objects.

### 2. Effect of Baseline

**Try this:**
- Set disparity to 20 pixels
- Adjust baseline from 0.05m to 1.0m
- Watch how depth changes

**What you'll learn:** Larger baseline increases disparity for the same depth, improving depth resolution. This is why some autonomous vehicles use wide-baseline stereo (>1m between cameras).

### 3. Focal Length Trade-offs

**Try this:**
- Set baseline to 0.12m and disparity to 50 pixels
- Vary focal length from 300 to 1500 pixels
- Observe depth changes

**What you'll learn:** Longer focal length (telephoto) increases disparity, improving depth accuracy. But it also narrows the field of view, limiting spatial coverage.

### 4. The Near-Far Problem

**Try this:**
- Try to achieve a depth of 10m by adjusting only disparity
- Note the required disparity value

**What you'll learn:** Very distant objects have tiny disparities (< 5 pixels), making them hard to measure accurately. This is why LiDAR is often used for long-range sensing in autonomous vehicles.

## Mathematical Deep Dive

### Depth Uncertainty

The relationship $Z = \frac{bf}{d}$ means that depth uncertainty ($\sigma_Z$) grows quadratically with distance:

$$
\sigma_Z = \frac{bf}{d^2} \sigma_d \propto Z^2
$$

where $\sigma_d$ is disparity measurement uncertainty (typically ±0.5 to ±1 pixel).

**Implication:** If you can measure depth to ±10cm at 1m, at 10m the uncertainty grows to ±10m (100x worse!).

### Practical Design Rules

1. **For close-range robotics (0.5-5m):**
   - Baseline: 10-20cm
   - Focal length: 500-800 pixels
   - Expected disparity: 20-200 pixels

2. **For autonomous driving (5-50m):**
   - Baseline: 30-120cm
   - Focal length: 1000-2000 pixels
   - Expected disparity: 5-100 pixels

3. **For wide baseline stereo (structure from motion):**
   - Baseline: meters to kilometers
   - Requires sophisticated matching algorithms
   - Can achieve millimeter precision at large scales

## Why the Inverse Relationship Matters

The inverse relationship $Z \propto 1/d$ has several important consequences:

1. **Non-uniform depth resolution:** Close objects are measured more accurately than distant ones

2. **Disparity search range:** You need to search more disparity values for close objects than far ones

3. **Computational cost:** Most stereo algorithms process all disparity values equally, wasting computation on unlikely distant matches

4. **Sensor fusion:** Often better to combine stereo (good at near range) with LiDAR (good at far range)

## Real-World Systems

### Smartphone Depth (Portrait Mode)
- Baseline: ~2-4cm (limited by phone width)
- Focal length: ~700-1000 pixels
- Effective range: 0.3-3m
- Challenge: Small baseline limits far-range accuracy

### Autonomous Vehicle Stereo
- Baseline: 30-120cm
- Focal length: 1500-2500 pixels
- Effective range: 2-80m
- Multiple camera pairs at different baselines for different ranges

### Industrial 3D Scanning
- Baseline: 10-100cm (adjustable)
- Focal length: 1000-4000 pixels
- Effective range: 0.1-10m
- Often combined with structured light for textureless surfaces

### Space Applications
- Baseline: meters (separate spacecraft)
- Focal length: 5000+ pixels (telephoto)
- Effective range: kilometers
- Used for asteroid mapping, planetary rovers

## Connection to Other Depth Sensing Methods

### Structured Light
- Projects patterns instead of relying on texture
- Still uses triangulation (similar to stereo)
- Disparity measured from pattern deformation
- Better for textureless surfaces

### Time-of-Flight (ToF)
- Measures time directly instead of disparity
- No baseline needed (simpler hardware)
- Lower resolution than cameras
- Complements stereo well

### LiDAR
- Active scanning with laser
- Very accurate at long range (>50m)
- Sparse measurements (expensive to get dense)
- Often fused with stereo cameras

## Next Steps

After exploring this visualization, you might want to learn more about:

1. **[Stereo Matching Algorithms]({{ site.baseurl }}/2026/01/15/depth-maps-computer-vision.html#stereo-matching-algorithms)** - How do we actually compute disparity from image pairs?

2. **[Epipolar Geometry]({{ site.baseurl }}/2026/01/15/depth-maps-computer-vision.html#epipolar-geometry)** - The mathematical framework behind stereo vision

3. **[Deep Learning for Depth]({{ site.baseurl }}/2026/01/15/depth-maps-computer-vision.html#deep-learning-for-depth-estimation)** - How neural networks learn to predict depth

4. **[Full Depth Maps Guide]({{ site.baseurl }}/2026/01/15/depth-maps-computer-vision.html)** - Comprehensive overview of depth estimation in computer vision

## Technical Details

This visualization uses HTML5 Canvas with JavaScript for real-time rendering. The 3D geometry view uses a simple isometric projection to visualize the stereo camera setup in 3D space. All computations are performed client-side in your browser.

---

*Have questions or suggestions for improving this visualization? The inverse relationship between depth and disparity is one of the most fundamental concepts in 3D computer vision—understanding it intuitively makes everything else click into place.*
