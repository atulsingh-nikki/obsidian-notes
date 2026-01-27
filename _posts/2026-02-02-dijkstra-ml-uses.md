---
layout: post
title: "Dijkstra’s Algorithm and Where Machine Learning Uses It"
description: "From shortest-path intuition to practical ML pipelines that rely on Dijkstra-style searches on graphs."
tags: [graphs, algorithms, machine-learning, optimization]
---

Dijkstra’s algorithm is the workhorse for finding **shortest paths in weighted graphs with non‑negative edges**. It’s taught in algorithms courses, but it also shows up quietly inside machine-learning (ML) pipelines whenever we need to propagate information across a graph efficiently and consistently.

This post refreshes the core idea of Dijkstra and then shows where the same logic appears in ML workflows.

## 1) The core idea in one paragraph

Given a start node, Dijkstra’s algorithm repeatedly expands the **closest unexplored node** and relaxes its neighbors. A priority queue stores the current best distances; once a node is popped from the queue, its distance is final (for non‑negative weights). This greedy property is what makes Dijkstra fast and reliable for shortest-path tasks.

## 2) Why it works (intuition)

The algorithm relies on this invariant:

- If all edge weights are non‑negative, the next node with the smallest tentative distance **cannot be improved by any other path** that goes through a node with a larger tentative distance.

That’s why the “best so far” node is safe to finalize, and the algorithm doesn’t need to backtrack.

## 3) Complexity & practical notes

- With a binary heap, the time complexity is **O((V + E) log V)**.
- With a Fibonacci heap, it can be **O(E + V log V)**, though the constants are often worse in practice.
- Dijkstra **fails with negative edge weights**; use Bellman–Ford or Johnson’s algorithm instead.

## 4) Where ML uses Dijkstra (directly or conceptually)

### A) Graph-based semi‑supervised learning

Label propagation and graph regularization often require **shortest paths or geodesic distances** on a graph of data points (e.g., k‑NN graphs). Dijkstra is a common engine for computing those distances efficiently when edge weights are non‑negative.

**Example:** You build a k‑NN graph for embeddings. Distances along the graph approximate manifold distance better than Euclidean distance in the ambient space. Dijkstra gives you those manifold distances.

### B) Prototype selection and clustering variants

Some clustering or prototype-selection pipelines compute **shortest-path distances** in a graph to decide which exemplars cover the space. When the metric is path-based (e.g., in density‑based clustering variants), Dijkstra provides the core distance computation.

### C) Spatial ML and planning models

In robotics or RL, learned cost maps are often converted into **graphs** where edges represent feasible moves with non‑negative costs. Dijkstra can be used to derive optimal or heuristic paths for planning, data collection, or policy evaluation.

**Example:** A vision model predicts a cost map. Dijkstra finds a minimal‑cost route for a robot, which then feeds into trajectory data for imitation or reinforcement learning.

### D) Neural networks on graphs (GNNs) and precomputation

Many GNN pipelines precompute **shortest-path distance features** (e.g., for positional encodings). Dijkstra is the standard approach when the graph is weighted and non‑negative.

### E) Approximate nearest neighbor (ANN) graph searches

Some ANN methods use graph‑traversal strategies over proximity graphs. While these are not always pure Dijkstra, they often mirror its **priority‑based expansion** logic to explore promising neighbors first.

## 5) A minimal pseudocode sketch

```text
Input: graph G(V, E), source s
Initialize dist[v] = ∞ for all v, dist[s] = 0
Initialize priority queue Q with (0, s)

while Q not empty:
    (d, u) = Q.pop_min()
    if d > dist[u]: continue
    for each edge (u, v) with weight w:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            Q.push(dist[v], v)

Output: dist[]
```

## 6) When to reach for Dijkstra in ML

Use Dijkstra when you have:

- A graph structure (explicit or implicit).
- Non‑negative edge costs.
- A need for **exact shortest‑path** or **geodesic** distances.

If you need **approximate** distances at scale, consider faster heuristics (e.g., A*, pruning, or ANN graph traversal), but Dijkstra is still the conceptual baseline.

## 7) Summary

Dijkstra’s algorithm is more than an algorithms‑class exercise. It provides a clean, efficient way to propagate costs across a graph, and that pattern shows up in ML whenever we represent data as nodes with weighted edges. From label propagation to robotic navigation and GNN positional encodings, Dijkstra’s greedy expansion is a reliable building block for ML systems.
