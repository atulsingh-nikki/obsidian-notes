---
title: "A* Sampling (2014)"
aliases:
  - A-Star Sampling
  - Gumbel Process Sampling
authors:
  - Chris J. Maddison
  - Daniel Tarlow
  - Tom Minka
year: 2014
venue: "NeurIPS 2014"
doi: ""     # not provided
arxiv: "https://arxiv.org/abs/1411.0030"
status: "to-read"
---

## Summary
A\* Sampling reframes sampling from continuous distributions as an **optimization problem** over a **Gumbel process** using **A\*** search. It offers **exact, independent samples** while making more intelligent use of bound and likelihood computations compared to adaptive rejection methods. It’s both correct and efficient under typical conditions. :contentReference[oaicite:2]{index=2}

## Key Idea
> Extend the discrete **Gumbel-Max trick** into continuous spaces using a **Gumbel process**, then use **A\*** search guided by bounds to find the argmax—which is then a valid sample from the target distribution. :contentReference[oaicite:3]{index=3}

## Method
1. **Gumbel Process Construction**  
   Generalizes Gumbel-Max to continuous domains: draws from a stochastic process so that maxima and argmax behave correctly. :contentReference[oaicite:4]{index=4}

2. **Top-down Sampling**  
   Rather than sampling infinitely many perturbations, the algorithm induces the relevant ones first via a hierarchical, region-based process. :contentReference[oaicite:5]{index=5}

3. **A\* Search over Regions**  
   Regions are scored with bounds (`G + M(region)`); once a region's lower bound exceeds all other upper bounds, that region’s sample is the exact draw. :contentReference[oaicite:6]{index=6}

4. **Log-density Factorization**  
   Breaks down the target log-density into a tractable part plus a bounded correction, enabling efficient sampling while maintaining exactness. :contentReference[oaicite:7]{index=7}

5. **Variants**  
   - Reuse bounds across samples  
   - Balance bound vs likelihood evaluations  
   - Special-case simplification when the correction term is unimodal in 1D :contentReference[oaicite:8]{index=8}

## Results
- **Exact, independent samples** — no chains or approximations.  
- **More efficient than adaptive rejection sampling**, avoiding wasted computations.  
- Demonstrated correctness and efficiency empirically. :contentReference[oaicite:9]{index=9}

## Why It Matters
It brings together **probabilistic inference** and **search-based optimization** in a novel way—exact sampling, smart search, no burn-in. This method was recognized with an **Outstanding Paper Award at NeurIPS 2014**. :contentReference[oaicite:10]{index=10}

## Connections
- Builds on the **Gumbel-Max trick** from discrete domains.  
- Defines a contrast to MCMC/rejection: no approximations, no envelopes, just structured search.  
- Lays groundwork for modern perturb-and-optimize sampling strategies.

## Educational Connections

**Undergraduate Students**
- Explore links between **extreme-value theory** (Gumbel distribution), **sampling methods**, and **search algorithms**.
- Learn how continuous sampling can be reframed as **search over regions**—mixing probabilistic reasoning with algorithm design.

**Graduate Students**
- Understand how to construct a **continuous Gumbel process** and why its max/argmax properties support exact sampling.
- Dive into designing **A\* search heuristics** using upper/lower likelihood bounds—a bridge between inference and optimization.
- Explore how **density decomposition** enables efficient exact sampling, and consider potential generalizations (e.g., high-dimensional spaces).

## Resources
- **NeurIPS 2014 paper**: concise overview of A* Sampling. :contentReference[oaicite:11]{index=11}  
- **arXiv preprint**: full public version. :contentReference[oaicite:12]{index=12}  
- **CS Toronto PDF**: deeper theoretical and algorithmic exposition. :contentReference[oaicite:13]{index=13}  
- **Citation listing/confirmation**: recognition details. :contentReference[oaicite:14]{index=14}

