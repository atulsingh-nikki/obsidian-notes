---
layout: post
title: "MathJax Test Page"
description: "Testing mathematical equation rendering on GitHub Pages"
tags: [test, mathjax]
---

# MathJax Test Page

## Inline Math Test
Here is an inline equation: $E = mc^2$ and another one $a^2 + b^2 = c^2$.

## Block Math Test
Here is a block equation:

$$\hat{x}_{k \mid k} = \hat{x}_{k \mid k-1} + K_k (z_k - H_k \hat{x}_{k \mid k-1})$$

## Conditional Probability Test
Bayes' theorem: $P(x \mid z) = \frac{P(z \mid x) P(x)}{P(z)}$

## Complex Kalman Filter Equations

**Prediction Step:**
$$\hat{x}_{k \mid k-1} = F_k \hat{x}_{k-1 \mid k-1} + B_k u_k$$

**Update Step:**
$$P_{k \mid k} = (I - K_k H_k) P_{k \mid k-1}$$

## Matrix Equation
$$\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix}$$

If you can see properly formatted mathematical equations above, MathJax is working correctly!
