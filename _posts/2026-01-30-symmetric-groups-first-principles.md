---
layout: post
title: "Symmetric Groups: Understanding Permutations from First Principles"
date: 2026-01-30
categories: [mathematics, abstract-algebra, group-theory]
tags: [symmetric-groups, permutations, group-theory, algebra, cycles, transpositions]
math: true
reading_time: "30 min read"
---

## Symmetric Groups: Understanding Permutations from First Principles

*30 min read*

The symmetric group is one of the most fundamental objects in mathematics—it's the group of all ways to rearrange a finite set. Despite its simple definition, symmetric groups appear everywhere: in solving polynomial equations, computing determinants, analyzing molecular symmetry, cryptography, and even understanding Rubik's cubes. This post builds the theory from scratch, assuming only basic set theory.

**Related Posts:**
- [Bijective Functions: The Perfect Correspondence]({{ site.baseurl }}{% post_url 2026-01-29-bijective-functions-invertibility %}) - Permutations are bijective functions from a set to itself
- [Matrix Determinants and Leibniz Theorem]({{ site.baseurl }}{% post_url 2026-01-27-matrix-determinants-leibniz-theorem %}) - The Leibniz formula sums over the symmetric group

---

## Table of Contents

1. [What is a Group?](#what-is-a-group)
2. [What is a Permutation?](#what-is-a-permutation)
3. [The Symmetric Group $S_n$](#the-symmetric-group-s_n)
4. [Cycle Notation](#cycle-notation)
5. [Composition and Inverses](#composition-and-inverses)
6. [Transpositions](#transpositions)
7. [Sign of a Permutation](#sign-of-a-permutation)
8. [Computing in Symmetric Groups](#computing-in-symmetric-groups)
9. [Why Symmetric Groups Matter](#why-symmetric-groups-matter)
10. [Interactive Examples](#interactive-examples)

---

## What is a Group?

Before we can understand symmetric groups, we need to know what a **group** is.

**Definition**: A **group** is a set $G$ equipped with a binary operation $\cdot: G \times G \to G$ satisfying:

### 1. Closure
For all $a, b \in G$:
$$
a \cdot b \in G
$$

### 2. Associativity
For all $a, b, c \in G$:
$$
(a \cdot b) \cdot c = a \cdot (b \cdot c)
$$

### 3. Identity Element
There exists $e \in G$ such that for all $a \in G$:
$$
e \cdot a = a \cdot e = a
$$

### 4. Inverse Element
For every $a \in G$, there exists $a^{-1} \in G$ such that:
$$
a \cdot a^{-1} = a^{-1} \cdot a = e
$$

**Intuition**: A group is a mathematical structure where you can "combine" elements (closure), order doesn't matter for grouping (associativity), there's a "do nothing" element (identity), and every action can be "undone" (inverse).

---

### Examples of Groups

**Example 1: Integers under Addition** ($\mathbb{Z}, +$)
- **Closure**: $3 + 5 = 8 \in \mathbb{Z}$
- **Associativity**: $(1 + 2) + 3 = 1 + (2 + 3) = 6$
- **Identity**: $0$ (since $n + 0 = n$)
- **Inverse**: The inverse of $n$ is $-n$ (since $n + (-n) = 0$)

**Example 2: Non-Zero Reals under Multiplication** ($\mathbb{R}^*, \times$)
- **Closure**: $2 \times 3 = 6 \in \mathbb{R}^*$
- **Associativity**: $(2 \times 3) \times 4 = 2 \times (3 \times 4) = 24$
- **Identity**: $1$ (since $x \times 1 = x$)
- **Inverse**: The inverse of $x$ is $\frac{1}{x}$ (since $x \times \frac{1}{x} = 1$)

**Example 3: Rotations of a Square**
- **Elements**: $\{0°, 90°, 180°, 270°\}$ rotations
- **Operation**: Composition of rotations
- **Identity**: $0°$ rotation (do nothing)
- **Inverses**: $90°$ and $270°$ are inverses; $180°$ is its own inverse

---

## What is a Permutation?

Now we get to the heart of symmetric groups.

**Definition**: A **permutation** of a set $X$ is a **bijective function** $\sigma: X \to X$.

In other words, a permutation is a way of rearranging the elements of $X$ such that:
- Every element is moved to exactly one position (function)
- Every position receives exactly one element (surjective)
- No two elements are sent to the same position (injective)

---

### Permutations as Rearrangements

**Example**: Let $X = \{1, 2, 3\}$. Here are all permutations:

**Identity** (do nothing):
$$
\sigma_1 = \begin{pmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \end{pmatrix} \quad \text{means} \quad 1 \to 1, \, 2 \to 2, \, 3 \to 3
$$

**Swap 1 and 2**:
$$
\sigma_2 = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 1 & 3 \end{pmatrix} \quad \text{means} \quad 1 \to 2, \, 2 \to 1, \, 3 \to 3
$$

**Cyclic shift** ($1 \to 2 \to 3 \to 1$):
$$
\sigma_3 = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 3 & 1 \end{pmatrix} \quad \text{means} \quad 1 \to 2, \, 2 \to 3, \, 3 \to 1
$$

And so on... there are $3! = 6$ total permutations.

---

### Two-Row Notation

The standard way to write a permutation $\sigma$ on $\{1, 2, \ldots, n\}$ is:

$$
\sigma = \begin{pmatrix} 1 & 2 & 3 & \cdots & n \\ \sigma(1) & \sigma(2) & \sigma(3) & \cdots & \sigma(n) \end{pmatrix}
$$

**Top row**: Original positions  
**Bottom row**: Where each element goes

**Example**: 
$$
\sigma = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 3 & 1 & 4 & 2 \end{pmatrix}
$$

This means: $1 \to 3$, $2 \to 1$, $3 \to 4$, $4 \to 2$

---

## The Symmetric Group $S_n$

**Definition**: The **symmetric group** $S_n$ is the set of all permutations of $\{1, 2, \ldots, n\}$ under the operation of **composition**.

$$
S_n = \{\sigma : \{1, \ldots, n\} \to \{1, \ldots, n\} \mid \sigma \text{ is bijective}\}
$$

**Size**: $|S_n| = n!$

Why? There are $n$ choices for where to send $1$, then $n-1$ choices for where to send $2$, and so on:

$$
n \times (n-1) \times (n-2) \times \cdots \times 1 = n!
$$

---

### $S_n$ is Indeed a Group

Let's verify the group axioms:

**1. Closure**: 
If $\sigma$ and $\tau$ are permutations (bijections), then $\sigma \circ \tau$ is also a bijection.
$$
\sigma, \tau \in S_n \implies \sigma \circ \tau \in S_n
$$

**2. Associativity**: 
Function composition is always associative:
$$
(\sigma \circ \tau) \circ \rho = \sigma \circ (\tau \circ \rho)
$$

**3. Identity**: 
The identity permutation $e$ (or $\text{id}$):
$$
e = \begin{pmatrix} 1 & 2 & \cdots & n \\ 1 & 2 & \cdots & n \end{pmatrix}
$$

Satisfies $e \circ \sigma = \sigma \circ e = \sigma$ for all $\sigma \in S_n$.

**4. Inverses**: 
Since permutations are bijections, each has an inverse function. If $\sigma(i) = j$, then $\sigma^{-1}(j) = i$.

$$
\sigma \circ \sigma^{-1} = \sigma^{-1} \circ \sigma = e
$$

---

### Small Symmetric Groups

**$S_1$**: Only one permutation (identity). Trivial group.
$$
|S_1| = 1! = 1
$$

**$S_2$**: Two permutations
$$
|S_2| = 2! = 2
$$

$$
e = \begin{pmatrix} 1 & 2 \\ 1 & 2 \end{pmatrix}, \quad \sigma = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}
$$

$S_2$ is isomorphic to $\mathbb{Z}_2$ (integers modulo 2).

**$S_3$**: Six permutations
$$
|S_3| = 3! = 6
$$

We'll explore $S_3$ in detail below—it's the smallest non-abelian group.

**$S_4$**: Twenty-four permutations
$$
|S_4| = 4! = 24
$$

Used in solving quartic polynomials, Rubik's cube mathematics, and more.

---

## Cycle Notation

Two-row notation is cumbersome for large $n$. **Cycle notation** is more compact and reveals structure.

### What is a Cycle?

A **cycle** is a permutation that moves elements in a circular fashion:

$$
(a_1 \, a_2 \, a_3 \, \ldots \, a_k)
$$

This means:
$$
a_1 \to a_2 \to a_3 \to \cdots \to a_k \to a_1
$$

All other elements are fixed (don't move).

**Example**: The cycle $(1 \, 3 \, 5)$ means:
$$
1 \to 3, \quad 3 \to 5, \quad 5 \to 1, \quad \text{and } 2 \to 2, \, 4 \to 4, \ldots
$$

In two-row notation (for $S_5$):
$$
(1 \, 3 \, 5) = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 3 & 2 & 5 & 4 & 1 \end{pmatrix}
$$

---

### Cycle Decomposition

**Theorem**: Every permutation can be written as a product of **disjoint cycles** (cycles that don't share any elements).

**Example 1**: 
$$
\sigma = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 3 & 5 & 1 & 4 & 2 \end{pmatrix}
$$

**Trace the cycles**:
- Start at $1$: $1 \to 3 \to 1$ (cycle $(1 \, 3)$)
- Start at $2$: $2 \to 5 \to 2$ (cycle $(2 \, 5)$)
- $4$ is fixed: $(4)$ (written or omitted)

So:
$$
\sigma = (1 \, 3)(2 \, 5)
$$

**Example 2**: 
$$
\tau = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 & 6 \\ 2 & 3 & 1 & 6 & 5 & 4 \end{pmatrix}
$$

**Trace the cycles**:
- Start at $1$: $1 \to 2 \to 3 \to 1$ (cycle $(1 \, 2 \, 3)$)
- Start at $4$: $4 \to 6 \to 4$ (cycle $(4 \, 6)$)
- $5$ is fixed

So:
$$
\tau = (1 \, 2 \, 3)(4 \, 6)
$$

**Key Property**: Disjoint cycles **commute** (order doesn't matter):
$$
(1 \, 3)(2 \, 5) = (2 \, 5)(1 \, 3)
$$

But non-disjoint cycles generally do NOT commute.

---

### Cycle Length

The **length** of a cycle $(a_1 \, a_2 \, \ldots \, a_k)$ is $k$.

**Terminology**:
- **1-cycle**: Fixed point (usually omitted)
- **2-cycle**: **Transposition** (swap)
- **3-cycle**: Triple exchange
- **$k$-cycle**: Permutes $k$ elements cyclically

**Order of a cycle**: The smallest positive integer $m$ such that applying the cycle $m$ times returns to the identity.

For a $k$-cycle: order = $k$

**Example**: $(1 \, 2 \, 3)$ has order 3
$$
(1 \, 2 \, 3) \circ (1 \, 2 \, 3) = (1 \, 3 \, 2)
$$
$$
(1 \, 2 \, 3) \circ (1 \, 2 \, 3) \circ (1 \, 2 \, 3) = e
$$

---

## Composition and Inverses

### Composing Permutations

**Convention**: We read compositions **right to left** (like function composition).

$$
(\sigma \circ \tau)(x) = \sigma(\tau(x))
$$

**Example**: Compute $(1 \, 2 \, 3) \circ (1 \, 2)$

**Method 1 (Right to left)**:
- First apply $(1 \, 2)$: $1 \to 2$
- Then apply $(1 \, 2 \, 3)$: $2 \to 3$
- Result: $1 \to 3$

Continue for all elements:
- $1 \to 2 \to 3$
- $2 \to 1 \to 2$
- $3 \to 3 \to 1$

So $(1 \, 2 \, 3) \circ (1 \, 2) = (1 \, 3)$

**Method 2 (Two-row notation)**:
$$
(1 \, 2) = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 1 & 3 \end{pmatrix}, \quad (1 \, 2 \, 3) = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 3 & 1 \end{pmatrix}
$$

Compose by following the path:
$$
\begin{pmatrix} 1 & 2 & 3 \\ 2 & 1 & 3 \end{pmatrix} \to \begin{pmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \end{pmatrix}
$$

Result: $(1 \, 3)$ ✓

---

### Computing Inverses

**For cycles**: Reverse the order!

$$
(a_1 \, a_2 \, a_3 \, \ldots \, a_k)^{-1} = (a_k \, a_{k-1} \, \ldots \, a_2 \, a_1)
$$

**Examples**:
$$
(1 \, 2 \, 3)^{-1} = (3 \, 2 \, 1) = (1 \, 3 \, 2)
$$

$$
(1 \, 2)^{-1} = (2 \, 1) = (1 \, 2) \quad \text{(transpositions are self-inverse)}
$$

**For products of cycles**: Reverse the order and invert each cycle:

$$
(\sigma_1 \circ \sigma_2 \circ \cdots \circ \sigma_k)^{-1} = \sigma_k^{-1} \circ \cdots \circ \sigma_2^{-1} \circ \sigma_1^{-1}
$$

**Example**:
$$
[(1 \, 2 \, 3)(4 \, 5)]^{-1} = (4 \, 5)^{-1} \circ (1 \, 2 \, 3)^{-1} = (4 \, 5)(1 \, 3 \, 2)
$$

---

## Transpositions

A **transposition** is a 2-cycle: a permutation that swaps exactly two elements.

$$
\tau = (i \, j)
$$

**Properties**:
1. Self-inverse: $(i \, j)^{-1} = (i \, j)$
2. Order 2: $(i \, j) \circ (i \, j) = e$
3. Commute if disjoint: $(1 \, 2)(3 \, 4) = (3 \, 4)(1 \, 2)$

---

### Every Permutation is a Product of Transpositions

**Theorem**: Every permutation can be written as a product (composition) of transpositions.

**Proof Idea**: First decompose into cycles, then decompose each cycle into transpositions.

**Key Identity**: A $k$-cycle can be written as:
$$
(a_1 \, a_2 \, a_3 \, \ldots \, a_k) = (a_1 \, a_k)(a_1 \, a_{k-1}) \cdots (a_1 \, a_3)(a_1 \, a_2)
$$

**Example**: $(1 \, 2 \, 3)$
$$
(1 \, 2 \, 3) = (1 \, 3)(1 \, 2)
$$

Verify:
- $1 \to 2 \to 2$ (via $(1,2)$, fixed by $(1,3)$) ✗

Wait, let me recalculate:
- First $(1 \, 2)$: $1 \to 2$
- Then $(1 \, 3)$: $2 \to 2$ (not affected), but $1 \to 3$

Actually, let's trace carefully:
- $1$: $(1,2)$ sends $1 \to 2$, then $(1,3)$ sends $2 \to 2$. So $1 \to 2$. ✗

The correct decomposition uses the formula:
$$
(1 \, 2 \, 3) = (1 \, 2)(2 \, 3)
$$

Verify:
- $1$: $(2,3)$ fixes $1$, $(1,2)$ sends $1 \to 2$. So $1 \to 2$. ✓
- $2$: $(2,3)$ sends $2 \to 3$, $(1,2)$ fixes $3$. So $2 \to 3$. ✓
- $3$: $(2,3)$ sends $3 \to 2$, $(1,2)$ sends $2 \to 1$. So $3 \to 1$. ✓

**General formula** (adjacent transpositions):
$$
(a_1 \, a_2 \, \ldots \, a_k) = (a_1 \, a_2)(a_2 \, a_3) \cdots (a_{k-1} \, a_k)
$$

---

### Non-Uniqueness

**Important**: The decomposition into transpositions is **not unique**.

**Example**:
$$
(1 \, 2 \, 3) = (1 \, 2)(2 \, 3) = (1 \, 3)(1 \, 2) = (2 \, 3)(1 \, 3)
$$

However, the **parity** (even or odd number of transpositions) is always the same!

---

## Sign of a Permutation

### Even and Odd Permutations

**Definition**: A permutation is **even** if it can be written as a product of an even number of transpositions, and **odd** if it requires an odd number.

**Theorem**: The parity is well-defined—a permutation cannot be both even and odd.

**Sign function**:
$$
\text{sgn}(\sigma) = \begin{cases} +1 & \text{if } \sigma \text{ is even} \\ -1 & \text{if } \sigma \text{ is odd} \end{cases}
$$

---

### Computing the Sign

**Method 1: Count transpositions**

Decompose $\sigma$ into transpositions and count them.

**Example**: $(1 \, 2 \, 3) = (1 \, 2)(2 \, 3)$ (2 transpositions)
$$
\text{sgn}((1 \, 2 \, 3)) = (-1)^2 = +1 \quad \text{(even)}
$$

**Example**: $(1 \, 2) = (1 \, 2)$ (1 transposition)
$$
\text{sgn}((1 \, 2)) = (-1)^1 = -1 \quad \text{(odd)}
$$

---

**Method 2: Cycle structure**

A $k$-cycle has sign:
$$
\text{sgn}((a_1 \, \ldots \, a_k)) = (-1)^{k-1}
$$

**Why?** A $k$-cycle decomposes into $k-1$ transpositions.

**For disjoint cycles**:
$$
\text{sgn}(\sigma_1 \circ \sigma_2 \circ \cdots) = \text{sgn}(\sigma_1) \cdot \text{sgn}(\sigma_2) \cdots
$$

**Example**: $(1 \, 2 \, 3)(4 \, 5 \, 6 \, 7)$
$$
\text{sgn} = (-1)^{3-1} \cdot (-1)^{4-1} = (+1) \cdot (-1) = -1 \quad \text{(odd)}
$$

---

**Method 3: Inversion count**

An **inversion** in $\sigma$ is a pair $(i, j)$ with $i < j$ but $\sigma(i) > \sigma(j)$.

$$
\text{sgn}(\sigma) = (-1)^{\text{number of inversions}}
$$

**Example**: $\sigma = \begin{pmatrix} 1 & 2 & 3 \\ 3 & 1 & 2 \end{pmatrix}$

Inversions:
- $(1, 2)$: $\sigma(1) = 3 > \sigma(2) = 1$ ✓
- $(1, 3)$: $\sigma(1) = 3 > \sigma(3) = 2$ ✓
- $(2, 3)$: $\sigma(2) = 1 < \sigma(3) = 2$ ✗

Total: 2 inversions
$$
\text{sgn}(\sigma) = (-1)^2 = +1 \quad \text{(even)}
$$

---

### The Alternating Group $A_n$

The set of all **even permutations** forms a subgroup of $S_n$ called the **alternating group** $A_n$.

$$
A_n = \{\sigma \in S_n \mid \text{sgn}(\sigma) = +1\}
$$

**Size**: $|A_n| = \frac{n!}{2}$

**Why?** Exactly half of all permutations are even.

**Example**: $|A_3| = \frac{3!}{2} = 3$

The three even permutations in $S_3$ are:
- $e$ (identity)
- $(1 \, 2 \, 3)$
- $(1 \, 3 \, 2)$

---

## Computing in Symmetric Groups

### Example: $S_3$ Multiplication Table

$S_3$ has 6 elements:
$$
\begin{align}
e &= \text{identity} \\
\sigma &= (1 \, 2) \\
\tau &= (1 \, 3) \\
\rho &= (2 \, 3) \\
\alpha &= (1 \, 2 \, 3) \\
\beta &= (1 \, 3 \, 2)
\end{align}
$$

**Partial multiplication table**:

|     | $e$ | $\sigma$ | $\alpha$ | $\beta$ |
|-----|-----|----------|----------|---------|
| $e$ | $e$ | $\sigma$ | $\alpha$ | $\beta$ |
| $\sigma$ | $\sigma$ | $e$ | $\beta$ | $\alpha$ |
| $\alpha$ | $\alpha$ | $\rho$ | $\beta$ | $e$ |
| $\beta$ | $\beta$ | $\tau$ | $e$ | $\alpha$ |

**Note**: $S_3$ is **non-abelian** (not commutative):
$$
\sigma \circ \alpha = (1 \, 2) \circ (1 \, 2 \, 3) = (2 \, 3) = \rho
$$

$$
\alpha \circ \sigma = (1 \, 2 \, 3) \circ (1 \, 2) = (1 \, 3) = \tau
$$

So $\sigma \circ \alpha \neq \alpha \circ \sigma$.

---

### Computing Cycle Orders

**Order of a permutation**: The smallest $m > 0$ such that $\sigma^m = e$.

**For a single cycle**: $\text{order}((a_1 \, \ldots \, a_k)) = k$

**For disjoint cycles**: $\text{order} = \text{lcm}(\text{cycle lengths})$

**Example**: $(1 \, 2 \, 3)(4 \, 5)$
$$
\text{order} = \text{lcm}(3, 2) = 6
$$

Check:
$$
[(1 \, 2 \, 3)(4 \, 5)]^6 = (1 \, 2 \, 3)^6 \circ (4 \, 5)^6 = e \circ e = e \quad ✓
$$

---

## Why Symmetric Groups Matter

### 1. Determinants and the Leibniz Formula

The determinant of an $n \times n$ matrix is defined as a sum over $S_n$:

$$
\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)}
$$

**Example** ($2 \times 2$):

$S_2 = \{e, (1 \, 2)\}$

$$
\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = \text{sgn}(e) \cdot a \cdot d + \text{sgn}((1 \, 2)) \cdot b \cdot c = ad - bc
$$

**Connection**: The sign of the permutation determines whether a term is added or subtracted!

---

### 2. Polynomial Roots and Galois Theory

The **Galois group** of a polynomial is (roughly) the group of permutations of its roots that preserve algebraic relations.

**Example**: For $x^2 - 2 = 0$, the roots are $\pm \sqrt{2}$. Swapping them is a permutation in $S_2$.

**Abel-Ruffini Theorem**: There's no general formula for roots of degree $\geq 5$ polynomials because $S_n$ for $n \geq 5$ is "too complicated" (not solvable).

---

### 3. Molecular Symmetry (Chemistry)

Molecules have symmetry groups that are often subgroups of $S_n$.

**Example**: Methane ($\text{CH}_4$) has tetrahedral symmetry, a subgroup of $S_4$ acting on the 4 hydrogen atoms.

---

### 4. Cryptography

**Permutation ciphers** rearrange plaintext characters according to a permutation in $S_n$.

**Example**: Caesar cipher is a cyclic permutation of the alphabet.

Modern block ciphers (AES, DES) use complex combinations of permutations and substitutions.

---

### 5. Combinatorics

Counting problems often involve counting permutations or orbits under group actions.

**Example**: How many ways to arrange 5 distinct books on a shelf? Answer: $|S_5| = 5! = 120$.

---

### 6. Rubik's Cube

The Rubik's cube group is a subgroup of $S_{48}$ (permutations of 48 movable pieces), with order approximately $4.3 \times 10^{19}$.

---

## Interactive Examples

### Example 1: All Permutations of $S_3$

List in cycle notation:

$$
\begin{align}
e &= \text{(identity)} \\
(1 \, 2) &= \text{swap 1 and 2} \\
(1 \, 3) &= \text{swap 1 and 3} \\
(2 \, 3) &= \text{swap 2 and 3} \\
(1 \, 2 \, 3) &= \text{rotate: } 1 \to 2 \to 3 \to 1 \\
(1 \, 3 \, 2) &= \text{rotate: } 1 \to 3 \to 2 \to 1
\end{align}
$$

**Signs**:
- $e$: even (0 transpositions)
- $(1 \, 2)$, $(1 \, 3)$, $(2 \, 3)$: odd (1 transposition each)
- $(1 \, 2 \, 3)$, $(1 \, 3 \, 2)$: even (2 transpositions each)

So $A_3 = \{e, (1 \, 2 \, 3), (1 \, 3 \, 2)\}$.

---

### Example 2: Composition Practice

Compute $(1 \, 3 \, 5)(2 \, 4) \circ (1 \, 2)(3 \, 4 \, 5)$.

**Step 1**: Apply $(1 \, 2)(3 \, 4 \, 5)$ first (right to left).

Trace each element:
- $1$: $(3,4,5)$ fixes $1$, $(1,2)$ sends $1 \to 2$. Result: $1 \to 2$
- $2$: $(3,4,5)$ fixes $2$, $(1,2)$ sends $2 \to 1$. Result: $2 \to 1$
- $3$: $(3,4,5)$ sends $3 \to 4$, $(1,2)$ fixes $4$. Result: $3 \to 4$
- $4$: $(3,4,5)$ sends $4 \to 5$, $(1,2)$ fixes $5$. Result: $4 \to 5$
- $5$: $(3,4,5)$ sends $5 \to 3$, $(1,2)$ fixes $3$. Result: $5 \to 3$

So $(1 \, 2)(3 \, 4 \, 5) = (1 \, 2)(3 \, 4 \, 5)$ (already in cycle form).

**Step 2**: Apply $(1 \, 3 \, 5)(2 \, 4)$ to this result.

Actually, let's use the shortcut: trace through both compositions at once.

- $1$: $(3,4,5)$ fixes → $(1,2)$ sends to $2$ → $(2,4)$ sends to $4$ → $(1,3,5)$ fixes. Result: $1 \to 4$
- $2$: $(3,4,5)$ fixes → $(1,2)$ sends to $1$ → $(2,4)$ fixes → $(1,3,5)$ sends to $3$. Result: $2 \to 3$
- $3$: $(3,4,5)$ sends to $4$ → $(1,2)$ fixes → $(2,4)$ sends to $2$ → $(1,3,5)$ fixes. Result: $3 \to 2$
- $4$: $(3,4,5)$ sends to $5$ → $(1,2)$ fixes → $(2,4)$ fixes → $(1,3,5)$ sends to $1$. Result: $4 \to 1$
- $5$: $(3,4,5)$ sends to $3$ → $(1,2)$ fixes → $(2,4)$ fixes → $(1,3,5)$ sends to $5$. Result: $5 \to 5$

So the result is:
$$
\begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 4 & 3 & 2 & 1 & 5 \end{pmatrix} = (1 \, 4)(2 \, 3)
$$

---

### Example 3: Finding Inverses

Find the inverse of $(1 \, 2 \, 3 \, 4)(5 \, 6 \, 7)$.

**Method**: Reverse each cycle:
$$
[(1 \, 2 \, 3 \, 4)(5 \, 6 \, 7)]^{-1} = (1 \, 4 \, 3 \, 2)(5 \, 7 \, 6)
$$

**Verify**:
$$
(1 \, 2 \, 3 \, 4)(5 \, 6 \, 7) \circ (1 \, 4 \, 3 \, 2)(5 \, 7 \, 6)
$$

Trace $1$: $(5,7,6)$ fixes → $(1,4,3,2)$ sends to $4$ → $(5,6,7)$ fixes → $(1,2,3,4)$ sends back to $1$. ✓

(All elements return to themselves—it's the identity.)

---

### Example 4: Sign Calculation

Find $\text{sgn}((1 \, 5 \, 3)(2 \, 7)(4 \, 6 \, 8 \, 9))$.

**Method**: Use cycle lengths.
- $(1 \, 5 \, 3)$: length 3, sign $(-1)^{3-1} = +1$
- $(2 \, 7)$: length 2, sign $(-1)^{2-1} = -1$
- $(4 \, 6 \, 8 \, 9)$: length 4, sign $(-1)^{4-1} = -1$

**Total sign**:
$$
\text{sgn} = (+1) \cdot (-1) \cdot (-1) = +1 \quad \text{(even permutation)}
$$

---

## Summary

### Key Takeaways

1. **Permutations** are bijective functions from a set to itself—ways of rearranging elements.

2. **Symmetric group** $S_n$ is the group of all permutations of $n$ elements under composition.
   - Size: $|S_n| = n!$
   - Non-abelian for $n \geq 3$

3. **Cycle notation** provides a compact way to express permutations.
   - Every permutation decomposes into disjoint cycles
   - Cycle structure reveals important properties

4. **Transpositions** (2-cycles) generate all permutations.
   - Every permutation is a product of transpositions
   - Parity (even/odd) is well-defined

5. **Sign function** $\text{sgn}(\sigma) = \pm 1$ depends on parity.
   - Even permutations form the alternating group $A_n$
   - Critical for determinant formulas

6. **Applications** span mathematics, physics, chemistry, computer science:
   - Determinants (Leibniz formula)
   - Galois theory (solving polynomials)
   - Molecular symmetry
   - Cryptography
   - Combinatorics

---

### Conceptual Hierarchy

```
Bijective Functions (invertible rearrangements)
         ↓
Permutations (bijections from set to itself)
         ↓
Symmetric Group S_n (all permutations with composition)
         ↓
Structure: Cycles, Transpositions, Sign
         ↓
Applications: Determinants, Galois Theory, Symmetry, ...
```

---

## Further Reading

**Foundational Texts**:
- Dummit & Foote. *Abstract Algebra* (3rd ed., 2004) - Comprehensive treatment of group theory
- Artin, M. *Algebra* (2nd ed., 2011) - Elegant introduction with geometric intuition
- Rotman, J. *An Introduction to the Theory of Groups* (1995) - Group theory deep dive

**Specialized Topics**:
- Sagan, B. *The Symmetric Group* (2001) - Dedicated to $S_n$ and its representations
- James & Liebeck. *Representations and Characters of Groups* (2001) - Representation theory
- Knuth, D. *The Art of Computer Programming, Vol. 3* (1998) - Permutations in algorithms

**Applications**:
- Stewart, I. *Galois Theory* (4th ed., 2015) - Galois groups and polynomial solvability
- Cotton, F. *Chemical Applications of Group Theory* (3rd ed., 1990) - Molecular symmetry

---

**Final Thought**:

Symmetric groups are the "atoms" of group theory—every finite group embeds into some $S_n$ (Cayley's theorem). Understanding how permutations compose, how they decompose into cycles, and how their signs behave unlocks deep connections across mathematics. From the simple act of rearranging a deck of cards to the profound Abel-Ruffini theorem about polynomial unsolvability, $S_n$ is everywhere. 🎴

---

*If you found this helpful, check out the related posts on bijective functions and matrix determinants to see symmetric groups in action!*
