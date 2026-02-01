---
layout: post
title: "From Linear Equations to Vector Spaces"
description: "A high-signal tour from solving Ax=b to the abstractions that define vector spaces, subspaces, and determinants."
tags: [linear-algebra, math, study-notes]
---

Linear algebra did not start as an abstract theory. It started as a practical question: **when does a system of equations have a solution, and how many are there?** From that concrete problem, the field discovered the structure that eventually became the idea of a *vector space*.

This post is a tight, exam-ready map of that journey.

## 1. The beginning: solving linear systems

Everything starts with the system

\[
Ax = b
\]

People wanted to know:
- Does a solution exist?
- Is it unique?
- If not, how many solutions are there?

Those questions pushed mathematicians to look beyond individual solutions and study the **set** of all solutions.

## 2. Homogeneous systems reveal subspaces

Consider the homogeneous system:

\[
Ax = 0
\]

The solution set has two crucial properties:
- **Closed under addition**
- **Closed under scalar multiplication**

That means the solution set is a **subspace** of \(\mathbb{R}^n\).

This was the first sign that entire collections of solutions had structure worth studying on their own.

## 3. Non-homogeneous systems are shifted subspaces

For the non-homogeneous system:

\[
Ax = b
\]

If \(x_0\) is one particular solution, then **every** solution looks like:

\[
x = x_0 + \ker(A)
\]

So the solution set is a **shifted subspace**: a line, plane, or hyperplane that doesn’t necessarily pass through the origin.

## 4. The abstraction leap: vector spaces

Mathematicians noticed that the same algebraic behavior appears in many different settings, not just \(\mathbb{R}^n\). They abstracted the common structure:

- Addition
- Scalar multiplication
- Linear combinations

They stopped caring what the objects *were* and focused on **how they behaved**.

That abstraction is what we now call a **vector space**.

## 5. Why it’s called a “space”

In math, a *space* is a set with structure that supports geometry:
- Directions
- Dimension
- Linear motion
- Bases

Vector spaces let us talk about those geometric ideas even when the elements are **polynomials, functions, signals, or matrices**.

## 6. Polynomials are vectors too

Polynomials form a vector space because:
- You can add them
- You can scale them
- The result is still a polynomial

Each polynomial corresponds to a coefficient vector, and a standard basis is:

\[
\{1, x, x^2, \dots\}
\]

This space is **infinite-dimensional**, because you need infinitely many basis elements to span it.

*Note:* Polynomial **multiplication** is extra structure (an algebra), not required for a vector space.

## 7. Subspaces clarified

A **subspace** is:
- A subset of a vector space
- That is itself a vector space

The subspace test:
1. Contains the zero vector
2. Closed under addition
3. Closed under scalar multiplication

Key relationship:
- ✅ Every subspace is a vector space
- ❌ Not every vector space is a subspace (a subspace needs a parent space)

## 8. Linear independence and determinants

For **square matrices only**:

\[
\det(A) \neq 0
\quad\Longleftrightarrow\quad
\text{columns (or rows) are linearly independent}
\]

Geometric meaning:
- Determinant = volume
- Zero volume ⇒ dependence

Determinant **does not apply** to non-square matrices.

## 9. Common traps to avoid

- “Multiplication” in vector spaces means *scalar* multiplication, not vector–vector multiplication
- The zero vector space breaks statements that assume “nonempty”
- Infinite-dimensional vector spaces exist (polynomials are a prime example)
- Complements exist but are **not unique**
- Determinants are only for **square** matrices

## 10. Mental models that stick

- **Vector space** → playground of directions
- **Subspace** → smaller playground *through the origin*
- **Basis** → minimal generators
- **Dimension** → degrees of freedom
- **Polynomials** → coefficient vectors
- **Determinant** → volume test (square only)

---

### One-sentence takeaway

**Linear algebra grew from studying solution sets of linear equations and abstracting the structure that makes addition, scaling, and geometry possible everywhere.**
