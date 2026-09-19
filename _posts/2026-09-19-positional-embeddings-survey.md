---
title: "Positional Embeddings Across the Literature: Sinusoidal, Learned, Relative, RoPE, ALiBi and Beyond"
date: 2026-09-19
tags: [transformers, vit, positional-embeddings, survey]
summary: "Quick, practical overview of positional embeddings and a pointer to a full technical survey in the Research folder."
---

TL;DR

- Positional embeddings tell transformers where tokens live; the main trade-off is between flexibility and generalization.
- For ViTs, a learned 2D positional grid is simple and effective when image size is fixed.
- For long-context models, RoPE and ALiBi are often better choices because they generalize more naturally beyond training lengths.

Why this matters

Transformers are permutation-invariant: they see a set of tokens, not an ordered sequence. Positional information is therefore not optional; it is how the model learns ordering, locality, and distance. The question is not whether to add positions, but which encoding matches the task and how much extrapolation you need.

Quick takeaways

- Use learned absolute positions for a fixed ViT input size; they are simple and work very well in-distribution.
- Use relative or distance-aware schemes when you care about how far apart tokens are, not just where they are.
- Use RoPE or ALiBi when long-context generalization matters, especially in language models.

Read the full technical survey

The detailed treatment with formulas, references, and implementation notes is here: [Research/2026-09-19-positional-embeddings-survey]({{ site.baseurl }}/Research/2026-09-19-positional-embeddings-survey.html).

Where the deeper material lives

This short note is the overview. The longer research document includes the worked numeric example, the positional-grid shape calculation for ViT, and the RoPE/ALiBi notes and snippets that are useful for implementation work.
