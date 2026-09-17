---
title: "Designing and Interpreting Probes with Control Tasks (2019)"
aliases:
  - Hewitt & Liang 2019
  - Probing Control Tasks
  - Selectivity in Probing Classifiers
authors:
  - John Hewitt
  - Percy Liang
year: 2019
venue: "EMNLP"
doi: "10.18653/v1/D19-1275"
arxiv: "https://arxiv.org/abs/1909.03368"
code: "https://github.com/john-hewitt/control-tasks"
citations: ~600+
dataset:
  - Penn Treebank (POS tagging)
  - CCG supertagging corpus
tags:
  - paper
  - interpretability
  - probing
  - linear-probe
  - representation-learning
  - nlp
fields:
  - nlp
  - interpretability
  - representation-learning
related:
  - "[[Understanding intermediate layers using linear classifier probes (Alain & Bengio, 2016)]]"
  - "[[What Is a Linear Probe]]"
predecessors:
  - "[[Understanding intermediate layers using linear classifier probes (Alain & Bengio, 2016)]]"
successors:
  - "[[Information-Theoretic Probing with Minimum Description Length (Voita & Titov, 2020)]]"
impact: ⭐⭐⭐⭐☆
status: "read"
---

### Summary

This paper asks an uncomfortable question about probing classifiers: if a probe can decode a linguistic property (like part-of-speech) from a representation with high accuracy, is that because the representation encodes the property — or because the probe itself is powerful enough to memorize the mapping given enough training data? The authors show that probe accuracy alone conflates two very different things: how much task-relevant information the representation contains, and how much the probe learned to memorize on its own. They propose **control tasks** — a variant of the same probing setup but with random, structure-matched labels — and a resulting metric, **selectivity**, to disentangle the two.

### Key Insights

- A probe's accuracy on a real linguistic task (e.g., POS tagging) is not by itself evidence that the representation "contains" that property in a meaningful way — a sufficiently expressive probe can fit almost any label given enough capacity and training.
- **Control tasks** pair each real task with a task of matched complexity (same input types, same output space, same class-size distribution) but with labels assigned essentially at random (e.g., a random POS tag per word type, memorized rather than predicted from structure).
- **Selectivity** = accuracy on the real task − accuracy on the control task. High selectivity means the representation, not the probe, is doing the work.
- More expressive probes (e.g., deep MLPs) tend to have high accuracy on *both* the real and control tasks — meaning low selectivity — because they can simply memorize the control task's random mapping.
- Simpler probes (linear, or low-capacity MLPs) tend to have lower absolute accuracy but much higher selectivity, making their results easier to trust as a statement about the underlying representation.
- The choice of probe architecture is itself an experimental design decision with consequences for what conclusions can be drawn — it is not a neutral implementation detail.

### Why It's Worth Its Salt

Before this paper, "probe accuracy" was often reported as if it were a direct measurement of what a representation encodes. Hewitt & Liang showed that this measurement is contaminated by the probe's own capacity to memorize, and gave the field a concrete, cheap methodology (control tasks + selectivity) to correct for it. It reframed probing from "train a classifier and report accuracy" to "design a controlled experiment," which is the same spirit behind later work distinguishing decodability from disentanglement.

### Architectural Pattern

- Take the representation to be probed (frozen, not fine-tuned).
- Define the real task (e.g., POS tag prediction from word representation).
- Define a matched control task: same input distribution and output cardinality, but labels drawn once at random per input type (so they're structure-free but still learnable by memorization).
- Train the same probe family (e.g., linear, or MLP with a given hidden size) on both tasks.
- Compute selectivity = real-task accuracy − control-task accuracy.
- Prefer probe architectures/hyperparameters that maximize selectivity, not raw accuracy, when the goal is to interpret what the representation encodes.

### Connections

- Directly extends and critiques the linear-probe methodology popularized by Alain & Bengio (2016).
- Foreshadows later information-theoretic reframings of probing (e.g., minimum-description-length probes, Voita & Titov 2020) that formalize the same "probe capacity contaminates the measurement" concern.
- Conceptually parallel to the disentanglement literature: just as a linear probe's accuracy doesn't prove disentanglement (accessibility ≠ structure), a probe's accuracy doesn't prove "true" encoding without correcting for probe capacity (accuracy ≠ selectivity). See [[What Is a Linear Probe]] for the accessibility-vs-structure distinction this paper's selectivity metric complements.

### Implementation Notes

- Evaluated on POS tagging and CCG supertagging using word representations from trained (and untrained/random) LSTMs.
- Varied probe capacity (linear vs. MLPs with increasing hidden size) and tracked how accuracy and selectivity moved in opposite directions as capacity increased.
- Released code for constructing control tasks and computing selectivity, making the methodology directly reusable for other probing setups.

### Critiques / Limitations

- Constructing a good control task requires care — a poorly matched control task can under- or over-estimate how much a probe would memorize on the real task.
- Selectivity is a diagnostic, not a full solution: it tells you when to be suspicious of a probe result, but doesn't itself explain *what* structure the representation actually has.
- Focused on NLP sequence-labeling tasks (POS, CCG supertags); applying the same control-task logic to other modalities (vision, multimodal) requires re-deriving what a "structure-free but matched" control task looks like.

---

### Educational Connections

**Undergraduate-Level**

- Illustrates a general experimental-design lesson: any measurement tool (here, a probe) has its own capacity/bias, and that capacity must be controlled for before trusting the measurement.
- Good first example of "accuracy alone is not evidence" — a theme that recurs throughout ML evaluation.

**Postgraduate-Level**

- Directly relevant to designing new probing studies: always ask "what would this probe's accuracy be on random labels of the same shape?"
- Connects to broader interpretability methodology — the same selectivity-style correction shows up in causal-intervention and information-theoretic probing work.

---

### My Notes

- This is the natural next step after asking "does a linear probe prove disentanglement?" — even before disentanglement, we should ask whether the probe's own capacity is inflating the *decodability* claim itself.
- Useful mental model: control tasks do for probing accuracy what a placebo group does for a drug trial — they isolate the effect of the "treatment" (the representation) from the effect of the "instrument" (the probe).
- Open question for the VLM factorization work: what would a good control task look like for the color/shape factor-swapping experiments — i.e., a task with matched structure but randomized color/shape assignment, to check that factor-swap consistency isn't just probe/metric memorization?
