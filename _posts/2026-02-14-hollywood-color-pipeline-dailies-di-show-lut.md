---
layout: post
title: "Hollywood Color Pipeline: From Dailies to DI (and Why Show LUT Is the Film's Visual DNA)"
description: "A practical mental model for understanding how footage moves from set monitors to final theatrical and HDR masters."
tags: [color, filmmaking, dailies, digital-intermediate, show-lut, aces]
---

If you want to understand Hollywood color, you do **not** start with knobs and curves.
You start with a pipeline.

The same shot can look different at every stage of post-production, not because anyone is "doing it wrong," but because each stage has a different job. Once you see those jobs clearly, the whole system becomes intuitive.

This post compresses that workflow into one mental model you can carry into real projects.

## The Big Picture: Three Phases

### 1) Capture & Preview
**Goal:** quickly and reliably see what was shot.

This phase is about confidence and communication, not perfection.

- **Show / Look LUT**
  - A pre-designed creative color style for the project.
  - Used on set (monitors, video village) and in early viewing passes.
  - Establishes the film's early visual identity—often most of the perceived mood.
- **Dailies**
  - First processed footage after shooting.
  - Usually includes a viewing transform (often the show LUT), basic balancing, synced audio, and editorial-friendly proxies.
  - Reviewed by director, DP, producers, and editor.
  - Explicitly **not** the final color result.

**Core idea:** Dailies are a usable preview of reality.

---

### 2) Editorial & Preparation
**Goal:** lock the story before final image authorship.

- Editorial cuts using proxy dailies.
- Notes, revisions, and story decisions happen here.
- After **picture lock**, the timeline is conformed/re-linked to original camera source (RAW/high-quality masters).
- The project is then prepared for final grading (DI).

**Core idea:** Story first. Perfect color later.

---

### 3) Final Finishing (Digital Intermediate)
**Goal:** create the definitive image the audience will see.

- **DI (Digital Intermediate)** uses full-quality source material.
- Shot-by-shot, scene-by-scene precision grading.
- Creative decisions made/approved by director + DP + colorist.
- Final deliverables produced (theatrical, SDR, HDR, streaming variants, etc.).

**Core idea:** DI is final cinematic truth.

## Dailies vs DI: Same Film, Different Mission

| Aspect | Dailies | Digital Intermediate (DI) |
|---|---|---|
| Timing | Right after shooting | End of post-production |
| Purpose | Review + editorial workflow | Final artistic + technical finish |
| Quality | Proxies / temporary processing | Full RAW / highest fidelity |
| Color treatment | Viewing LUT + basic balancing | Precise creative grade + mastering |
| Audience | Internal filmmaking team | Public release |
| Primary owners | DIT / dailies lab / post support | Director + DP + colorist |

A good memory hook:

> **Dailies help filmmakers see the footage. DI decides what the world sees.**

## Why Show LUT Matters So Much

Show LUT is not "just a monitoring trick." It acts like the project's visual throughline.

It influences:
- On-set exposure choices.
- How dailies feel emotionally.
- Editorial rhythm and tone (because cuts are made against that look).
- The starting point and intent in DI.

In practice:

> **Show LUT = visual DNA of the film.**

It may evolve in DI, but its fingerprints usually survive to the final master.

## The Deeper System Design: Hollywood Separates Two Problems

Hollywood intentionally splits the workflow into two different optimization targets:

1. **Technical visibility problem** → solved by dailies.
   - Speed, consistency, logistics, communication.
2. **Creative authorship problem** → solved by DI.
   - Emotion, narrative emphasis, visual meaning.

This separation is what makes high-end pipelines robust.

It also explains why "AI color" is not one task:
- **Auto-dailies** is primarily an efficiency/operations challenge.
- **AI grading** is a creative-intelligence challenge.

They are connected, but they are not interchangeable.

## One-Glance Pipeline Diagram

```text
Capture (Sensor/RAW)
   ↓
On-set monitoring (Show LUT)
   ↓
Dailies generation (LUT + balance + audio sync + proxies)
   ↓
Editorial (proxy cut)
   ↓
Picture lock
   ↓
Conform/relink to RAW
   ↓
DI grading (shot-by-shot final creative decisions)
   ↓
Mastering (Theatrical / SDR / HDR deliverables)
```

If you remember only one thing, remember this:

> **Color in cinema is a staged decision process, not a single correction pass.**

## Final Takeaway

When people ask, "Why can't we just grade once at the beginning and be done?" this pipeline is the answer.

Early stages optimize for speed and shared understanding.
Late stages optimize for authorship and final fidelity.

That is why professional color pipelines scale—and why the best future tools (including AI tools) will respect this separation instead of trying to collapse it.
