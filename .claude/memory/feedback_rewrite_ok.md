---
name: feedback-rewrite-ok
description: User is willing to rewrite everything from scratch; previous codebase was just a starting point from a related project
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5cb5a86e-ac46-4e67-80da-cf058c4016b6
---

User explicitly said all existing code (dataset, training loop, preprocessing) can be completely rewritten for best results. The previous project (temporal velocity predictor) was just a starting point — same dataset (nuScenes), overlapping ideas, but not the right architecture for this project.

**Why:** Starting fresh avoids patching the wrong abstraction (e.g. pre-saved BEV files, single-agent dataset) and gives a cleaner foundation.

**How to apply:** Don't try to preserve or patch existing `src/data/dataset.py`, `scripts/train.py`, etc. when the architecture no longer fits. A clean rewrite is preferred. See `docs/implementation_plan.md` for the full build order and what to rewrite vs keep.
