# Implementation Plan

This document is the developer reference for build order, phase definitions, architecture,
and what needs to be written or rewritten. The research framing lives in RESEARCH.md.

---

## Architecture Pipeline (Target)

```
nuScenes raw LiDAR (.pcd.bin)
        │
        ▼  beam dropout — filter by ring index at load time (stochastic per batch)
        │  beam levels: {32, 16, 8, 4, 2, 0} sampled randomly each training step
        │
        ▼  BEV voxelizer — render (C, 500, 500) occupancy/height/intensity grid
        │  online in DataLoader workers — no pre-saved BEV files needed
        │
        ▼  Scene-level Dataset — returns ALL agents in a scene simultaneously
        │  (N agents per scene, variable N, zero-padded in collate_fn)
        │  annotations loaded from nuScenes-devkit (boxes, timestamps, categories)
        │
        ▼  ResNet18BEVEncoderWithFeatures — shared BEV encoder, ONE pass per scene
        │  returns: pooled (B, 256)  +  feat_map (B, 512, 16, 16)
        │
        ├──[Phase 1]── CropEncoder  (80px crop → resize 64×64 → 128-dim)
        │              per agent, extracted from feat_map or raw BEV
        │
        └──[Phase 2]── RoIAgentEncoder  (RoI Align on shared feat_map)
                       torchvision.ops.roi_align, spatial_scale=16/500
                       returns (B, N, 256) — O(1) BEV encoder cost
        │
        ▼  q_i — per-agent quality score from local point density in agent's BEV region
        │  count non-zero voxels in each agent's crop region → sigmoid-normalize
        │  shape: (B, N, 1) in Phase 2 / (B, 1) in Phase 1
        │
        ▼  ModalityGating [Phase 1: (B, feat)] / MultiAgentModalityGating [Phase 2: (B, N, feat)]
        │  fused_i = q_i · f_LiDAR_i + (1 − q_i) · f_camera_i
        │
        ▼  [Phase 2 only] AgentInteractionModule — 2-layer, 4-head
        │  attn_{ij} = softmax((Q_i·K_j − λ·(1−q_j)) / √d_k)
        │  adapted from HiVT [Zhou et al., CVPR 2022]; λ is learnable
        │
        ▼  Trajectory Head — per agent, per waypoint
           output: (μ_i, log σ_i) × 6 future steps (3s at 0.5s intervals)
           loss:   NLL (Gaussian) + optional ADE supervision
           metrics: ADE, FDE, NLL, ECE
```

---

## Disk & Environment

- **nuScenes trainval raw data**: ~400GB total (keep all — cameras needed for modality gating)
  - LiDAR sweeps: ~80GB
  - Camera images: ~200GB
  - Radar + maps + annotations: ~25GB
- **No pre-saved BEV files** — render online from raw LiDAR in DataLoader workers
- **Conda env**: `bevrobust` (Python 3.10, torch 2.7.0+cu126, numpy 1.26.4 pinned)
- **Isaac Sim env**: `env_isaacsim` — keep separate, do NOT merge

---

## Phase 1 — Single-Agent Baseline (C1–C3)

**Scope:** `vehicle.car` only. Prove the core sensor degradation idea works before adding
multi-agent complexity.

**Goals:**
- C1/C2: Stochastic beam dropout training bakes degradation robustness into the model
- C3: Per-agent quality score `q_i` gates LiDAR vs camera; `σ_i` widens as `q_i → 0`
- Metrics: ADE/FDE + NLL + ECE across degradation levels

**What to build (in order — see Build Order section):**

| Step | Component | File | Status |
|---|---|---|---|
| 1 | Beam dropout (ring index filter) | `src/data/beam_degradation.py` | Rewrite stub |
| 2 | BEV voxelizer (online rendering) | `src/data/bev_renderer.py` | New |
| 3 | nuScenes scene-level dataset | `src/data/nuscenes_dataset.py` | New |
| 4 | q_i computation (local density) | `src/data/quality_score.py` | New |
| 5 | BEV encoder (shared, feat map) | `src/models/encoders.py` | Exists — `ResNet18BEVEncoderWithFeatures` |
| 6 | Crop encoder (Phase 1 agent feats) | `src/models/encoders.py` | Exists — `CropEncoder` |
| 7 | Modality gating (single-agent) | `src/models/gating.py` | Exists — `ModalityGating` (stub, wire) |
| 8 | Trajectory head (μ, log σ) | `src/models/heads.py` | Rewrite stub |
| 9 | NLL loss + ADE/FDE | `src/training/losses.py` | Rewrite |
| 10 | ECE metric | `src/training/metrics.py` | New |
| 11 | Training loop (Phase 1) | `scripts/train/train_p1.py` | New |
| 12 | Degradation curve eval | `scripts/eval/eval_degradation_curve.py` | Stub → wire |

---

## Phase 2 — Multi-Agent + Interaction (C4)

**Scope:** All dynamic categories (vehicles, pedestrians, cyclists). All agents in a scene
predicted jointly. Relaxed dataset filters (min track length ~12 frames, no speed floor).

**Goals:**
- C4: Agent-agent interaction with degradation-aware attention
- Trajectory ADE/FDE/NLL/ECE for all agent categories
- RoI Align replaces per-agent CropEncoder (O(1) BEV encoder cost)

**What to build (after Phase 1 is validated):**

| Step | Component | File | Status |
|---|---|---|---|
| 1 | Multi-agent collate_fn (variable N, padding) | `src/data/nuscenes_dataset.py` | Extend Phase 1 dataset |
| 2 | Multi-category filter (all dynamic) | `src/data/nuscenes_dataset.py` | Extend |
| 3 | RoI Align agent encoder | `src/models/encoders.py` | Exists — `RoIAgentEncoder` (wire) |
| 4 | Multi-agent modality gating | `src/models/gating.py` | Exists — `MultiAgentModalityGating` (wire) |
| 5 | Interaction module | `src/models/interaction.py` | Exists — `AgentInteractionModule` (wire) |
| 6 | Training loop (Phase 2) | `scripts/train/train_p2.py` | New |
| 7 | Per-category eval | `scripts/eval/eval_per_category.py` | New |

---

## Phase 3 — Isaac Sim Cross-Domain Validation

**Scope:** Purely held-out evaluation. Zero Isaac Sim data at training time.

**Goals:** Prove robustness transfers from synthetic beam dropout training to
physics-accurate degradation (fog, rain, hardware failure). Stronger claim than
held-out nuScenes splits alone.

**Timing:** After Phase 2 is validated. Do NOT start this before Phase 2.

**What to do:**

| Step | Task | Notes |
|---|---|---|
| 1 | Configure Isaac Sim sensor params | Match HDL-32E: 32 beams, 10Hz, beam angles |
| 2 | Generate degraded scenes | Fog, rain, 4-beam hardware failure, full LiDAR dropout |
| 3 | Export to nuScenes-compatible format | Reuse eval pipeline unchanged |
| 4 | Run trained Phase 2 model, no fine-tuning | `scripts/eval/eval_isaac_sim.py` |
| 5 | Report ADE/FDE/ECE vs beam dropout baseline | Cross-domain generalization result |

**One early task (can do before Phase 2):** lock down Isaac Sim HDL-32E sensor parameters
so they don't need revisiting late. One config file, no model work.

---

## Build Order (Phase 1 — detailed)

This is the recommended linear order. Each step unblocks the next.

```
Step 1: beam_degradation.py
    - Input:  (N, 4) LiDAR point cloud with ring index in column 4
    - Output: filtered (N', 4) point cloud with only k rings retained
    - Sample k from {32, 16, 8, 4, 2, 0} uniformly each call
    - Validate: plot point cloud before/after, count rings

Step 2: bev_renderer.py
    - Input:  filtered point cloud, BEV config (size=500, resolution=0.2m)
    - Output: (C, 500, 500) numpy array — channels: max height, density, intensity, ...
    - Validate: visualise rendered BEV, compare 32-beam vs 4-beam visually

Step 3: nuscenes_dataset.py (Phase 1 version)
    - One sample = one scene × one keyframe × one agent (single-agent for now)
    - Loads raw LiDAR → beam_degradation → bev_renderer → crop → q_i
    - Returns: (bev, crop, box_params, q_score, future_waypoints, mask)
    - Train/val split: use nuScenes official splits (scene tokens in nuscenes-devkit)

Step 4: quality_score.py
    - Count non-zero voxels in agent's crop region of the rendered BEV
    - Normalise to [0, 1] — at 32 beams: q ≈ 1.0, at 0 beams: q ≈ 0.0
    - Return as scalar per agent per frame

Step 5: wire ResNet18BEVEncoderWithFeatures
    - Already implemented — just needs to be called from the training loop
    - Returns (pooled, feat_map); feat_map used for CropEncoder in Phase 1

Step 6: wire CropEncoder (Phase 1) / RoIAgentEncoder (Phase 2)
    - Phase 1: extract 80px crop from BEV around agent bbox → resize to 64×64 → CropEncoder
    - Already implemented; just needs to receive feat_map or raw BEV

Step 7: wire ModalityGating
    - Input: LiDAR feat (B, 256), camera feat (B, 256), q_score (B, 1)
    - Output: fused (B, 256)
    - Phase 1: single agent, pc_stats = [q_score, local_density_features]

Step 8: TrajectoryHead
    - Input: fused feat (B, 256)
    - Output: (μ, log σ) × 6 waypoints = (B, 6, 2), (B, 6, 2)
    - Gaussian NLL loss: -log N(y | μ, σ²)

Step 9: Training loop (scripts/train/train_p1.py)
    - DataLoader with online BEV rendering (num_workers=6-8 on NVMe)
    - Stochastic beam level sampled per batch by dataset
    - Log: train NLL, val NLL, val ADE, val FDE, val ECE per beam level

Step 10: ECE evaluation
    - Bin predictions by predicted σ, check empirical coverage matches
    - Report as reliability diagram + scalar ECE
    - Evaluate separately for each beam level: {32, 16, 8, 4, 2, 0}
```

---

## Code Status Summary

### Rewrite (existing stubs are wrong architecture or incomplete)

| File | What to do |
|---|---|
| `src/data/dataset.py` | Full rewrite → `src/data/nuscenes_dataset.py` using nuScenes-devkit natively; online LiDAR loading |
| `src/data/beam_degradation.py` | Rewrite stub — implement ring index filter on raw point cloud |
| `src/models/heads.py` | Rewrite stub — implement `TrajectoryHead` outputting (μ, log σ) |
| `src/training/losses.py` | Rewrite — implement Gaussian NLL loss |
| `src/training/metrics.py` | Rewrite — implement ADE, FDE, ECE |
| `scripts/train/train.py` | Replace with `train_p1.py` (Phase 1) and `train_p2.py` (Phase 2) |

### New files needed

| File | Purpose |
|---|---|
| `src/data/bev_renderer.py` | Online BEV voxelizer from raw LiDAR |
| `src/data/nuscenes_dataset.py` | Scene-level dataset using nuScenes-devkit |
| `src/data/quality_score.py` | q_i from local point density |
| `scripts/train/train_p1.py` | Phase 1 training entrypoint |
| `scripts/train/train_p2.py` | Phase 2 training entrypoint |

### Exists and ready to wire (no changes needed)

| File | What's there |
|---|---|
| `src/models/encoders.py` | `ResNet18BEVEncoderWithFeatures`, `RoIAgentEncoder`, `CropEncoder` |
| `src/models/gating.py` | `ModalityGating` (P1), `MultiAgentModalityGating` (P2) |
| `src/models/interaction.py` | `AgentInteractionModule` with `DegradationAwareAttention` |
| `src/models/__init__.py` | Exports all Phase 1 + Phase 2 classes |

### Keep as-is (Phase 1 still uses them)

| File | Notes |
|---|---|
| `src/data/box_noise.py` | Camera detector noise — still relevant |
| `src/utils/kalman.py` | Kalman smoother for kinematic features — still relevant |
| `src/utils/geometry.py` | Geometry helpers |
| `configs/` | Update paths/hyperparams as needed |

### Legacy (previous project — can ignore or delete)

| File | Notes |
|---|---|
| `scripts/train.py` | Old single-file training script |
| `scripts/infer.py` | Old inference script |
| `src/data/dataset.py` | Being replaced by `nuscenes_dataset.py` |
| `src/models/temporal.py` | `TemporalVelocityPredictor` — velocity-only model, replaced by trajectory head |
