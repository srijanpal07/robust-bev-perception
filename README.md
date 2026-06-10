# Robust BEV Perception under Sensor Degradation

> Calibrated, graceful trajectory forecasting across the full LiDAR degradation spectrum — from 32-beam to camera-only — via stochastic dropout training and uncertainty-aware modality gating.

## The Problem

LiDAR-equipped AV systems experience structured sensor degradation in adverse weather and hardware wear — fog scatters 60–80% of LiDAR returns, rain blinds cameras, dust storms degrade both. This creates a training-inference gap: models trained on clean 32-beam LiDAR data fail **silently and unpredictably** when deployed sensors degrade, because the degraded-sensor distribution was never seen during training. Current approaches either hard-switch between modalities when one fails (brittle boundary) or fuse sensors assuming all are healthy (no graceful fallback). Neither handles the continuous degradation spectrum that real deployments encounter.

This is not a solved problem. Waymo's Gen 6 (2026) added mechanical sensor cleaning and redundancy hardware specifically to compensate for this software gap. Foundation models and scale alone do not fix it — they still require training data covering the degradation distribution.

## The Approach

We train a unified perception framework on real nuScenes sensor data (dense LiDAR, camera, radar) while injecting a continuous spectrum of synthetic degradation at training time via **stochastic LiDAR beam dropout** — randomly sampling beam counts {32, 16, 8, 4, 0} per batch. The model learns a shared BEV latent representation that encodes scene geometry and agent motion regardless of current sensor quality.

At inference, a lightweight **modality gating network** estimates per-modality quality from point cloud statistics and continuously re-weights LiDAR vs. camera BEV features — smoothly degrading from full-stack perception down to camera-only without hard switching.

Critically, the trajectory prediction head outputs a **distribution** over future waypoints (not just a point estimate). Uncertainty should widen as sensor quality drops — a model that is inaccurate but knows it is still useful for safe planning. We measure this with ECE (Expected Calibration Error) and NLL alongside ADE/FDE.

**Cross-domain robustness validation:** The model is trained exclusively on real nuScenes data. Physics-accurate degraded point clouds are generated in NVIDIA Isaac Sim (fog, rain, 4-beam hardware, LiDAR failure). We evaluate on Isaac Sim outputs without any Isaac Sim training data — measuring whether stochastic dropout training generalizes to physically accurate degradation beyond artificially subsampled beams.

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        FULL SYSTEM                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TRAINING                                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Real nuScenes: 32-beam LiDAR + 6 cameras + box detects  │    │
│  │  + Stochastic beam dropout: {32, 16, 8, 4, 2, 0} / batch │    │
│  │    0 beams per batch = camera-only training              │    │
│  └────────────────────────┬─────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  BEV Encoder ── learns P(BEV | any sensor quality)              │
│  ResNet18(500×500 BEV) · CropEncoder(64×64) · Box MLP           │
│  concatenated: 256 + 128 + 64 = 448-dim per frame               │
│                           │                                      │
│                           ▼                                      │
│  Temporal Transformer ── 2-layer, 4-head, T=3 frames            │
│  mean-pool over history → 256-dim context vector                │
│                           │                                      │
│                           ▼                                      │
│  Modality Gating ── pc stats → quality score q ∈ [0, 1]        │
│  fused = q · f_LiDAR  +  (1−q) · f_camera                      │
│  (no hard switch — q transitions continuously)                  │
│                           │                                      │
│                           ▼                                      │
│  Trajectory Head ── (μ, log σ) × 6 steps × 0.5s = 3s horizon   │
│  uncertainty σ widens as q → 0  ·  ECE-calibrated               │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INFERENCE  ─  one model, continuous graceful degradation        │
│                                                                  │
│  32-beam        ──→     8-beam        ──→   camera-only          │
│  q ≈ 1.0                q ≈ 0.4–0.6         q ≈ 0.0             │
│  narrow σ               moderate σ           wide σ             │
│  (confident)            (cautious)           (very uncertain)   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Baseline

This project builds on a temporal transformer-based velocity predictor trained on nuScenes LiDAR BEV maps and camera-derived bounding boxes ([baseline repo](https://github.com/srijanpal07/temporal-lidar-camera-bev-fusion)):

- BEV encoder (ResNet18) + crop encoder + 2-layer 4-head transformer
- Camera-detector noise injection at runtime (distance-dependent, non-Gaussian)
- Kalman filter for kinematic velocity estimation from noisy detections
- Residual velocity prediction: model predicts `(GT_vel - noisy_vhat)`

## Implementation Status

| Component | Status | Location |
|---|---|---|
| BEV encoder + temporal transformer (velocity head) | **Working** | `src/models/` |
| Camera-detector noise injection | **Working** | `src/data/box_noise.py` |
| Kalman + RTS smoother | **Working** | `src/utils/kalman.py` |
| LiDAR beam dropout pipeline (C1/C2) | **Stub** | `src/data/beam_degradation.py` |
| Stochastic dropout training (C2) | **Not wired** — awaits C1 | `configs/train_dropout.yaml` |
| Uncertainty-aware modality gating (C3) | **Stub** | `src/models/gating.py` |
| Trajectory head + NLL loss (C4) | **Stub** | `src/models/heads.py` |
| Cross-domain Isaac Sim evaluation | **Stub** | `src/data/isaac_sim.py` |
| CLIP-based gating ablation | **Not started** | — |

## Project Structure

```
robust-bev-perception/
├── src/
│   ├── data/               # Dataset, degradation, augmentation
│   │   ├── dataset.py      # BEVVelocityDataset  ← canonical
│   │   ├── beam_degradation.py  # C1/C2 beam dropout (stub)
│   │   ├── trajectory_targets.py
│   │   ├── isaac_sim.py    # cross-domain eval dataset (stub)
│   │   ├── box_noise.py    # camera-detector noise
│   │   └── transforms.py
│   ├── models/             # Model components
│   │   ├── encoders.py     # BEVEncoder, ResNet18BEVEncoder, CropEncoder
│   │   ├── temporal.py     # TemporalVelocityPredictor  ← canonical
│   │   ├── heads.py        # VelocityHead (working), TrajectoryHead (stub)
│   │   └── gating.py       # ModalityGating (stub)
│   ├── training/           # Losses, metrics, calibration, checkpointing
│   ├── eval/               # Degradation curves, forecasting, calibration metrics
│   ├── utils/              # Kalman smoother, geometry helpers
│   │   ├── kalman.py
│   │   └── geometry.py
│   │
│   # Legacy re-exports (kept for backward compat — import from src.models.* instead)
│   ├── model.py  ·  dataset.py  ·  kalman.py  ·  box_noise.py
│
├── scripts/
│   ├── train/train.py      # ← canonical training entrypoint (YAML-driven)
│   ├── eval/               # eval_degradation_curve, eval_calibration, eval_isaac_sim
│   ├── baselines/          # BEVFormer matching / inference (migrating)
│   │
│   # Legacy scripts (original baseline — still functional)
│   ├── train.py  ·  infer.py  ·  visualize.py
│   └── save_bev.py  ·  filter_dataset.py  ·  prepare_dataset.py  ·  …
│
├── configs/                # YAML experiment configs
├── docs/                   # overview.html — open in browser for full research overview
├── data/                   # Symlink or mount point for nuScenes
└── outputs/                # Training outputs (not tracked in git)
```

## Setup

```bash
pip install -r requirements.txt
```

See `configs/train.yaml` for dataset paths and training hyperparameters.

## Hardware

- GPU: NVIDIA RTX Pro 5000 Ada (~32GB VRAM)
- Simulator: NVIDIA Isaac Sim (Omniverse)

## Research Goals

See [RESEARCH.md](RESEARCH.md) for the full research plan, short-term and long-term goals, related work, and target venues.

## Related Work

**Direct competitors (2025–2026):**
- [Grace-BEV (arXiv May 2026)](https://arxiv.org/abs/2605.30983) — closest competitor: plug-in TrustGate Router + binary modality dropout, mAP on nuScenes-R/C; our deltas: continuous beam spectrum, trajectory + calibrated uncertainty
- [Benchmarking Sensor-Fault Robustness in Forecasting (arXiv May 2026)](https://arxiv.org/abs/2605.10822) — closest benchmark; read before finalizing evaluation setup
- [MoME (CVPR 2025)](https://arxiv.org/abs/2503.19776) — discrete expert routing for sensor failure; we use soft continuous gating with calibrated output
- [RESBev (arXiv Mar 2026)](https://arxiv.org/abs/2603.09529) — post-hoc latent recovery for BEV robustness; we bake robustness into training policy
- [MetaBEV (2023)](https://arxiv.org/abs/2304.09801) — sensor failure for BEV detection; detection focus, not trajectory + calibration

**Foundations:**
- [LEROjD (ECCV 2024)](https://arxiv.org/abs/2409.05564) — LiDAR-train / radar-infer transfer; binary, not continuous
- [BEVWorld (2024)](https://arxiv.org/html/2407.05679v3) — multimodal world model in BEV latent space
- [Cocoon (2024)](https://arxiv.org/html/2410.12592v1) — uncertainty-aware sensor fusion
