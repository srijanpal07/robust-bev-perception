# Robust BEV Perception under Sensor Degradation

> Graceful, predictable degradation across the full sensor quality spectrum — from dense LiDAR to camera-only — via stochastic dropout training, language-grounded modality gating, and zero-shot sim-to-real transfer validation.

## The Problem

LiDAR-equipped AV systems experience structured sensor degradation in adverse weather and hardware wear — fog scatters 60–80% of LiDAR returns, rain blinds cameras, dust storms degrade both. This creates a training-inference gap: models trained on clean 32-beam LiDAR data fail **silently and unpredictably** when deployed sensors degrade, because the degraded-sensor distribution was never seen during training. Current approaches either hard-switch between modalities when one fails (brittle boundary) or fuse sensors assuming all are healthy (no graceful fallback). Neither handles the continuous degradation spectrum that real deployments encounter.

This is not a solved problem. Waymo's Gen 6 (2026) added mechanical sensor cleaning and redundancy hardware specifically to compensate for this software gap. Foundation models and scale alone do not fix it — they still require training data covering the degradation distribution.

## The Approach

We train a unified perception framework on real nuScenes sensor data (dense LiDAR, camera, radar) while injecting a continuous spectrum of synthetic degradation at training time via **stochastic LiDAR beam dropout** — randomly sampling beam counts {32, 16, 8, 4, 0} per batch. The model learns a shared BEV latent representation that encodes scene geometry and agent motion regardless of current sensor quality.

At inference, a **language-grounded degradation estimator** (CLIP ViT-L/14, LoRA fine-tuned on Isaac Sim data) takes front-camera frames and produces a semantic quality embedding — cameras can see weather conditions (fog, haze, rain) even when LiDAR cannot measure its own degradation. This embedding drives a **soft modality gating network** that continuously re-weights LiDAR vs. camera BEV features, smoothly degrading from full-stack perception down to camera-only without hard switching.

**Sim-to-real transfer:** The model is trained exclusively on real nuScenes data. Physics-accurate degraded point clouds are generated in NVIDIA Isaac Sim (fog, rain, 4-beam hardware, LiDAR failure). We evaluate zero-shot — no Isaac Sim data at training time — measuring how well stochastic dropout training generalizes to physically accurate degradation.

The temporal transformer outputs per-agent future trajectories (multi-step waypoints), benchmarked on nuScenes motion forecasting, making the output directly useful for downstream planning.

## Baseline

This project builds on a temporal transformer-based velocity predictor trained on nuScenes LiDAR BEV maps and camera-derived bounding boxes ([baseline repo](https://github.com/srijanpal07/temporal-lidar-camera-bev-fusion)):

- BEV encoder (ResNet18) + crop encoder + 2-layer 4-head transformer
- Camera-detector noise injection at runtime (distance-dependent, non-Gaussian)
- Kalman filter for kinematic velocity estimation from noisy detections
- Residual velocity prediction: model predicts `(GT_vel - noisy_vhat)`

## Project Structure

```
robust-bev-perception/
├── src/                    # Core model and dataset code (baseline)
│   ├── model.py            # Temporal transformer architecture
│   ├── dataset.py          # nuScenes dataloader + noise injection
│   ├── kalman.py           # Kalman filter for kinematic features
│   ├── box_noise.py        # Camera-detector noise simulation
│   └── utils.py
├── scripts/                # Training, inference, visualization
├── configs/                # YAML experiment configs
├── experiments/            # Per-experiment results and notes
├── docs/                   # Research notes, paper drafts
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
- [MoME (CVPR 2025)](https://arxiv.org/abs/2503.19776) — discrete expert routing for sensor failure; we use soft language-grounded gating
- [RESBev (arXiv 2026)](https://arxiv.org/abs/2603.09529) — latent world model for BEV robustness; we operate at training policy + gating level

**Foundations:**
- [LEROjD (ECCV 2024)](https://arxiv.org/abs/2409.05564) — LiDAR-train / radar-infer transfer; binary, not continuous
- [BEVWorld (2024)](https://arxiv.org/html/2407.05679v3) — multimodal world model in BEV latent space
- [Cocoon (2024)](https://arxiv.org/html/2410.12592v1) — uncertainty-aware sensor fusion

**VLM / language for AV:**
- [DriveX @ CVPR 2026](https://drivex-workshop.github.io/cvpr2026/) — foundation models + adverse weather perception
- [AUTOPILOT @ CVPR 2026](https://www.autopilot-cvpr.net/) — VLMs for safety-critical AV perception
