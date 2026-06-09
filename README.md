# Robust BEV Perception under Sensor Degradation

> Graceful, predictable degradation across the full sensor quality spectrum — from dense LiDAR to camera-only — via world-model-inspired multi-modal training.

## The Problem

Modern AV perception systems face a fundamental deployment problem: they are trained on expensive, high-quality sensor data — dense 32-beam LiDAR, calibrated cameras, and radar — but in the real world sensors degrade continuously due to weather (fog scatters LiDAR, rain blinds cameras), hardware aging, or cost constraints (a $200 4-beam LiDAR behaves nothing like the $5000 unit the model trained on). Current approaches either hard-switch between modalities when one fails, or fuse sensors assuming all are healthy — both strategies break unpredictably at the boundary.

## The Approach

We train a unified perception framework on the full high-quality sensor stack — dense LiDAR, camera, and 4D radar — while simultaneously simulating a continuous spectrum of sensor degradation at training time through stochastic LiDAR beam dropout, camera-detector noise injection, and radar sparsity augmentation. The model learns a shared Bird's Eye View (BEV) latent representation, inspired by world models, that encodes scene geometry and agent motion regardless of which sensors are available at any given moment.

At inference, a lightweight degradation-predictor network monitors real-time sensor quality per modality and produces a soft confidence score that continuously re-weights each modality's contribution to the shared BEV latent — smoothly transitioning from a full (dense LiDAR + camera + radar) configuration down through (sparse LiDAR + camera), (camera + radar), and ultimately (camera only) without hard switching, distribution shift, or catastrophic failure. The unified temporal transformer then operates on this robust latent representation to predict per-agent future trajectories, not just instantaneous velocity, making the output directly useful for downstream planning.

The entire framework is validated on nuScenes using real sensor data and extended to physics-accurate adverse weather scenarios generated in NVIDIA Isaac Sim, where ground-truth degradation conditions can be precisely controlled — something no existing dataset provides.

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

- [LEROjD (ECCV 2024)](https://arxiv.org/abs/2409.05564) — LiDAR-train / radar-infer knowledge transfer
- [BEVWorld (2024)](https://arxiv.org/html/2407.05679v3) — multimodal world model in BEV latent space
- [Cocoon (2024)](https://arxiv.org/html/2410.12592v1) — uncertainty-aware sensor fusion
- [DriveMoE (2025)](https://arxiv.org/pdf/2505.16278) — mixture-of-experts for autonomous driving
- [Cam4DOcc (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Cam4DOcc_Benchmark_for_Camera-Only_4D_Occupancy_Forecasting_in_Autonomous_Driving_CVPR_2024_paper.pdf) — camera-only 4D occupancy forecasting
