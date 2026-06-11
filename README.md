# Robust BEV Perception under Sensor Degradation

> Calibrated, graceful trajectory forecasting across the full LiDAR degradation spectrum — from 32-beam to camera-only — via stochastic dropout training and uncertainty-aware modality gating.

## The Problem

LiDAR-equipped AV systems experience structured sensor degradation in adverse weather and hardware wear — fog scatters 60–80% of LiDAR returns, rain blinds cameras, dust storms degrade both. This creates a training-inference gap: models trained on clean 32-beam LiDAR data fail **silently and unpredictably** when deployed sensors degrade, because the degraded-sensor distribution was never seen during training. Current approaches either hard-switch between modalities when one fails (brittle boundary) or fuse sensors assuming all are healthy (no graceful fallback). Neither handles the continuous degradation spectrum that real deployments encounter.

This is not a solved problem. Waymo's Gen 6 (2026) added mechanical sensor cleaning and redundancy hardware specifically to compensate for this software gap. Foundation models and scale alone do not fix it — they still require training data covering the degradation distribution.

## The Approach

We train a unified perception framework on real nuScenes sensor data while injecting a continuous spectrum of synthetic degradation at training time via **stochastic LiDAR beam dropout** — randomly sampling beam counts {32, 16, 8, 4, 0} per batch. The model encodes all dynamic agents in the scene (vehicles, pedestrians, cyclists) simultaneously and learns a shared BEV representation that encodes geometry and agent motion regardless of current sensor quality.

At inference, a lightweight **per-agent modality gating network** estimates a quality score `q_i` for each agent from its local point cloud density, and continuously re-weights LiDAR vs. camera features per agent — smoothly degrading from full-stack perception down to camera-only without hard switching.

Agents are predicted jointly via a **degradation-aware interaction module** adapted from HiVT [Zhou et al., CVPR 2022]. Standard cross-agent attention is extended to down-weight poorly-covered agents as context sources: `attn_{ij} ∝ exp(Q_i·K_j − λ·(1−q_j))`. Degraded agents contribute less to their neighbours' predictions — no existing interaction model accounts for per-agent sensor quality.

The trajectory head outputs a **distribution** `(μ_i, σ_i)` over future waypoints per agent. Uncertainty widens as `q_i → 0` — a model that is inaccurate but knows it is still useful for safe planning. We measure calibration with ECE and NLL alongside ADE/FDE.

**Cross-domain robustness validation:** The model is trained exclusively on real nuScenes data. Physics-accurate degraded point clouds are generated in NVIDIA Isaac Sim (fog, rain, 4-beam hardware, LiDAR failure) and used as a held-out test domain — no Isaac Sim data at training time.

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        FULL SYSTEM                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TRAINING                                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Real nuScenes: 32-beam LiDAR + 6 cameras + box detects  │    │
│  │  All dynamic agents: vehicles, pedestrians, cyclists      │    │
│  │  + Stochastic beam dropout: {32, 16, 8, 4, 2, 0} / batch │    │
│  │    0 beams per batch = camera-only training              │    │
│  └───────────────────────┬──────────────────────────────────┘    │
│                          │  N agents per scene simultaneously    │
│                          ▼                                       │
│  Per-Agent Encoding ── ResNet18 BEV + RoI Align · Box MLP       │
│  → (N, 256) embeddings  +  q_i per agent (local point density)  │
│  (Phase 1: CropEncoder; Phase 2: RoI Align on shared feat map)  │
│                          │                                       │
│                          ▼                                       │
│  Temporal Transformer ── 2-layer, 4-head, T=3 frames            │
│  → (N, 256) temporally-aggregated per-agent embeddings          │
│                          │                                       │
│                          ▼                                       │
│  Modality Gating ── per agent: q_i gates LiDAR vs camera        │
│  fused_i = q_i · f_LiDAR_i  +  (1−q_i) · f_camera_i           │
│                          │                                       │
│                          ▼                                       │
│  Agent-Agent Interaction ── HiVT context encoder [cited]        │
│  attn_{ij} = softmax(Q_i·K_j − λ·(1−q_j))   ← novel           │
│  degraded agents contribute less context to neighbours           │
│  → (N, 256) interaction-refined embeddings                      │
│                          │                                       │
│                          ▼                                       │
│  Trajectory Head ── (μ_i, log σ_i) × 6 steps = 3s per agent   │
│  σ_i widens as q_i → 0  ·  ECE-calibrated                      │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  INFERENCE  ─  N agents jointly, continuous graceful degradation │
│                                                                  │
│  32-beam        ──→     8-beam        ──→   camera-only          │
│  q_i ≈ 1.0              q_i ≈ 0.4–0.6       q_i ≈ 0.0           │
│  narrow σ_i             moderate σ_i         wide σ_i           │
│  high interaction trust  reduced trust        very uncertain     │
└──────────────────────────────────────────────────────────────────┘
```

## Starting Point

This project is a clean rewrite inspired by a prior temporal transformer velocity predictor ([baseline repo](https://github.com/srijanpal07/temporal-lidar-camera-bev-fusion)). The dataset is the same (nuScenes) and some ideas overlap, but the full pipeline is being rewritten from scratch for best results:

- **No pre-saved BEV files** — BEV rendered online from raw LiDAR at training time so beam dropout is applied before voxelization
- **Scene-level dataset** — all agents in a scene returned together (not one agent per sample)
- **Probabilistic trajectory head** — (μ, σ) per waypoint, not point-estimate velocity
- See [docs/implementation_plan.md](docs/implementation_plan.md) for the full build order

## Implementation Status

| Component | Status | Location |
|---|---|---|
| BEV encoder + temporal transformer (velocity head) | **Working** | `src/models/` |
| Camera-detector noise injection | **Working** | `src/data/box_noise.py` |
| Kalman + RTS smoother | **Working** | `src/utils/kalman.py` |
| LiDAR beam dropout pipeline (C1/C2) | **Stub** | `src/data/beam_degradation.py` |
| Stochastic dropout training (C2) | **Not wired** — awaits C1 | `configs/train_dropout.yaml` |
| Uncertainty-aware modality gating (C3) | **Stub** | `src/models/gating.py` |
| Trajectory head + NLL loss (C4, per-agent) | **Stub** | `src/models/heads.py` |
| RoI Align per-agent encoder (Phase 2) | **Stub** | `src/models/encoders.py` |
| Agent-agent interaction module (HiVT-adapted, C4) | **Stub** | `src/models/interaction.py` |
| Degradation-aware attention (novel λ penalty) | **Stub** | `src/models/interaction.py` |
| Multi-agent modality gating (B, N, feat_dim) | **Stub** | `src/models/gating.py` |
| Multi-category agent dataset (Phase 2) | **Not started** | filter_dataset.py expansion |
| Cross-domain Isaac Sim evaluation | **Stub** | `src/data/isaac_sim.py` |
| CLIP-based gating ablation | **Not started** | — |

## Project Structure

```
robust-bev-perception/
├── src/
│   ├── data/
│   │   ├── nuscenes_dataset.py  # ← Phase 1/2 scene-level dataset (to build)
│   │   ├── bev_renderer.py      # online BEV voxelizer from raw LiDAR (to build)
│   │   ├── quality_score.py     # per-agent q_i from local point density (to build)
│   │   ├── beam_degradation.py  # C1/C2 ring-index beam dropout (stub — rewrite)
│   │   ├── isaac_sim.py         # cross-domain eval dataset (stub)
│   │   ├── box_noise.py         # camera-detector noise (keep)
│   │   └── transforms.py        # BEV augmentation (stub)
│   ├── models/
│   │   ├── encoders.py     # ResNet18BEVEncoderWithFeatures, RoIAgentEncoder (P2 ready);
│   │   │                   # CropEncoder (P1); BEVEncoder, ResNet18BEVEncoder (legacy)
│   │   ├── gating.py       # MultiAgentModalityGating (P2 stub); ModalityGating (P1 stub)
│   │   ├── interaction.py  # AgentInteractionModule + DegradationAwareAttention (P2 stub)
│   │   ├── heads.py        # TrajectoryHead (μ, log σ) — stub, rewrite
│   │   └── temporal.py     # TemporalVelocityPredictor (legacy velocity model)
│   ├── training/           # losses.py · metrics.py · calibration.py · checkpointing.py
│   ├── eval/               # degradation_curves · forecasting_metrics · calibration_metrics
│   └── utils/              # kalman.py · geometry.py
│
├── scripts/
│   ├── train/
│   │   ├── train_p1.py     # ← Phase 1 training entrypoint (to build)
│   │   └── train_p2.py     # ← Phase 2 training entrypoint (to build)
│   ├── eval/               # eval_degradation_curve · eval_calibration · eval_isaac_sim
│   │
│   # Legacy (previous project — reference only)
│   └── train.py · infer.py · save_bev.py · filter_dataset.py · prepare_dataset.py
│
├── configs/                # YAML experiment configs
├── docs/
│   ├── overview.html          # visual research overview (open in browser)
│   └── implementation_plan.md # build order, phase definitions, code status
├── data/                   # symlink or mount point for nuScenes (~400GB)
└── outputs/                # training outputs (not tracked in git)
```

## Setup

```bash
bash setup.sh   # creates conda env 'bevrobust', installs torch 2.7.0+cu126, registers Jupyter kernel
```

See [docs/implementation_plan.md](docs/implementation_plan.md) for build order and what to implement next.

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
