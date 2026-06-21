---
name: project_context
description: Research project on calibrated multi-agent trajectory forecasting under LiDAR sensor degradation — repo at /home/beast/Repos/robust-bev-perception
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cb5a86e-ac46-4e67-80da-cf058c4016b6
---

## Active Research Project

**Local path:** `/home/beast/Repos/robust-bev-perception`
**Conda env:** `bevrobust` (Python 3.10, torch 2.7.0+cu126, all deps installed)
**Status:** Architecture expanded (June 2026) — multi-agent scene, interaction modeling added. No hard deadline — user willing to take the time needed.
**Target venues:** RA-L (rolling), CVPR 2027 (stretch), ICRA 2027 (fallback)

---

## Thesis Statement

A single model trained on real nuScenes data produces accurate, **calibrated** joint trajectory predictions for **all dynamic agents** in the scene across a continuous LiDAR degradation spectrum — by combining stochastic beam dropout training, per-agent uncertainty-aware modality gating, and degradation-aware agent-agent interaction modeling (adapted from HiVT, cited).

---

## Research Contributions

- **C1:** LiDAR degradation simulation — beam reduction (32→16→8→4→2→0) + Isaac Sim physics-accurate degradation (fog, rain, dust, hardware failure). Community benchmark deliverable.
- **C2:** Stochastic beam dropout training — sample {32,16,8,4,2,0} beams randomly per batch; bakes graceful degradation into the model.
- **C3:** Per-agent uncertainty-aware modality gating — quality score **q_i** per agent from local crop point density (not global scene score). Soft-weights LiDAR vs camera per agent. Calibrated probabilistic output (Gaussian over waypoints); ECE + NLL primary metrics.
- **C4:** Multi-agent trajectory forecasting with degradation-aware interaction:
  - All dynamic categories: vehicles, pedestrians, cyclists (Phase 1: cars only to establish baseline)
  - Agent-agent interaction from **HiVT [Zhou et al., CVPR 2022]** — cited, not claimed novel
  - **Novel extension:** degradation-aware attention: `attn_{ij} = softmax(Q_i·K_j − λ·(1−q_j))` — poorly-covered agents contribute less context to neighbours. No existing interaction model does this.
  - (μ_i, log σ_i) × 6 steps per agent; σ_i widens from both own q_i and context agents' q_j
- **Ablation (exploratory):** CLIP ViT-L/14 LoRA on Isaac Sim images as alternative gating signal — only promote if clearly beats geometric baseline.

## Dataset Scope (phased)
- **Phase 1:** vehicle.car only, existing filters — establish C1–C3 baseline
- **Phase 2:** All dynamic categories; remove speed filter (>0.5 m/s); reduce min-track-length from 40 to ~12 frames

## Key Novelty Claims
1. Continuous beam spectrum (not binary modality switch) — vs Grace-BEV, MoME
2. Per-agent q_i (not global scene quality) — vs Cocoon, FDSNet
3. Degradation-aware attention in interaction graph — **no prior work does this**
4. Trajectory ADE/FDE + ECE/NLL under degradation — vs detection mAP only in Grace-BEV

## Key Related Work
- **Grace-BEV** (arXiv:2605.30983, May 2026) — closest competitor: binary modality dropout, mAP only, no interaction, no calibration
- **Sensor-Fault Forecasting Benchmark** (arXiv:2605.10822) — read before finalizing evaluation protocol
- **HiVT** (CVPR 2022) — interaction module we adapt (cite, don't claim)
- **MTR, Wayformer** — clean-input forecasting baselines to compare against
- **MoME** (CVPR 2025) — discrete MoE gating; we use soft per-agent continuous gating

## Baseline Architecture
- BEV encoder (ResNet18 500×500) + CropEncoder (64×64) + Box MLP → 448-dim per agent per frame
- 2-layer 4-head Temporal Transformer → 256-dim per agent
- T_MODEL vs T_kf distinction: BEV steps vs keyframe count — prior inference crash from mismatch, still relevant
- Camera-detector noise injected in dataset (distance-dependent, non-Gaussian)
- Kalman + RTS smoother for kinematic velocity features

## Phase 1 Implementation (as of June 2026)

Phase 1 model is implemented and the codebase was significantly cleaned up (old scripts/configs removed):

**Current model files:**
- `src/models/phase1_model.py` — top-level Phase 1 model
- `src/models/encoders.py` — BEV + crop + box encoders
- `src/models/gating.py` — per-agent quality score gating
- `src/models/interaction.py` — degradation-aware agent-agent interaction (HiVT-adapted)
- `src/models/heads.py` — trajectory prediction heads

**Removed (no longer exist):**
- `src/model.py`, `src/models/temporal.py`, `src/kalman.py`
- `scripts/train.py`, `scripts/train/train.py` (old entrypoints — gone)
- `configs/train.yaml`, `configs/train_baseline.yaml`, `configs/train_dropout.yaml`
- Old pipeline scripts: `infer.py`, `filter_dataset.py`, `prepare_dataset.py`, `save_bev.py`, `visualize.py`, BEVFormer scripts, etc.

## Hardware & Environment
- GPU: NVIDIA RTX 5000 Ada (32.8 GB VRAM), CUDA driver 12.8
- Conda env: `bevrobust` (torch 2.7.0+cu126, numpy 1.26.4 pinned for nuscenes-devkit compat)
- Isaac Sim: `env_isaacsim` (separate env, do not merge)
- Simulator: NVIDIA Isaac Sim 5.1.0

## Canonical Entrypoints (current)
- Training: `scripts/train/train_p1.py --config configs/train_p1.yaml`
- Eval: `scripts/eval/eval_calibration.py`, `eval_degradation_curve.py`, `eval_isaac_sim.py`
