---
name: project_context
description: Research project on graceful sensor degradation for AV perception — new repo at /home/beast/Documents/robust-bev-perception
metadata: 
  node_type: memory
  type: project
  originSessionId: 44b2c07f-492d-4c21-bf73-c1167f824fa3
---

## Active Research Project

**Local path:** `/home/beast/Documents/robust-bev-perception`
**GitHub:** to be created at `github.com/srijanpal07/robust-bev-perception` (not pushed yet as of 2026-06-09)
**Status:** Repo initialized, baseline committed, research plan written. Next: push to GitHub, then start Month 1 work.
**Timeline:** 3+ months, targeting IROS 2026 or ECCV 2026 (March 2026 deadline).

**Why:** Real industry problem — LiDAR expensive, degrades in weather, creates training-inference gap. Goal: single model that degrades gracefully across full sensor quality spectrum.

---

## Archived Course Project

**Local path:** `/home/beast/Documents/CSCI 5527/csci5527_final_project`
**GitHub:** `github.com/srijanpal07/temporal-lidar-camera-bev-fusion` (fully mirrored, independent of original contributor Jackbaude)
**Status:** SUBMITTED and archived. Serves as baseline for research project.

---

## Research Thesis

"A single model trained on high-quality LiDAR + camera + radar can maintain robust velocity/trajectory prediction across a continuous sensor degradation spectrum at inference — via stochastic LiDAR beam dropout training and uncertainty-aware modality gating — closing the training-inference gap for cost-constrained AV platforms."

---

## Baseline Architecture (carried into new repo)

- BEV encoder (ResNet18) + crop encoder + 2-layer 4-head transformer temporal model
- Box params: [x,y,z,l,w,h,yaw] + [dt, vx̂, vŷ] (kinematic features)
- Camera-detector noise injected at runtime in dataset.py (distance-dependent, non-Gaussian)
- Kalman filter in src/kalman.py for kinematic velocity from noisy positions
- Residual velocity output: model predicts (GT_vel − noisy_vhat)
- T_MODEL vs T_kf distinction: BEV steps vs keyframe count — prior inference crash from mismatch

---

## Research Contributions (C1–C4)

- **C1:** LiDAR degradation simulation — beam reduction (32→16→8→4→2 beams) + Isaac Sim physics-accurate degradation
- **C2:** Stochastic beam dropout training — randomly sample LiDAR quality per batch
- **C3:** Uncertainty-aware modality gating — lightweight degradation predictor, soft modality weighting
- **C4:** Velocity → trajectory extension — multi-step waypoint prediction, nuScenes motion forecasting benchmark

---

## Hardware

- GPU: NVIDIA RTX Pro 5000 Ada (~32GB VRAM)
- Simulator: NVIDIA Isaac Sim (Omniverse) — installed and ready

---

## Month 1 Goals (immediate next steps)

1. Extend velocity head → trajectory (waypoints × T_future), benchmark on nuScenes
2. Implement beam-reduction pipeline, plot error vs. beam count (the key motivating curve)
3. Isaac Sim: configure LiDAR sensor models, generate 500–1000 degraded scenarios

---

## Key Related Work

- LEROjD (ECCV 2024): closest analogue — LiDAR-train / radar-infer
- BEVWorld (2024): multimodal world model in BEV latent space
- Cocoon (2024): uncertainty-aware sensor fusion
- DriveMoE (2025): MoE routing for AV
- nuScenes: 32-beam HDL-32E LiDAR, 6 cameras, 5 classical 3D radars

---

## How to apply

- When user asks about model: check new repo at `/home/beast/Documents/robust-bev-perception/src/`
- When suggesting experiments: degradation curves (ADE/FDE vs beam count) are the key evaluation
- When scoping: generative hallucinator is PhD-scope; stochastic dropout + gating is conference-scope
- T_MODEL vs T_kf still relevant in new codebase (same architecture carried over)
