# Research Plan: Robust BEV Perception under Sensor Degradation

## Thesis Statement

A single model trained on high-quality LiDAR + camera + radar can maintain robust velocity and trajectory prediction across a continuous sensor degradation spectrum at inference — by training with stochastic LiDAR beam dropout and uncertainty-aware modality gating — closing the training-inference gap for cost-constrained and adverse-weather AV platforms.

---

## Motivation: The Industry Problem

The AV sensor debate is real and unresolved as of 2025:
- **Waymo** uses 5 LiDARs + 6 radars + 13 cameras per vehicle (~$12,700 sensor cost)
- **Tesla** uses cameras only (~$400 sensor cost) — no LiDAR, no radar
- **Mobileye, Cruise, Continental** bet on 4D imaging radar as the cost-effective safety net

The core unresolved problem: **models trained on rich sensor data fail unpredictably when sensors degrade**. LiDAR loses 60–80% of returns in heavy fog. Cameras blind in direct sunlight or dust storms (Waymo demonstrated this at Google I/O 2025). No existing perception framework handles this gracefully.

---

## Research Contributions

### C1: LiDAR Degradation Simulation Pipeline
- Beam reduction on nuScenes 32-beam HDL-32E → 16, 8, 4, 2 beam variants
- Simple: keep every N-th elevation ring in the range image
- Physics-accurate: NVIDIA Isaac Sim with configurable LiDAR sensor models (beam count, noise, weather)
- Deliverable: a released benchmark for nuScenes degradation levels (useful to the community)

### C2: Multi-Resolution LiDAR Training (stochastic beam dropout)
- Randomly sample LiDAR quality per training batch: {32, 16, 8, 4, 0 beams}
- BEV encoder learns to extract useful features from variable-density point clouds
- Train on high-quality nuScenes LiDAR, generalize to low-cost deployment
- Key claim: training diversity alone buys graceful degradation at inference

### C3: Uncertainty-Aware Modality Gating
- Lightweight degradation-predictor head: estimates per-modality quality score q ∈ [0,1]
- Soft modality weighting: blend LiDAR BEV features with camera BEV features by q
- Avoids hard switching (brittle boundary) and full MoE complexity
- Degrades to camera-only when LiDAR score → 0, to radar fallback in adverse weather

### C4: Velocity → Trajectory Extension
- Extend output head from (vx, vy) to future waypoints [(x, y) × T_future]
- Benchmark directly on nuScenes motion forecasting task
- Comparable to MTR, Wayformer, HiVT — but with noisy/degraded sensor robustness they lack

---

## Short-Term Goals (Month 1 — Foundation)

- [ ] **Week 1–2:** Extend existing model from velocity prediction to multi-step trajectory (waypoint head). Run on nuScenes motion forecasting split. Establish baseline numbers.
- [ ] **Week 3:** Implement beam-reduction pipeline. Plot velocity/trajectory error vs. beam count (32→16→8→4→2→0). This curve motivates everything.
- [ ] **Week 4:** Isaac Sim setup — configure a nuScenes-like urban scene, calibrate 32-beam and 4-beam LiDAR sensor models, generate 500–1000 labeled scenarios.

---

## Medium-Term Goals (Month 2 — Core Contribution)

- [ ] **Week 5–6:** Stochastic beam dropout training. Randomly sample LiDAR quality per batch. Retrain and re-plot degradation curve vs. Month 1 baseline. Show improvement.
- [ ] **Week 7:** Uncertainty-aware gating network. Train lightweight head to predict point cloud density/quality. Soft-weight LiDAR vs. camera BEV features.
- [ ] **Week 8:** Isaac Sim cross-domain evaluation. Train on nuScenes real data, test on Isaac Sim physics-accurate degraded scans. Cross-domain generalization test.

---

## Long-Term Goals (Month 3 — Polish + Paper)

- [ ] **Week 9–10:** Full ablation study — each component in isolation. Degradation curves. Comparison to: (1) no-dropout baseline, (2) hard-switch baseline, (3) Cocoon-style soft fusion.
- [ ] **Week 11:** Edge case experiments — complete LiDAR failure, night driving (camera only), heavy fog (radar only), dust storm. Show the system handles all gracefully.
- [ ] **Week 12:** Paper writing. Introduction, related work, method, experiments, conclusion.

---

## Future Extensions (Beyond 3 Months)

- **4D Radar integration:** nuScenes has classical 3D radar (5 units). Extend to simulated 4D imaging radar in Isaac Sim as a weather-robust fallback. 4D radar provides native Doppler velocity and survives fog/rain where LiDAR fails.
- **Full world model:** Replace the BEV encoder with a generative world model that hallucinates missing modalities (e.g., predict what LiDAR would see given camera + motion history). BEVWorld is the closest reference.
- **Occupancy flow prediction:** Extend from per-agent trajectory to dense BEV occupancy forecasting. Comparable to Cam4DOcc benchmark.
- **Real adverse weather validation:** Collect or license data with annotated adverse weather (K-Radar, View-of-Delft datasets).

---

## Target Venues

| Venue | Deadline (approx.) | Fit | Notes |
|---|---|---|---|
| **IROS 2026** | March 2026 | Strong | Robotics + perception, sensor robustness |
| **ECCV 2026** | March 2026 | Strong | LEROjD (most related work) was at ECCV 2024 |
| **CoRL 2026** | June 2026 | Strong | Real deployment + sensor robustness story |
| **CVPR 2026 Workshop** | March 2026 | Good | Lower bar, good first submission |
| **ICRA 2027** | Sep 2026 | Strong | Fallback if IROS/ECCV miss |

**Realistic target:** IROS 2026 or ECCV 2026 (both March 2026 deadline, ~9 months away).

---

## Novelty vs. Prior Work

| Claim | Prior art | Our delta |
|---|---|---|
| Train rich LiDAR, infer weak sensor | LEROjD (radar) | Continuous quality spectrum, not binary switch |
| Soft modality gating | Cocoon, FDSNet | Learned degradation predictor, not just feature disagreement |
| Trajectory prediction under degradation | MTR, Wayformer | They assume clean detection; we handle noisy/missing LiDAR |
| Cross-domain Isaac Sim validation | None in this area | Novel evaluation: physics-accurate degradation with GT |
| 4D radar as LiDAR fallback in weather | Separate radar papers | Combined within a degradation-aware unified framework |

---

## Evaluation Plan

**Primary metric:** ADE (Average Displacement Error) and FDE (Final Displacement Error) at 1s, 2s, 3s horizons — standard nuScenes motion forecasting metrics.

**Key experiment:** Degradation curves — plot ADE/FDE vs. LiDAR beam count (32→16→8→4→2→0 beams) for:
1. Baseline (no dropout training)
2. Ours (stochastic dropout training)
3. Ours + gating network

The gap between curves 1 and 2/3 is the paper's main result.

**Secondary experiments:**
- Adverse weather (Isaac Sim): fog, rain, dust — compare to hard-switch baseline
- Ablations: dropout only, gating only, both
- Cross-domain: train nuScenes, test Isaac Sim

---

## Related Work (Key Papers)

**Closest analogues:**
- [LEROjD (ECCV 2024)](https://arxiv.org/abs/2409.05564) — LiDAR → radar-only transfer
- [BEVWorld (2024)](https://arxiv.org/html/2407.05679v3) — multimodal world model, BEV latent
- [Cocoon (2024)](https://arxiv.org/html/2410.12592v1) — uncertainty-aware sensor fusion

**Trajectory prediction:**
- [MTR++ (2024)](https://arxiv.org/html/2306.17770v2) — motion transformer with intention queries
- [Wayformer](https://waymo.com/research/wayformer) — attention-based motion forecasting
- [LiDAR MOT-DETR (2025)](https://arxiv.org/html/2505.12753v2) — temporal transformer for tracking

**Occupancy / world models:**
- [Cam4DOcc (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Cam4DOcc_Benchmark_for_Camera-Only_4D_Occupancy_Forecasting_in_Autonomous_Driving_CVPR_2024_paper.pdf)
- [DriveTransformer (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/a7afc9957f1190223763b6ea93218f98-Paper-Conference.pdf)
- [DriveMoE (2025)](https://arxiv.org/pdf/2505.16278)

**Sensor degradation:**
- [MSC-Bench (2025)](https://arxiv.org/html/2501.1037) — multi-sensor corruption benchmark
- [Sparse Points to Dense Clouds (2024)](https://arxiv.org/html/2404.06715v1)
- [Survey on Sensor Failures in AVs](https://pmc.ncbi.nlm.nih.gov/articles/PMC11360603/)

---

## Hardware & Tools

- **GPU:** NVIDIA RTX Pro 5000 Ada (~32GB VRAM) — can run large BEV models
- **Simulator:** NVIDIA Isaac Sim (Omniverse) — physics-accurate LiDAR, camera, radar simulation
- **Dataset:** nuScenes (32-beam LiDAR, 6 cameras, 5 radars, 1000 scenes)
- **Baseline codebase:** `src/` — temporal transformer velocity predictor

---

## Notes

- nuScenes uses Velodyne HDL-32E (32 beams, ~40K points/scan)
- Beam reduction: keep every (32/N)-th elevation ring to simulate N-beam LiDAR
- nuScenes radar is classical 3D (range, azimuth, Doppler) — 4D radar requires Isaac Sim or View-of-Delft dataset
- T_MODEL vs T_kf distinction in baseline code: BEV steps vs keyframe count — prior inference crash from mismatch, keep this in mind when modifying the temporal architecture
