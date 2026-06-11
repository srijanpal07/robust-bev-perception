# Research Plan: Robust BEV Perception under Sensor Degradation

## Thesis Statement

A single model trained on high-quality LiDAR + camera data produces accurate, **calibrated** joint trajectory predictions for all dynamic agents in the scene — across a continuous sensor degradation spectrum — by combining stochastic LiDAR beam dropout training, per-agent uncertainty-aware modality gating, and degradation-aware agent-agent interaction modeling. The system closes the training-inference gap that causes silent perception failure when sensors degrade in adverse weather or hardware wear, validated via cross-domain robustness evaluation on physics-accurate Isaac Sim degradation.

---

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
│  │  + Stochastic beam dropout (C2): {32, 16, 8, 4, 2, 0}    │    │
│  │    sampled per batch — 0 beams = camera-only training    │    │
│  └───────────────────────┬──────────────────────────────────┘    │
│                          │  N agents per scene simultaneously    │
│                          ▼                                       │
│  Per-Agent Encoding (C1+C2)                                      │
│  ResNet18 BEV + Box MLP → (N, 256) embeddings                   │
│  Phase 1: CropEncoder (64×64 crop per agent)                    │
│  Phase 2: RoI Align on shared BEV feat map (O(1) encoder cost)  │
│  q_i = per-agent quality score (local point density in crop)    │
│                          │                                       │
│                          ▼                                       │
│  Temporal Transformer ── 2-layer, 4-head, T=3 frames            │
│  → (N, 256) temporally-aggregated per-agent embeddings          │
│                          │                                       │
│                          ▼                                       │
│  Modality Gating (C3) ── per-agent: q_i gates LiDAR vs camera  │
│  fused_i = q_i · f_LiDAR_i  +  (1−q_i) · f_camera_i           │
│                          │                                       │
│                          ▼                                       │
│  Agent-Agent Interaction ── HiVT local context encoder [cited]  │
│  attn_{ij} = softmax(Q_i·K_j − λ·(1−q_j))  ← novel extension  │
│  degraded agents (low q_j) contribute less context to neighbours│
│  → (N, 256) interaction-refined embeddings                      │
│                          │                                       │
│                          ▼                                       │
│  Trajectory Head (C4) ── (μ_i, log σ_i) × 6 steps per agent   │
│  NLL training · σ_i widens as q_i → 0 · ECE-calibrated         │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  INFERENCE  ─  N agents jointly, continuous graceful degradation │
│                                                                  │
│  32-beam        ──→     8-beam        ──→   camera-only          │
│  q_i ≈ 1.0              q_i ≈ 0.4–0.6       q_i ≈ 0.0           │
│  narrow σ_i             moderate σ_i         wide σ_i           │
│  high interaction trust  reduced trust        very uncertain     │
│                                                                  │
│  RESULT: one model, all agents jointly, graceful degradation     │
│  across full beam-count spectrum — no mode switching             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Motivation: The Industry Problem

The AV industry has bifurcated into two camps as of 2026:
- **Waymo** (Gen 6): 13 cameras + 4 LiDARs + 6 radars, ~$200K vehicle cost. Deployed in 10+ US cities. Added mechanical cleaning, hydrophobic coatings, and modular sensors specifically to handle weather-induced degradation.
- **Tesla**: cameras only, no LiDAR. Commercially operating. Argues scale + neural networks obviates sensor redundancy.

**Neither camp has solved the core reliability problem.** LiDAR loses 60–80% of returns in heavy fog. Cameras lose contrast in dust storms and direct glare. Waymo's Gen 6 invests heavily in mechanical mitigation precisely because their software stack does not degrade gracefully when a sensor is partially compromised — it needs good data. Tesla's approach avoids the problem by removing LiDAR entirely, but at the cost of geometric precision.

The training-inference gap is the specific, unsolved problem this work targets: **models trained on clean 32-beam data fail silently and unpredictably when deployed sensors degrade continuously**, whether from weather, hardware wear, or cost-tier hardware. No existing perception framework provides a principled, trained-in graceful degradation path. Foundation models and scale alone do not fix this — they still require training data that covers the degradation distribution, which is exactly what this work provides via stochastic dropout and physics-accurate simulation.

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

### C3: Uncertainty-Aware Modality Gating with Calibrated Output
- Lightweight degradation-predictor head: estimates per-modality quality score q ∈ [0,1] from point cloud statistics (density, intensity variance, range histogram)
- Soft modality weighting: blend LiDAR BEV features with camera BEV features by q
- Avoids hard switching (brittle boundary) and full MoE complexity
- Degrades to camera-only when LiDAR score → 0
- **Calibrated uncertainty output:** the prediction head outputs a distribution over future waypoints (Gaussian or GMM), not just a point estimate. Uncertainty should widen correctly as sensor quality drops — this is the "calibrated graceful degradation" claim. Measured via Expected Calibration Error (ECE) and NLL alongside ADE/FDE.

### C4: Multi-Agent Trajectory Forecasting with Degradation-Aware Interaction

**Scene coverage (phased):**
- Phase 1: `vehicle.car` only — establish degradation baseline, validate C1–C3
- Phase 2: All dynamic categories (`vehicle.*`, `human.pedestrian.*`, `vehicle.bicycle`) with relaxed filters:
  - Remove speed threshold (stationary agents that may become dynamic are valid targets)
  - Reduce min-track-length from 40 frames to ~12 (= T_context + T_future + buffer)
  - Keep front-camera visibility for now; revisit with multi-camera detection

**Per-agent encoding:**
- Each of N agents in the scene is encoded independently: BEV crop + box params → 256-dim embedding
- Per-agent quality score `q_i` estimated from local point density in that agent's BEV crop — not a global scene score. Occluded or sparsely-covered agents have lower q_i even under clean global LiDAR.

**Agent-agent interaction — adapted from HiVT [Zhou et al., CVPR 2022]:**
- HiVT's local context encoder performs cross-agent attention over all N agents
- Standard attention: `attn_{ij} = softmax(Q_i · K_j)`
- **Novel extension — degradation-aware attention:**
  ```
  attn_{ij} = softmax(Q_i · K_j − λ · (1 − q_j))
  ```
  Agent j's quality score q_j down-weights how much j's features are used as context by agent i. A poorly-covered agent (low q_j) contributes less to its neighbours' predictions. No existing interaction model accounts for per-agent sensor quality.
- We cite HiVT for the interaction architecture; the degradation-aware attention weighting is novel.

**Calibrated joint output:**
- `(μ_i, log σ_i) × T_future` per agent — same probabilistic head as C3
- σ_i widens when (a) agent i's own q_i is low, and (b) the agents it attended to have low q_j — uncertainty propagates through the interaction graph

---

## Ablation: Language-Grounded Gating (exploratory, not a core claim)

This is an interesting idea to test once the baseline (C1–C4) is working and strong:
- A CLIP-based model (ViT-L/14, LoRA fine-tuned) takes front-camera frames and outputs a semantic degradation embedding, which is compared against: (a) a simple learned MLP on point cloud density, (b) a hand-engineered density score
- **Only promote to a contribution if results clearly show CLIP gating > simpler alternatives.** The honest reviewer question is "why not a simpler weather classifier?" — answer that in the ablation table before claiming novelty.
- Training data: Isaac Sim paired (camera render, degradation label) — zero annotation cost
- If CLIP wins: frame as "language-grounded quality estimation enables zero-shot generalization to unseen degradation types." If it doesn't: drop it, keep geometric gating.

---

## Implementation Goals (No Fixed Timeline)

The work is organised into three phases. No hard deadlines — take the time needed to get the results right. See [docs/implementation_plan.md](docs/implementation_plan.md) for the detailed build order and code status.

### Phase 1 — Single-Agent Baseline (C1–C3)
- [ ] Online BEV renderer from raw LiDAR (replaces pre-saved BEV files)
- [ ] Beam dropout pipeline — ring-index filter on raw point cloud, {32, 16, 8, 4, 2, 0}
- [ ] nuScenes scene-level dataset using nuScenes-devkit natively
- [ ] Per-agent quality score `q_i` from local point density
- [ ] Modality gating wired into training loop
- [ ] Trajectory head (μ, log σ) + NLL loss + ECE metric
- [ ] **Key result:** degradation curve — ADE/FDE + ECE vs beam count {32→0}. This is Figure 1 of the paper.

### Phase 2 — Multi-Agent + Interaction (C4)
- [ ] Expand dataset to all dynamic categories; relax filters
- [ ] Wire RoI Align encoder (replaces CropEncoder for scale)
- [ ] Wire AgentInteractionModule into training loop
- [ ] Multi-agent collate_fn with variable N + padding mask
- [ ] **Key result:** interaction ablation — with vs without degradation-aware attention

### Phase 3 — Cross-Domain Validation (Isaac Sim)
- [ ] Configure Isaac Sim sensor params (HDL-32E: 32 beams, 10 Hz, beam angles)
- [ ] Generate physics-accurate degraded scenes (fog, rain, 4-beam, LiDAR failure)
- [ ] Run Phase 2 model zero-shot on Isaac Sim outputs — no fine-tuning
- [ ] **Key result:** cross-domain ADE/FDE + ECE — does robustness transfer beyond beam subsampling?
- [ ] Early task (can do before Phase 2): lock HDL-32E sensor parameters in a config file

---

## Future Extensions

- **4D Radar integration:** nuScenes has classical 3D radar (5 units). Extend to simulated 4D imaging radar in Isaac Sim as a weather-robust fallback. 4D radar provides native Doppler velocity and survives fog/rain where LiDAR fails.
- **Full world model:** Replace the BEV encoder with a generative world model that hallucinates missing modalities (e.g., predict what LiDAR would see given camera + motion history). BEVWorld is the closest reference.
- **Occupancy flow prediction:** Extend from per-agent trajectory to dense BEV occupancy forecasting. Comparable to Cam4DOcc benchmark.
- **Real adverse weather validation:** Collect or license data with annotated adverse weather (K-Radar, View-of-Delft datasets).
- **Learned interaction graph under degradation:** Currently the interaction graph topology is fixed (fully-connected within radius). Future work: learn which agent-agent edges to prune based on joint sensor coverage — two agents both poorly covered should have their interaction edge down-weighted more aggressively than the simple `q_j` penalty.

---

## Target Venues

> **Status as of June 2026:** CVPR 2026, IROS 2026, and ECCV 2026 deadlines have all passed. Realistic upcoming targets:

| Venue | Deadline (approx.) | Fit | Notes |
|---|---|---|---|
| **RA-L** (IEEE Robotics & Automation Letters) | Rolling — submit Aug–Oct 2026 | **Primary** | Rolling deadline, fast review (~3 months). Sensor robustness + trajectory prediction is core RA-L territory. Accepted RA-L papers are presented at ICRA/IROS. |
| **CoRL 2026** | ~June–July 2026 | Strong | Check current deadline. Real deployment + calibrated graceful degradation + trajectory forecasting fits CoRL well. |
| **CVPR 2027** | ~Nov 2026 | Strong | Highest visibility. Calibrated robustness + trajectory forecasting under degradation fits the perception + planning convergence trend. Requires paper-ready by Oct 2026. |
| **ICRA 2027** | ~Sep 2026 | Strong | Fallback to RA-L; or submit simultaneously (RA-L → ICRA presentation). |
| **ICCV 2027** | ~March 2027 | Good | More time to mature the VLM component. Strong venue for perception + vision. |
| **IROS 2027** | ~March 2027 | Strong | Robotics + sensor reliability + real deployment focus. |

**Recommended path:** Submit to **RA-L** when results are solid (target: September 2026). This gets you reviewed fast and into ICRA 2027 as a presentation vehicle. Use the CVPR 2027 deadline (Nov 2026) as a parallel stretch target if the VLM component is strong enough by then.

---

## Novelty vs. Prior Work

| Claim | Prior art | Our delta |
|---|---|---|
| Train rich LiDAR, infer weak sensor | LEROjD (radar) | Continuous quality spectrum, not binary switch |
| Multi-agent trajectory forecasting under degradation | MTR, Wayformer, HiVT — clean input only; EgoTraj-Bench — ego noise only | First work connecting structured hardware degradation (beam loss) → multi-agent ADE/FDE |
| Calibrated uncertainty under degradation | All prior forecasting work reports ADE/FDE only | ECE + NLL at each degradation level: model knows when it is unreliable |
| Per-agent quality estimation | Global scene-level quality only (Cocoon, FDSNet) | Per-agent q_i from local crop density — occluded agents get lower q even under clean global LiDAR |
| Agent-agent interaction | HiVT, MTR, Wayformer — all assume equally reliable agent features | Degradation-aware attention: `attn_{ij} ∝ exp(Q_i·K_j − λ·(1−q_j))` — first interaction model to account for per-agent sensor quality |
| Soft modality gating | Cocoon, FDSNet | Per-agent geometric quality score with soft continuous weighting (not discrete MoE) |
| Cross-domain robustness evaluation | RESBev (nuScenes only), Sensor-Fault Forecasting Benchmark (2026) | Physics-accurate Isaac Sim degradation (fog, rain, 4-beam hardware) as held-out test domain |
| BEV robustness mechanism | RESBev (post-hoc latent recovery) | Training policy (stochastic dropout) — baked into the model, not a plug-in wrapper |

**Key positioning against Grace-BEV (arXiv May 2026, the closest competitor):**
Grace-BEV ["Can BEV Perception Gracefully Degrade under Sensor Failures?"](https://arxiv.org/abs/2605.30983) proposes a plug-in TrustGate Router + FailSafe Fusion Block with 3-phase binary modality dropout training, evaluated on nuScenes-R/C with mAP. Three concrete gaps our work fills:
1. **Binary vs. continuous degradation:** Grace-BEV drops full modalities (LiDAR present or absent). We operate on a continuous beam-count spectrum {32, 16, 8, 4, 2, 0} — the realistic hardware failure mode is gradual quality loss, not sudden disappearance.
2. **Detection vs. trajectory:** Grace-BEV reports mAP. We report ADE/FDE — the output a motion planner actually consumes. Robust detection that produces inaccurate tracks is not enough.
3. **Calibrated uncertainty:** Grace-BEV reports no uncertainty. We measure ECE and NLL — a planner needs to know *when* to trust the prediction, not just whether it is accurate on average.

---

## Evaluation Plan

**Primary metrics:**
- ADE / FDE at 1s, 2s, 3s — standard nuScenes motion forecasting metrics
- **ECE (Expected Calibration Error)** — measures whether predicted uncertainty is calibrated to actual error. A model that outputs high uncertainty on 2-beam inputs and low uncertainty on 32-beam inputs is calibrated; one that outputs the same confidence regardless of sensor quality is not.
- **NLL (Negative Log-Likelihood)** — proper scoring rule for probabilistic predictions

Together these answer: *is the model accurate AND does it know when it isn't?*

**Key experiment — Degradation Curves (nuScenes, Figure 1):**
Plot ADE/FDE and ECE vs. LiDAR beam count {32, 16, 8, 4, 2, 0} for:
1. Baseline — clean-trained, no dropout (collapses at low beams)
2. + Stochastic dropout training (C2)
3. + Modality gating (C2 + C3)

The gap between curve 1 and curve 3 across all metrics is the paper's main result.

**Cross-domain robustness evaluation (Isaac Sim):**
- Train exclusively on real nuScenes (no Isaac Sim data at training time)
- Generate physics-accurate degraded point clouds in Isaac Sim: fog, rain, 4-beam hardware LiDAR, complete LiDAR failure
- Run inference on Isaac Sim outputs — no fine-tuning, no Isaac Sim training data
- Report ADE/FDE + ECE on Isaac Sim across degradation conditions
- Answers: does stochastic dropout training generalize to physically simulated degradation beyond artificially subsampled beams?
- **Do not call this "sim-to-real."** Call it "cross-domain robustness validation" or "physics-accurate degradation evaluation."

**Ablation table:**
| Method | ADE↓ | FDE↓ | ECE↓ | NLL↓ |
|---|---|---|---|---|
| Baseline (clean) | | | | |
| + Stochastic dropout (C2) | | | | |
| + Soft gating, geometric (C3) | | | | |
| + Hard switch (MoME-style) | | | | |
| + CLIP gating (exploratory) | | | | |

**Secondary experiments:**
- Edge cases on Isaac Sim: fog only, rain only, dust storm, complete LiDAR blackout
- Calibration reliability: does ECE worsen gracefully or collapse? A planner that receives widening uncertainty under degradation can respond conservatively — this is the "graceful" in graceful degradation.

---

## Related Work (Key Papers)

**Closest analogues — must cite and differentiate:**
- [Grace-BEV (arXiv May 2026)](https://arxiv.org/abs/2605.30983) — **most direct competitor**: TrustGate Router + FailSafe Fusion + binary modality dropout training; mAP on nuScenes-R/C; gaps: binary not continuous spectrum, detection not trajectory, no calibrated uncertainty
- [LEROjD (ECCV 2024)](https://arxiv.org/abs/2409.05564) — LiDAR-train / radar-infer transfer; binary switch, not continuous spectrum
- [MetaBEV (2023)](https://arxiv.org/abs/2304.09801) — sensor failure for BEV detection/segmentation; detection mAP focus, not trajectory + calibration
- [MoME (CVPR 2025)](https://arxiv.org/abs/2503.19776) — discrete per-query expert routing for sensor failure; we use soft per-modality gating with calibrated uncertainty
- [RESBev (arXiv Mar 2026)](https://arxiv.org/abs/2603.09529) — post-hoc latent-space BEV recovery; we bake robustness into training policy
- [Benchmarking Sensor-Fault Robustness in Forecasting (arXiv May 2026)](https://arxiv.org/abs/2605.10822) — **directly relevant**: benchmarks forecasting under sensor faults; read before finalizing evaluation protocol and Table 1
- [BEVWorld (2024)](https://arxiv.org/html/2407.05679v3) — multimodal world model in BEV latent space
- [Cocoon (2024)](https://arxiv.org/html/2410.12592v1) — uncertainty-aware sensor fusion

**Trajectory prediction:**
- [MTR++ (2024)](https://arxiv.org/html/2306.17770v2) — motion transformer with intention queries; assumes clean LiDAR
- [Wayformer](https://waymo.com/research/wayformer) — attention-based motion forecasting; clean input assumed
- [EgoTraj-Bench (arXiv Oct 2025)](https://arxiv.org/abs/2510.00405) — robust trajectory under ego-view noise; ours addresses hardware/weather degradation, not ego noise

**Sensor degradation & adverse weather:**
- [MSC-Bench (2025)](https://arxiv.org/html/2501.1037) — multi-sensor corruption benchmark
- [4D Radar Meets LiDAR + Camera (CVPR 2026 Workshop)](https://arxiv.org/abs/2606.00416) — physics-based LiDAR degradation + 4D radar fallback; compare our Isaac Sim pipeline to their OPV2V-R/Adver-City-R benchmarks

**VLMs for AV (context only — follow up if CLIP ablation shows strong results):**
- [AUTOPILOT Workshop @ CVPR 2026](https://www.autopilot-cvpr.net/) — VLMs for safety-critical AV perception
- [Foundation Models for AV Perception Survey (2025)](https://arxiv.org/html/2509.08302v1) — CLIP increasingly used for scene understanding, not yet for sensor quality estimation — potential novelty if ablation confirms it

---

## Hardware & Tools

- **GPU:** NVIDIA RTX Pro 5000 Ada (~32GB VRAM) — can run large BEV models
- **Simulator:** NVIDIA Isaac Sim (Omniverse) — physics-accurate LiDAR, camera, radar simulation
- **Dataset:** nuScenes (32-beam LiDAR, 6 cameras, 5 radars, 1000 scenes)
- **Codebase:** full rewrite from scratch — nuScenes-devkit native, online LiDAR loading, scene-level dataset. See [docs/implementation_plan.md](docs/implementation_plan.md).

---

## Notes

- nuScenes uses Velodyne HDL-32E (32 beams, ~40K points/scan)
- Beam reduction: keep every (32/N)-th elevation ring to simulate N-beam LiDAR
- nuScenes radar is classical 3D (range, azimuth, Doppler) — 4D radar requires Isaac Sim or View-of-Delft dataset
- T_MODEL vs T_kf: BEV steps vs keyframe count — was a source of crashes in the old pipeline. The new nuScenes-native dataset will handle this explicitly via keyframe timestamps.
