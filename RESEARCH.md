# Research Plan: Robust BEV Perception under Sensor Degradation

## Thesis Statement

A single model trained on high-quality LiDAR + camera + radar can maintain robust velocity and trajectory prediction across a continuous sensor degradation spectrum at inference — by training with stochastic LiDAR beam dropout, uncertainty-aware modality gating driven by language-grounded sensor quality estimation, and zero-shot sim-to-real transfer validation — closing the training-inference gap that causes silent perception failure when sensors degrade in adverse weather or hardware wear.

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

### C3: Uncertainty-Aware Modality Gating
- Lightweight degradation-predictor head: estimates per-modality quality score q ∈ [0,1]
- Soft modality weighting: blend LiDAR BEV features with camera BEV features by q
- Avoids hard switching (brittle boundary) and full MoE complexity
- Degrades to camera-only when LiDAR score → 0, to radar fallback in adverse weather
- Gating scores are informed by C5 (language-grounded quality estimation) rather than hand-engineered heuristics

### C4: Velocity → Trajectory Extension
- Extend output head from (vx, vy) to future waypoints [(x, y) × T_future]
- Benchmark directly on nuScenes motion forecasting task
- Comparable to MTR, Wayformer, HiVT — but with noisy/degraded sensor robustness they lack

### C5: Language-Grounded Degradation Estimation (VLM)
- A CLIP-based vision-language model (ViT-L/14, LoRA fine-tuned) takes front-camera frames as input
- Fine-tuned on Isaac Sim rendered scenes paired with natural language degradation labels: e.g., "32-beam LiDAR in clear weather", "4-beam LiDAR in heavy fog", "LiDAR failed — camera only"
- Produces a semantic degradation embedding that feeds directly into the C3 gating network as a quality prior
- Key insight: cameras can *see* weather conditions (fog, rain, haze) even when LiDAR cannot measure its own degradation internally — the VLM bridges this sensing gap
- Novelty vs. prior gating work (Cocoon, MoME): they use geometric/feature-level disagreement signals; we use language as a semantic bottleneck for quality estimation, enabling interpretable gating ("why downweight LiDAR?" → text answer) and zero-shot generalization to unseen degradation descriptions
- Compute: CLIP ViT-L/14 inference is lightweight (~1GB VRAM). LoRA fine-tuning on Isaac Sim data runs comfortably on the RTX Pro 5000 Ada
- Training data: Isaac Sim generates unlimited paired (camera render, degradation label) examples at zero annotation cost

---

## Short-Term Goals (Month 1 — Foundation)

- [ ] **Week 1–2:** Extend existing model from velocity prediction to multi-step trajectory (waypoint head). Run on nuScenes motion forecasting split. Establish baseline numbers (ADE/FDE at 1s/2s/3s with clean 32-beam LiDAR).
- [ ] **Week 3:** Implement beam-reduction pipeline. Plot ADE/FDE vs. beam count (32→16→8→4→2→0). This degradation curve motivates everything — it is Figure 1 of the paper.
- [ ] **Week 4:** Isaac Sim setup — configure a nuScenes-like urban scene, calibrate 32-beam and degraded LiDAR sensor models (fog, rain, dust, 4-beam hardware), generate 500–1000 labeled scenarios with ground-truth degradation conditions and paired camera renders.

---

## Medium-Term Goals (Month 2 — Core Contributions)

- [ ] **Week 5–6:** Stochastic beam dropout training (C2). Randomly sample LiDAR beam count per batch {32, 16, 8, 4, 0}. Retrain and re-plot degradation curve vs. Month 1 baseline. Show the gap closes.
- [ ] **Week 7:** CLIP-based degradation estimator (C5). Fine-tune CLIP ViT-L/14 with LoRA on Isaac Sim camera renders paired with natural language degradation labels. Verify it correctly classifies degradation conditions from camera frames alone.
- [ ] **Week 8:** Language-grounded modality gating (C3). Wire C5 embeddings into the soft gating network; replace or augment the handcrafted quality score. Ablate: geometric-only gating vs. language-gating vs. both.
- [ ] **Week 9:** Sim-to-real transfer experiment. Take the model trained exclusively on real nuScenes data (with stochastic dropout). Run inference on Isaac Sim physics-accurate degraded point clouds — no fine-tuning. Plot ADE/FDE on Isaac Sim across degradation levels. This is the zero-shot transfer figure: it shows the model generalizes to physics-accurate simulation without ever training on it.

---

## Long-Term Goals (Month 3 — Polish + Paper)

- [ ] **Week 10–11:** Full ablation study — C2 only, C3 only, C5 only, C2+C3, C2+C3+C5. Degradation curves for each. Compare to: (1) no-dropout baseline, (2) hard-switch baseline, (3) MoME-style discrete routing.
- [ ] **Week 11:** Edge case experiments on Isaac Sim — complete LiDAR failure, night (camera only), heavy fog (radar only), dust storm. Show graceful degradation across all.
- [ ] **Week 12:** Paper writing. Reframe introduction around reliability gap (not cost). Include: degradation motivation curve, method figure, sim-to-real transfer figure, ablation table.

---

## Future Extensions (Beyond 3 Months)

- **4D Radar integration:** nuScenes has classical 3D radar (5 units). Extend to simulated 4D imaging radar in Isaac Sim as a weather-robust fallback. 4D radar provides native Doppler velocity and survives fog/rain where LiDAR fails.
- **Full world model:** Replace the BEV encoder with a generative world model that hallucinates missing modalities (e.g., predict what LiDAR would see given camera + motion history). BEVWorld is the closest reference.
- **Occupancy flow prediction:** Extend from per-agent trajectory to dense BEV occupancy forecasting. Comparable to Cam4DOcc benchmark.
- **Real adverse weather validation:** Collect or license data with annotated adverse weather (K-Radar, View-of-Delft datasets).

---

## Target Venues

> **Status as of June 2026:** CVPR 2026, IROS 2026, and ECCV 2026 deadlines have all passed. Realistic upcoming targets:

| Venue | Deadline (approx.) | Fit | Notes |
|---|---|---|---|
| **RA-L** (IEEE Robotics & Automation Letters) | Rolling — submit Aug–Oct 2026 | **Primary** | Rolling deadline, fast review (~3 months). Sensor robustness + trajectory prediction is core RA-L territory. Accepted RA-L papers are presented at ICRA/IROS. |
| **CoRL 2026** | ~June–July 2026 | Strong | Check current deadline. Real deployment + sensor robustness + VLM angle fits the CoRL audience well. |
| **CVPR 2027** | ~Nov 2026 | Strong | Highest visibility. VLM + BEV robustness story fits the 2027 trend toward foundation models in AV. Requires paper-ready by Oct 2026. |
| **ICRA 2027** | ~Sep 2026 | Strong | Fallback to RA-L; or submit simultaneously (RA-L → ICRA presentation). |
| **ICCV 2027** | ~March 2027 | Good | More time to mature the VLM component. Strong venue for perception + vision. |
| **IROS 2027** | ~March 2027 | Strong | Robotics + sensor reliability + real deployment focus. |

**Recommended path:** Submit to **RA-L** when results are solid (target: September 2026). This gets you reviewed fast and into ICRA 2027 as a presentation vehicle. Use the CVPR 2027 deadline (Nov 2026) as a parallel stretch target if the VLM component is strong enough by then.

---

## Novelty vs. Prior Work

| Claim | Prior art | Our delta |
|---|---|---|
| Train rich LiDAR, infer weak sensor | LEROjD (radar) | Continuous quality spectrum, not binary switch |
| Soft modality gating | Cocoon, FDSNet, MoME (CVPR 2025) | Language-grounded quality signal (C5), not geometric heuristics or discrete MoE routing |
| VLM for sensor quality estimation | None in AV perception | CLIP fine-tuned on degradation semantics; camera sees weather → infers LiDAR quality |
| Trajectory prediction under degradation | MTR, Wayformer, EgoTraj-Bench | They assume clean detection or ego-view noise; we handle structured hardware degradation |
| Sim-to-real transfer validation | RESBev (arXiv 2026) uses nuScenes only | Zero-shot: train on real nuScenes, test on Isaac Sim physics-accurate degradation — no fine-tuning |
| BEV robustness as plug-in | RESBev (latent world model) | We operate at training policy level (stochastic dropout) + gating, not post-hoc feature recovery |

---

## Evaluation Plan

**Primary metric:** ADE (Average Displacement Error) and FDE (Final Displacement Error) at 1s, 2s, 3s horizons — standard nuScenes motion forecasting metrics.

**Key experiment — Degradation Curves (nuScenes real data):**
Plot ADE/FDE vs. LiDAR beam count (32→16→8→4→2→0 beams) for:
1. Baseline (no dropout training, clean model)
2. + Stochastic dropout training (C2)
3. + Modality gating (C2 + C3)
4. + Language-grounded gating (C2 + C3 + C5)

The gap between curve 1 and curve 4 is the paper's main result. Each intermediate curve shows additive gain from each contribution.

**Sim-to-real transfer experiment (Isaac Sim):**
- Train exclusively on real nuScenes (with stochastic dropout — no Isaac Sim data at training time)
- Generate physics-accurate degraded point clouds in Isaac Sim (fog, rain, 4-beam hardware, LiDAR failure)
- Run inference on Isaac Sim outputs — zero-shot, no fine-tuning
- Plot ADE/FDE on Isaac Sim across degradation conditions
- Compare: baseline (collapses), ours (degrades gracefully)
- This answers: does stochastic dropout training generalize to *physically simulated* degradation, not just artificially subsampled beams?

**VLM ablation:**
- Frozen CLIP (no fine-tune) as gating prior → LoRA fine-tuned CLIP → handcrafted geometric quality score
- Show that language-grounded estimation outperforms geometric heuristics, especially on unseen degradation conditions

**Secondary experiments:**
- Adverse weather edge cases (Isaac Sim): fog only, rain only, dust storm, complete LiDAR blackout
- Hard-switch baseline (MoME-style discrete routing) vs. our soft gating
- Full ablation table: C2 only, C3 only, C5 only, C2+C3, C2+C3+C5

---

## Related Work (Key Papers)

**Closest analogues — must cite and differentiate:**
- [LEROjD (ECCV 2024)](https://arxiv.org/abs/2409.05564) — LiDAR-train / radar-infer transfer; binary switch, not continuous spectrum
- [MoME (CVPR 2025)](https://arxiv.org/abs/2503.19776) — multi-modal expert fusion for sensor failure; discrete per-query routing vs. our soft language-grounded gating
- [RESBev (arXiv Mar 2026)](https://arxiv.org/abs/2603.09529) — plug-in BEV robustness via latent world model; post-hoc feature recovery vs. our train-time dropout + gating
- [BEVWorld (2024)](https://arxiv.org/html/2407.05679v3) — multimodal world model in BEV latent space
- [Cocoon (2024)](https://arxiv.org/html/2410.12592v1) — uncertainty-aware sensor fusion; geometric disagreement signal vs. our language-grounded signal

**VLMs in autonomous driving (C5 context):**
- [AUTOPILOT Workshop @ CVPR 2026](https://www.autopilot-cvpr.net/) — VLMs for safe AV perception; our C5 fits this trend
- [Foundation Models for AV Perception Survey (2025)](https://arxiv.org/html/2509.08302v1) — CLIP and VLMs increasingly used for scene understanding; not yet for sensor quality estimation
- [DriveX @ CVPR 2026](https://drivex-workshop.github.io/cvpr2026/) — 4D Radar + LiDAR + Camera cooperative perception under adverse weather; complementary to C5

**Trajectory prediction:**
- [MTR++ (2024)](https://arxiv.org/html/2306.17770v2) — motion transformer with intention queries
- [Wayformer](https://waymo.com/research/wayformer) — attention-based motion forecasting
- [EgoTraj-Bench (arXiv Oct 2025)](https://arxiv.org/abs/2510.00405) — robust trajectory prediction under ego-view noise; our C4 analogue but for hardware degradation

**Sensor degradation benchmarks:**
- [MSC-Bench (2025)](https://arxiv.org/html/2501.1037) — multi-sensor corruption benchmark
- [4D Radar Meets LiDAR and Camera (CVPR 2026 Workshop)](https://arxiv.org/abs/2606.00416) — physics-based LiDAR degradation benchmarks; compare our Isaac Sim pipeline to their OPV2V-R/Adver-City-R

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
