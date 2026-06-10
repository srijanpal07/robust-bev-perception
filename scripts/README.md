# Scripts

All scripts are run from the **project root**, not from inside `scripts/`.

## Canonical entrypoints

| Script | Purpose | Notes |
|---|---|---|
| `scripts/train/train.py` | **Training** — YAML-driven, velocity + trajectory tasks | Use this going forward |
| `scripts/eval/eval_degradation_curve.py` | Degradation sweep (ADE/FDE/ECE vs beam count) | Stub — needs `beam_degradation.py` |
| `scripts/eval/eval_calibration.py` | Calibration sweep + reliability diagrams | Stub — needs `TrajectoryHead` |
| `scripts/eval/eval_isaac_sim.py` | Cross-domain robustness evaluation | Stub — needs Isaac Sim scenes |

## Legacy scripts (original baseline — still functional)

`scripts/train.py` and `scripts/infer.py` are the original training and inference scripts from the baseline codebase. They are kept intact and still work — use them if you need the original Huber-loss velocity training loop or the full inference report with BEVFormer comparison. The new `scripts/train/train.py` replaces them for the research contributions.

```
scripts/
├── train/train.py             # ← NEW canonical training entrypoint
├── eval/                      # ← NEW eval scripts (stubs)
├── baselines/                 # ← NEW (stubs wrapping old baseline scripts)
│
│  Legacy — original baseline:
├── train.py                   # original velocity training (Huber loss, full checkpoint)
├── infer.py                   # original inference + BEVFormer comparison
├── visualize.py               # Rerun 3D visualizer
├── save_bev.py                # generate BEV training data
└── …                          # dataset prep, BEVFormer pipeline (see below)
```

```
scripts/
├── prepare_dataset.py             # Merge extracted blob directories into devkit-compatible layout
├── check_dataset.py               # Verify raw nuScenes dataset
├── filter_dataset.py              # Find qualifying vehicle instances; write filtered CSV
├── visualize.py                   # Rerun 3D visualizer
├── save_bev.py                    # Generate BEV training data (single instance or batch)
├── check_bev.py                   # Verify save_bev.py output
├── convert_bevs_float16.py        # Convert full_bevs/ .npy files float32 → float16 to halve disk I/O
├── train.py                       # Train the velocity prediction model
├── infer.py                       # Run inference and save predictions
│
│   BEVFormer evaluation pipeline:
├── create_bevformer_val_infos.py  # Build nuscenes_infos_temporal_val.pkl (no CAN bus required)
├── match_bevformer_instances.py   # Match BEVFormer boxes to filtered instances by center-distance
├── infer_with_bevformer_boxes.py  # Run model on BEVFormer boxes; compare vs BEVFormer velocity head
├── compare_bevformer_thresholds.py# Aggregate 2m/5m/10m comparison results into one report
└── finetune_bevformer.py          # Fine-tune pretrained model on matched BEVFormer windows
```

---

## Recommended workflow

```
1. prepare_dataset.py  → merge raw_full blobs (full dataset only, run once per new blob)
2. check_dataset.py    → confirm dataset loads correctly
3. filter_dataset.py   → find qualifying instances → datasets/filtered_{version}.csv
4. save_bev.py --batch → generate all BEVs from the CSV automatically
5. check_bev.py        → verify saved data is correct
6. train.py            → train the model
7. infer.py            → run inference on val set
```

### BEVFormer evaluation + fine-tuning workflow

See `setup_bevformer.sh` for one-shot environment setup (creates the `bevformer` conda env, installs all packages, downloads checkpoints).

**Step 1 — build val infos**
```bash
conda run -n bevformer python3 scripts/create_bevformer_val_infos.py
```

**Step 2 — BEVFormer-Small inference (~18 min, 1 GPU)**
```bash
cd bevformer
conda run -n bevformer bash tools/dist_test.sh \
    projects/configs/bevformer/bevformer_small.py \
    ckpts/bevformer_small_epoch_24.pth \
    1 \
    --out ../detr3d/work_dirs/bevformer_val_results.pkl \
    --eval bbox
cd ..
```

**Steps 3–7 — matching, evaluation, fine-tuning**
```
3. match_bevformer_instances.py    → match BEVFormer boxes to filtered instances
4. infer_with_bevformer_boxes.py   → evaluate pretrained model on BEVFormer boxes
5. compare_bevformer_thresholds.py → aggregate 2m/5m/10m results
6. finetune_bevformer.py           → fine-tune on matched windows; save best_model_finetuned.pt
7. infer_with_bevformer_boxes.py   → re-evaluate with fine-tuned model (use --checkpoint flag)
```

---

## `prepare_dataset.py` — Merge blob directories

After extracting the nuScenes trainval archives into `datasets/raw_full/`, each blob lands in its own subdirectory (`v1.0-trainval01_blobs/`, `v1.0-trainval02_blobs/`, …). This script flattens them into the single root the devkit expects.

Safe to re-run — already-merged files are skipped, empty subdirectories are cleaned up.

```bash
python scripts/prepare_dataset.py                      # uses datasets/raw_full
python scripts/prepare_dataset.py /path/to/raw_full    # explicit path
```

**Expected output:**
```
Processing metadata...
  Moved: v1.0-trainval/
  Moved: maps/
Processing v1.0-trainval01_blobs/
  Moved   samples/LIDAR_TOP/
  ...
Final layout:
  maps/
  samples/
  sweeps/
  v1.0-trainval/

Ready — run: python scripts/check_dataset.py --version full
```

---

## `check_dataset.py` — Raw dataset sanity check

Loads the nuScenes devkit and prints the scene count plus sample annotation details.

```bash
python scripts/check_dataset.py              # full dataset (default)
python scripts/check_dataset.py --version mini
```

| Argument | Default | Description |
|---|---|---|
| `--version` | `full` | `mini` (10 scenes) or `full` (850 scenes) |

---

## `filter_dataset.py` — Find qualifying vehicle instances

Scans all scenes for `vehicle.car` instances satisfying **all** of the following:

1. Visible in at least one front camera (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`) in **every** frame
2. Present in **at least 40 frames**
3. Within **50 m** of ego in every frame
4. Average speed > **0.5 m/s** (excludes parked vehicles)

Prints a table to stdout and writes results to `datasets/filtered_{version}.csv` — used by `save_bev.py --batch`.

```bash
python scripts/filter_dataset.py              # full dataset (default)
python scripts/filter_dataset.py --version mini
```

| Argument | Default | Description |
|---|---|---|
| `--version` | `full` | Dataset version to scan |

**Output CSV** (`datasets/filtered_full.csv`):
```
scene,instance_token,avg_speed_mps,num_frames
scene-0239,034d413a26b14ca9aa29b179cebca8a9,0.622,40
...
```

---

## `save_bev.py` — Generate BEV training data

For each qualifying vehicle instance, iterates over all frames in its scene and saves 22-channel BEV arrays, crops, velocity labels, and metadata.

```bash
# Batch mode — process all instances from the filtered CSV (recommended)
python scripts/save_bev.py --batch

# Single instance
python scripts/save_bev.py --instance a3f2c1

# Mini dataset
python scripts/save_bev.py --batch --version mini
```

| Argument | Default | Description |
|---|---|---|
| `--batch` | off | Read `datasets/filtered_{version}.csv` and process all instances |
| `--instance` | — | Instance token or suffix (mutually exclusive with `--batch`) |
| `--scene` | auto | Scene name — auto-detected from instance if omitted |
| `--output` | `datasets/bev_data` | Root output directory |
| `--version` | `full` | Dataset version |

**Output structure:**
```
datasets/bev_data/
├── full_bevs/        scene-XXXX_frameNNN.npy  → (22, 500, 500) float32
├── crops/            scene-XXXX_frameNNN.npy  → (22,  64,  64) float32  [10 m window]
├── crops_context/    scene-XXXX_frameNNN.npy  → (22,  64,  64) float32  [20 m window]
├── labels/           scene-XXXX_frameNNN.npy  → (2,)           float32  [vx, vy m/s]
├── bev_channel_stats.npz                      → per-channel mean/std for normalization
└── metadata.json
```

**BEV channels (22 total):**

| Channel | Content |
|---|---|
| 0 | Log point density |
| 1 | Mean height (z) |
| 2 | Max height (z) |
| 3 | Mean LiDAR intensity |
| 4 | Min height (z) |
| 5 | Height spread (max − min z) |
| 6–21 | Log occupancy per height bin (−1 m to +3 m, 16 equal bins) |

Instances from blobs not yet downloaded are detected via a LiDAR file check and skipped with a `SKIP` message — the batch does not crash.

---

## `check_bev.py` — Verify BEV output

Sanity-checks every entry in `metadata.json` against the saved `.npy` files.

```bash
python scripts/check_bev.py                       # default: datasets/bev_data
python scripts/check_bev.py --data datasets/bev_data
```

| Argument | Default | Description |
|---|---|---|
| `--data` | `datasets/bev_data` | Root bev_data directory to verify |

**Checks performed:**
- All `.npy` files exist for valid frames (and absent for invalid frames)
- Array shapes: `(22, 500, 500)`, `(22, 64, 64)`, `(2,)`
- All arrays are `float32` with no non-finite values
- Density channel has no negative values
- Labels match `velocity_gt` in metadata; no NaN on valid frames
- No duplicate `fname` entries

---

## `convert_bevs_float16.py` — Float32 → Float16 BEV conversion

Converts all `full_bevs/` (and optionally `full_bevs_far/`) `.npy` files from float32 to float16
**in-place**, halving disk space and reducing I/O time during training. Uses a parallel worker
pool for throughput. Safe to re-run — files already in float16 are skipped.

```bash
python scripts/convert_bevs_float16.py --data-dir /mnt/datasets/bev_data
python scripts/convert_bevs_float16.py --data-dir /mnt/datasets/bev_data --also-far   # also convert full_bevs_far/
python scripts/convert_bevs_float16.py --data-dir /mnt/datasets/bev_data --dry-run    # preview without writing
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `/mnt/datasets/bev_data` | Root `bev_data/` directory |
| `--also-far` | off | Also convert `full_bevs_far/` (only populated when `use_subframes=true`) |
| `--workers` | `8` | Parallel worker processes |
| `--dry-run` | off | Print projected savings without writing any files |

**Expected output:**
```
Converting /mnt/datasets/bev_data/full_bevs/ ...
  42830 files  (37.8 GB)  → projected 18.9 GB after
  [  500/42830]  converted=500  already=0  skipped=0
  ...
  Done. 18.9 GB on disk (was 37.8 GB, saved 18.9 GB)
```

> **Note:** `crops/` and `labels/` are small and not converted — only the large full-scene BEV arrays
> benefit meaningfully from float16 compression. float16 precision (≈3 decimal digits) is sufficient
> for the BEV features, which are already in a compressed log-density space.

---

## `train.py` — Train the velocity prediction model

Trains `TemporalVelocityPredictor` end-to-end with **Huber loss** on normalized `[vx, vy]` labels. Saves `best_model.pt`, a `latest_ckpt.pt` after every epoch (for resuming), and `last_model.pt` at the end. Writes a per-run CSV log to `outputs/results/train_log_<timestamp>.csv`.

```bash
python scripts/train.py                                          # default config
python scripts/train.py --config configs/train.yaml
python scripts/train.py --resume                                 # resume from latest_ckpt.pt
python scripts/train.py --resume-from outputs/checkpoints/best_model.pt
python scripts/train.py --amp                                    # bfloat16 mixed precision
python scripts/train.py --compile                                # torch.compile speedup
```

**CLI flags:**

| Flag | Description |
|---|---|
| `--config` | Path to YAML config (default: `configs/train.yaml`) |
| `--resume` | Resume training from `outputs/checkpoints/latest_ckpt.pt` |
| `--resume-from PATH` | Resume from a specific checkpoint path |
| `--amp` | Enable bfloat16 automatic mixed precision (~40% faster on Ada/Ampere GPUs) |
| `--compile` | Apply `torch.compile` to the model (~20% additional speedup after warmup) |

**Config keys (`configs/train.yaml`):**

| Key | Default | Description |
|---|---|---|
| `data.data_dir` | — | Root of `bev_data/` |
| `data.meta_path` | — | Path to `metadata.json` |
| `data.nuscenes_version` | `full` | `mini` or `full` — selects official nuScenes scene split |
| `data.use_subframes` | `false` | Interleave far+near BEVs per keyframe (doubles T for model) |
| `data.delta_bev` | `false` | Append frame-to-frame BEV diffs as extra channels |
| `data.add_kinematics` | `true` | Append `[dt, vx̂, vŷ]` to box features |
| `data.box_noise` | `true` | Inject camera-only detector noise into box params during training |
| `data.use_kalman_velocity` | `true` | Use Kalman-filtered velocity for `vx̂/vŷ` (vs naive finite-diff) |
| `data.residual_velocity` | `true` | Predict `GT_vel − noisy_vhat`; requires `add_kinematics: true` |
| `data.box_noise_params.noise_scale` | `0.5` | σ multiplier: 0=off, 1=calibrated camera noise |
| `data.box_noise_params.depth_outlier_prob` | `0.015` | Per-frame probability of a gross depth error |
| `data.box_noise_params.yaw_pi_flip_prob` | `0.001` | Per-frame probability of a 180° heading flip |
| `training.seed` | — | Random seed for reproducibility |
| `training.T` | `4` | Temporal window — consecutive keyframes per sample |
| `training.batch_size` | `10` | Training batch size |
| `training.accum_steps` | `4` | Gradient accumulation (effective batch = `batch_size × accum_steps`) |
| `training.epochs` | `30` | Total epochs |
| `training.lr` | `2e-4` | Adam learning rate |
| `training.weight_decay_encoder` | `1e-3` | Weight decay for BEV/crop encoder weight matrices |
| `training.weight_decay_temporal` | `1e-2` | Weight decay for Transformer/GRU weight matrices |
| `training.weight_decay_other` | `0.0` | Weight decay for box projection and head |
| `training.split_mode` | `scene` | `scene` (official nuScenes splits) or `temporal` (last N frames per scene) |
| `training.val_last_n` | `5` | Last N frames held out for validation (temporal mode only) |
| `training.scheduler` | `cosine` | LR scheduler: `cosine` or `plateau` |
| `training.grad_clip` | `1.0` | Max gradient norm (0 to disable) |
| `training.huber_delta` | `0.5` | Huber loss transition point |
| `model.bev_encoder` | `lightweight` | `lightweight` (DS-block CNN) or `resnet18` |
| `model.crop_encoder` | `lightweight` | `lightweight` (3-layer CNN) or `efficientnet` |
| `model.temporal_model` | `transformer` | `gru` or `transformer` |
| `model.transformer_nhead` | `4` | Attention heads (must divide 448 evenly) |
| `model.transformer_num_layers` | `2` | Transformer encoder layer stacks |
| `model.transformer_dim_feedforward` | `512` | FFN inner width per encoder layer |
| `model.hidden_size` | `318` | Hidden size for GRU state or Transformer projection |
| `model.dropout` | `0.35` | Dropout rate in encoder and transformer stages |
| `model.bev_channels` | `22` | Must match `BEV_N_CHANNELS` in `save_bev.py` |
| `model.crop_channels` | `22` | Same as `bev_channels` (crops extracted from BEV) |
| `output.ckpt_dir` | `outputs/checkpoints` | Checkpoint save directory |
| `output.plots_dir` | `outputs/plots` | Training curve save directory |
| `output.results_dir` | `outputs/results` | Per-run CSV log directory |

**Expected output:**
```
Epoch 001/030 [train]: 100%|████████| 42/42 [loss=0.6421]
Epoch 001/030 [val]:   100%|████████| 12/12
Epoch 001 | train=0.6213 | val=0.7104 | lr=2.00e-04
  → Saved best model (val_loss=0.7104)
```

---

## `infer.py` — Run inference on the validation set

Loads `outputs/checkpoints/best_model.pt`, runs it on the val set, and saves per-sample predictions to `outputs/results/predictions.json`. Prints per-sample latency, and when `add_kinematics=True` shows Kalman and finite-diff baseline velocity per timestep. Prints a summary with percentile errors, mean latency, and distance-bucketed breakdown.

```bash
python scripts/infer.py
python scripts/infer.py --config configs/train.yaml
```

All architecture and split settings are read from the checkpoint — no need to match the config manually.

**Output fields in `predictions.json`:**

| Field | Description |
|---|---|
| `sample_idx` | Sample index |
| `dist_m` | Distance of target vehicle from ego at last keyframe (m) |
| `pred_vx`, `pred_vy` | Predicted velocity (m/s) |
| `gt_vx`, `gt_vy` | Ground-truth velocity (m/s) |
| `speed_pred`, `speed_gt` | Scalar speeds (m/s) |
| `speed_error` | `\|speed_pred − speed_gt\|` |
| `vector_error` | `‖pred − gt‖` — primary metric |
| `kf_error` | `‖kf_estimate − gt‖` — Kalman/FD baseline error (null if no kinematics) |
| `fd_error` | `‖fd_estimate − gt‖` — naive finite-diff baseline error (null if no kinematics or not using Kalman) |
| `latency_ms` | Model forward-pass time in milliseconds |

**Summary printed to stdout and saved to `outputs/results/infer_summary_<timestamp>.txt`:**
- Mean / median / P90 / P95 / max vector error
- Model vs Kalman and model vs FD delta (% improvement)
- Error breakdown by distance bucket: 0–20 m, 20–40 m, 40–60 m, 60–100 m

---

## `visualize.py` — Rerun 3D visualizer

Visualizes a nuScenes scene in [Rerun](https://rerun.io) with LiDAR point clouds, 6 camera feeds, 3D bounding boxes, and velocity arrows.

```bash
python scripts/visualize.py                            # default scene, all vehicles
python scripts/visualize.py --scene scene-0061
python scripts/visualize.py --mode front --scene scene-0061
python scripts/visualize.py --instance a3f2c1
python scripts/visualize.py --instance-only a3f2c1 --scene scene-0061
```

| Argument | Default | Description |
|---|---|---|
| `--mode` | `all` | `all` / `front` / `closest_front` |
| `--scene` | scene index 7 | Scene name, e.g. `scene-0061` |
| `--instance` | — | Token or suffix — annotate one vehicle |
| `--instance-only` | — | Token or suffix — isolate one vehicle (overrides `--mode`) |

---

## `create_bevformer_val_infos.py` — Build BEVFormer val infos

Creates `nuscenes_infos_temporal_val.pkl` for running BEVFormer on the nuScenes val split without requiring CAN bus data (`can_bus=zeros(18)` fallback). Processes only val scenes (~5 min vs ~30 min for the stock `create_data.py`). Replicates BEVFormer's yaw convention, velocity frame transform, and `NameMapping` exactly.

```bash
conda run -n bevformer python3 scripts/create_bevformer_val_infos.py
```

Output: `bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl`

---

## `match_bevformer_instances.py` — Match BEVFormer boxes to instances

For each filtered val instance and each frame, finds the closest BEVFormer predicted car box (score ≥ 0.3) within a configurable center-distance threshold. Reports T=4 complete-window coverage.

```bash
conda run -n bevformer python3 scripts/match_bevformer_instances.py \
    --filtered-csv datasets/filtered_full.csv \
    --bevformer-results detr3d/work_dirs/bevformer_val_results.pkl \
    --infos bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl \
    --dataroot /mnt/datasets/nuscenes \
    --max-center-distance 2.0 \
    --output outputs/results/bevformer_instance_boxes.json
```

| Argument | Default | Description |
|---|---|---|
| `--filtered-csv` | — | CSV of qualifying instances from `filter_dataset.py` |
| `--bevformer-results` | — | `.pkl` of BEVFormer inference results |
| `--infos` | — | `nuscenes_infos_temporal_val.pkl` |
| `--dataroot` | — | nuScenes raw data root |
| `--max-center-distance` | `2.0` | Match threshold in metres |
| `--output` | `outputs/results/bevformer_instance_boxes.json` | Output JSON path |

Typical match rates: **12.2% at 2m**, 29.1% at 5m, 53.7% at 10m. The low 2m rate reflects genuine BEVFormer miss-detections on fast-moving vehicles (median closest-car distance for unmatched frames: 12.7 m).

---

## `infer_with_bevformer_boxes.py` — Evaluate model on BEVFormer boxes

Runs the velocity model using BEVFormer predicted boxes as input instead of GT+noise. BEVFormer's own velocity estimate fills the `vhat` slot for residual decoding. Compares our prediction against BEVFormer's velocity head and GT.

```bash
# Pretrained model
conda run -n bevformer python3 scripts/infer_with_bevformer_boxes.py \
    --bevformer-boxes outputs/results/bevformer_instance_boxes.json \
    --checkpoint outputs/runs/residual_model/models/best_model.pt \
    --output outputs/results/bevformer_comparison_2m.json

# Fine-tuned model
conda run -n bevformer python3 scripts/infer_with_bevformer_boxes.py \
    --bevformer-boxes outputs/results/bevformer_instance_boxes.json \
    --checkpoint outputs/checkpoints/best_model_finetuned.pt \
    --output outputs/results/bevformer_comparison_ft.json
```

| Argument | Default | Description |
|---|---|---|
| `--bevformer-boxes` | `outputs/results/bevformer_instance_boxes.json` | Match JSON from `match_bevformer_instances.py` |
| `--checkpoint` | `ckpt_dir/best_model.pt` from config | Override checkpoint path |
| `--output` | `outputs/results/bevformer_comparison.json` | Output JSON path |
| `--config` | `configs/train.yaml` | Config for data paths |
| `--no-val-filter` | off | Evaluate all scenes (not just official val) |

---

## `compare_bevformer_thresholds.py` — Aggregate threshold comparison

Reads the three per-threshold inference JSONs (2m, 5m, 10m) and prints/saves a formatted comparison table covering match rates, velocity error statistics, percentile breakdowns, and distance-bucket analysis.

```bash
python scripts/compare_bevformer_thresholds.py
```

Output: `outputs/results/bevformer_threshold_comparison.txt`

---

## `finetune_bevformer.py` — Fine-tune on BEVFormer windows

Adapts the pretrained model to BEVFormer's noise distribution. Freezes `bev_encoder` and `crop_encoder` (LiDAR — unchanged from training); fine-tunes the box encoder, temporal model, and prediction head. Recomputes box and residual normalization stats from the fine-tuning training data. Splits matched instances 80/20 at the instance level (not window level) to prevent leakage. Stops early when val loss plateaus.

```bash
conda run -n bevformer python3 scripts/finetune_bevformer.py \
    --config configs/finetune_bevformer.yaml
```

Key config options (`configs/finetune_bevformer.yaml`):

| Key | Default | Description |
|---|---|---|
| `finetune.checkpoint` | — | Pretrained checkpoint to start from |
| `finetune.bevformer_boxes` | — | Match JSON (use 5m for best coverage) |
| `finetune.output_ckpt` | — | Where to save the fine-tuned model |
| `finetune.val_fraction` | `0.2` | Fraction of instances held out for validation |
| `finetune.freeze_encoders` | `true` | Freeze `bev_encoder` and `crop_encoder` |
| `training.lr` | `5e-5` | Learning rate (lower than original to preserve pretrained weights) |
| `training.patience` | `10` | Early stopping patience (epochs without val improvement) |

**Results on held-out instances (5m match JSON, 21/5 instance train/val split):**

| Threshold | Held-out windows | Fine-tuned | BEVFormer | Improvement |
|---|---|---|---|---|
| 2m | 46 | 1.11 m/s | 1.50 m/s | **+25.5%** |
| 5m | 89 | 1.48 m/s | 3.19 m/s | +53.5% |
| 10m | 130 | 1.61 m/s | 3.80 m/s | +57.7% |

Fine-tuned checkpoint saved to `outputs/checkpoints/best_model_finetuned.pt`. Interpret as a diagnostic for LiDAR-assisted refinement — effective independent sample count is ~22 (75% window overlap) across 5 instances.
