# src/

Python package containing the dataset class and model used by `scripts/train.py` and `scripts/infer.py`.

```
src/
├── __init__.py
├── dataset.py   # BEVVelocityDataset — temporal sliding-window PyTorch dataset
├── model.py     # TemporalVelocityPredictor — BEV encoder + temporal model velocity predictor
├── kalman.py    # Kalman filter + RTS smoother for velocity estimation from noisy positions
├── box_noise.py # Realistic camera-only 3D detector noise injection
└── utils.py     # Shared helper utilities
```

Run all scripts from the **project root** so that `src` resolves correctly:

```bash
python scripts/train.py   # not: cd scripts && python train.py
```

---

## `dataset.py` — `BEVVelocityDataset`

PyTorch `Dataset` that reads `metadata.json` and builds **sliding windows** of T consecutive frames for each `(scene, instance)` pair. Only `is_valid=True` frames are used. Windows with non-consecutive frame indices are discarded.

### Constructor

```python
BEVVelocityDataset(meta_path, data_dir, T=3, split='train', val_last_n=10,
                   split_scenes=None, use_subframes=False, delta_bev=False,
                   add_kinematics=False, box_noise=False, box_noise_params=None,
                   use_kalman=True, rng_seed=None)
```

| Argument | Description |
|---|---|
| `meta_path` | Path to `metadata.json` |
| `data_dir` | Root of `bev_data/` (parent of `full_bevs/`, `crops/`, `labels/`) |
| `T` | Temporal window length — consecutive keyframes per sample |
| `split` | `'train'` or `'val'` |
| `val_last_n` | Frames at the end of each scene reserved for validation (temporal split mode only) |
| `split_scenes` | Optional set of scene names — when provided, uses official nuScenes scene splits instead of temporal split |
| `use_subframes` | Interleave far+near BEVs per keyframe; doubles effective BEV timesteps fed to the model |
| `delta_bev` | Append frame-to-frame BEV differences as extra channels; doubles `bev_channels` |
| `add_kinematics` | Append `[dt, vx̂, vŷ, fd_vx, fd_vy]` to box features; `box_params` shape becomes `(T, 12)` |
| `box_noise` | Inject per-frame camera-only detector noise into box params during training (see `box_noise.py`) |
| `box_noise_params` | Dict passed to `add_camera_detector_noise` — keys: `noise_scale`, `depth_outlier_prob`, `yaw_pi_flip_prob` |
| `use_kalman` | Use Kalman-filtered velocity for `vx̂/vŷ` (True) or naive finite-diff (False); only matters when `add_kinematics=True` |
| `rng_seed` | Seed for box noise RNG — ensures reproducible noise across epochs per sample index |

**Split logic:** when `split_scenes` is provided, scenes are assigned to train/val by the official nuScenes split. Otherwise, the last `val_last_n` frames of every scene form the val set; a T−1 frame gap at the boundary prevents leakage.

### `__getitem__` return values

| Tensor | Shape | Description |
|---|---|---|
| `bev_stack` | `(T_eff × 22, 500, 500)` | Full-scene BEV frames; `T_eff = T×2` when `use_subframes=True`, else `T` |
| `crop_stack` | `(T × 22, 64, 64)` | T rotation-aligned vehicle crops, extracted dynamically from each frame's own position |
| `box_params` | `(T, 7)` or `(T, 12)` | `[x,y,z,l,w,h,yaw]` per frame; when `add_kinematics=True` appends `[dt, kf_vx, kf_vy, fd_vx, fd_vy]` |
| `label` | `(2,)` | `[vx, vy]` ground-truth velocity at the **last** frame |

When `add_kinematics=True`, columns 8–9 are Kalman (or FD) velocity estimate and columns 10–11 are naive finite-diff stored for display in `infer.py`. Only columns 0–9 are passed to the model.

---

## `model.py` — `TemporalVelocityPredictor`

Predicts vehicle velocity `[vx, vy]` from a temporal window of T BEV frames, crops, and 3D box parameters.

### Architecture

```
Full BEV   (B, T×22, 500, 500) ─→ reshape (B×T, 22, 500, 500) ─→ BEVEncoder  ─→ (B, T, 256) ─┐
Crop       (B, T×22,  64,  64) ─→ reshape (B×T, 22,  64,  64) ─→ CropEncoder ─→ (B, T, 128) ──┼─→ cat (B, T, 448) ─→ [GRU | Transformer] ─→ (B, hidden) ─→ head ─→ (B, 2)
Box params (B, T, box_dim)     ─→ Linear(box_dim→64)           ─→             ─→ (B, T,  64) ──┘
```

Both encoders process the full `(B×T)` batch in a single forward pass, then reshape back to `(B, T, features)` before the temporal model. When `use_subframes=True`, `T_bev = T_kf × 2`; crop and box features are repeat-interleaved to match.

GRU uses its final hidden state; Transformer uses mean pooling of all token outputs, followed by a linear projection to `hidden_size`.

### Sub-modules

| Class | Input | Output | Description |
|---|---|---|---|
| `BEVEncoder` | `(B, 22, 500, 500)` | `(B, 256)` | 4-stage depthwise-separable CNN with residual skips + adaptive avg pool |
| `ResNet18BEVEncoder` | `(B, 22, 500, 500)` | `(B, 256)` | torchvision ResNet18 (no pretrained weights) |
| `CropEncoder` | `(B, 22, 64, 64)` | `(B, 128)` | 3-layer CNN with BatchNorm |
| `EfficientNetCropEncoder` | `(B, 22, 64, 64)` | `(B, 128)` | EfficientNet-B0 (ImageNet pretrained, first conv replaced for 22 channels) |
| `_LearnedPositionalEncoding` | `(B, T, 448)` | `(B, T, 448)` | Learnable position embeddings (nn.Embedding) added before Transformer |
| `TemporalVelocityPredictor` | bev, crop, box | `(B, 2)` | Combines encoders → temporal model → head |

### Constructor

```python
TemporalVelocityPredictor(
    T=3,
    box_dim=7,           # 10 when add_kinematics=True
    hidden_size=256,
    dropout=0.1,
    bev_encoder='lightweight',    # 'lightweight' | 'resnet18'
    crop_encoder='lightweight',   # 'lightweight' | 'efficientnet'
    bev_channels=22,
    crop_channels=22,
    temporal_model='transformer', # 'gru' | 'transformer'
    nhead=4,                      # attention heads (must divide 448 evenly)
    num_layers=2,                 # TransformerEncoderLayer stacks
    dim_feedforward=512,          # FFN inner width per encoder layer
)
```

### Forward signature

```python
model(bev_stack, crop_stack, box_params) → (B, 2)
```

Outputs are in **normalized label space**. When `residual_velocity=True` the model predicts `GT_vel − noisy_vhat`; inference adds back `box[:, -1, 8:10]`. Otherwise denormalize with:

```python
ckpt = torch.load('outputs/checkpoints/best_model.pt')
pred_ms = pred * ckpt['label_std'] + ckpt['label_mean']
```

### Checkpoint keys

| Key | Description |
|---|---|
| `model_state` | Model weights |
| `optim_state` | Adam optimizer state |
| `sched_state` | LR scheduler state |
| `epoch` | Epoch at which checkpoint was saved |
| `val_loss` | Validation Huber loss (normalized space) |
| `best_val_loss` | Best val loss seen so far (for resuming) |
| `train_losses` / `val_losses` | Per-epoch loss history lists |
| `T` | Effective BEV timesteps fed to model (`T_kf × 2` with subframes) |
| `T_kf` | Keyframe count used by dataset |
| `hidden_size` / `box_dim` | Architecture hyperparameters |
| `label_mean` / `label_std` | `(2,)` tensors for denormalizing full-velocity predictions |
| `residual_mean` / `residual_std` | `(2,)` tensors for denormalizing residual predictions |
| `residual_velocity` | Whether model predicts residual `(GT − vhat)` instead of full velocity |
| `box_mean` / `box_std` | `(box_dim,)` tensors for normalizing box params at inference |
| `bev_encoder` / `crop_encoder` | Encoder variant strings |
| `bev_channels` / `crop_channels` | Channel counts |
| `dropout` / `scheduler` / `grad_clip` | Training settings |
| `temporal_model` | `'gru'` or `'transformer'` |
| `transformer_nhead` / `transformer_num_layers` / `transformer_dim_feedforward` | Transformer hyperparameters |
| `add_kinematics` / `use_kalman` / `use_subframes` / `delta_bev` | Data flags |
| `split_mode` / `val_scenes` | Split configuration for reproducible inference |
| `seed` | Training seed |

---

## `kalman.py` — `kalman_velocity`

Estimates velocity at every keyframe from a sequence of noisy 2D positions using a **constant-velocity Kalman forward filter + Rauch-Tung-Striebel (RTS) backward smoother**.

The measurement noise covariance `R` is built per-step from the object's current range and matches the anisotropic depth/lateral noise profile used in `box_noise.py` — radial (log-normal depth) and lateral (Laplace) variances are rotated from polar to Cartesian coordinates.

The RTS backward pass ensures every frame receives an estimate informed by all T observations, not just past ones. This avoids frames 0 and 1 collapsing to the same velocity estimate when initialized from forward finite-diff.

```python
from src.kalman import kalman_velocity

velocities = kalman_velocity(positions, timestamps, noise_scale=1.0)
# positions:   (T, 2) float — noisy [px, py] in reference LiDAR frame
# timestamps:  (T,)  int   — UNIX microsecond timestamps
# returns:     (T, 2) float — RTS-smoothed [vx, vy] at each step
```

---

## `box_noise.py` — `add_camera_detector_noise`

Injects realistic camera-only 3D bounding-box detector noise into GT box arrays. Calibrated to monocular/multi-camera detectors (FCOS3D, DETR3D, BEVDet) on nuScenes. Each noise source uses a distribution matched to how that error actually arises:

| Feature | Distribution | Rationale |
|---|---|---|
| Depth (radial) | Log-normal multiplicative + Laplace outlier | Depth error is roughly constant in *relative* (%) terms; gross 2D→3D failures add heavy-tailed jumps |
| Lateral | Laplace | Bursty image-space angular errors with heavy tails |
| Height z | Laplace + positive bias | Ground-plane imperfection causes slight height overestimation |
| Yaw | 4-component mixture: Laplace(0), Laplace(π), Laplace(±π/2), Uniform(−π,π) | Confident flip errors are the dominant failure mode |
| Dimensions | Shared log-normal scale + per-axis Laplace + regression-to-mean | Networks learn strong size priors and partially shrink toward dataset average |

All noise is i.i.d. per frame (no shared bias across frames). Controlled by `noise_scale` (0=off, 1=calibrated, >1=worse), `depth_outlier_prob`, and `yaw_pi_flip_prob`.

```python
from src.box_noise import add_camera_detector_noise

noisy = add_camera_detector_noise(box_params, rng=rng, noise_scale=0.5)
# box_params: (T, 7) float — [x,y,z,l,w,h,yaw] per frame
# returns:    (T, 7) float — same dtype, noisy copy
```
