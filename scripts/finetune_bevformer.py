"""
Fine-tune the pretrained TemporalVelocityPredictor on matched BEVFormer windows.

Adapts the model's residual corrections to BEVFormer's actual noise distribution.
BEV/crop encoders are frozen (they see LiDAR, identical to training). Only the
box encoder, temporal model, and prediction head are updated.

Normalization stats (box_mean/std, residual_mean/std) are recomputed from the
fine-tuning training windows so that the different vhat distribution (BEVFormer
velocity head vs. FD from noisy GT positions) is handled correctly.

Run from project root:
    conda run -n detr3d python3 scripts/finetune_bevformer.py \\
        --config configs/finetune_bevformer.yaml
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import _extract_crop, _rotate_crop
from src.model import TemporalVelocityPredictor


# ---------------------------------------------------------------------------
# BEV / box helpers (mirrors infer_with_bevformer_boxes.py)
# ---------------------------------------------------------------------------

def _build_bev_stack(meta_win, data_dir, use_subframes, delta_bev):
    bev_list, crops = [], []
    for meta_f in meta_win:
        bev_near = np.load(
            f"{data_dir}/full_bevs/{meta_f['fname']}.npy"
        ).astype(np.float32, copy=False)

        if use_subframes:
            bev_far = np.load(
                f"{data_dir}/full_bevs_far/{meta_f['fname']}.npy"
            ).astype(np.float32, copy=False)
            bev_list.extend([bev_far, bev_near])
        else:
            bev_list.append(bev_near)

        row_v, col_v = meta_f["box_3d"]["vehicle_bev_px"]
        crop = _extract_crop(bev_near, int(row_v), int(col_v))
        crops.append(_rotate_crop(crop, meta_f["box_3d"]["yaw_ref"]))

    bev_tensor = torch.tensor(
        np.concatenate(bev_list, axis=0), dtype=torch.float32
    )
    if delta_bev:
        C = bev_list[0].shape[0]
        T_eff = len(bev_list)
        H, W = bev_tensor.shape[1], bev_tensor.shape[2]
        frames_t = bev_tensor.reshape(T_eff, C, H, W)
        deltas = torch.zeros_like(frames_t)
        deltas[1:] = frames_t[1:] - frames_t[:-1]
        bev_tensor = torch.cat([frames_t, deltas], dim=1).reshape(
            T_eff * 2 * C, H, W
        )

    crop_tensor = torch.tensor(
        np.concatenate(crops, axis=0), dtype=torch.float32
    )
    return bev_tensor, crop_tensor


def _build_box_params(meta_win, bev_boxes, add_kinematics, T):
    timestamps = [m["timestamp"] for m in meta_win]
    centers = np.array(
        [b["center_lidar"] for b in bev_boxes], dtype=np.float64
    )
    rows = []
    for t, (meta_f, bev_box) in enumerate(zip(meta_win, bev_boxes)):
        c   = bev_box["center_lidar"]
        d   = bev_box["dims_lwh"]
        yaw = bev_box["yaw_lidar"]
        row = [c[0], c[1], c[2], d[0], d[1], d[2], yaw]

        if add_kinematics:
            if t == 0:
                dt = max((timestamps[1] - timestamps[0]) / 1e6, 0.1) if T > 1 else 0.5
            else:
                dt = max((timestamps[t] - timestamps[t - 1]) / 1e6, 0.1)

            vx_hat, vy_hat = bev_box["velocity_lidar"]

            if t == 0:
                dt_fd = max((timestamps[1] - timestamps[0]) / 1e6, 0.1) if T > 1 else 0.5
                fd_vx = (centers[1, 0] - centers[0, 0]) / dt_fd if T > 1 else 0.0
                fd_vy = (centers[1, 1] - centers[0, 1]) / dt_fd if T > 1 else 0.0
            else:
                dt_fd = max((timestamps[t] - timestamps[t - 1]) / 1e6, 0.1)
                fd_vx = (centers[t, 0] - centers[t - 1, 0]) / dt_fd
                fd_vy = (centers[t, 1] - centers[t - 1, 1]) / dt_fd

            row += [dt, vx_hat, vy_hat, fd_vx, fd_vy]

        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BEVFormerFinetuneDataset(Dataset):
    def __init__(self, samples, data_dir, T, add_kinematics,
                 use_subframes, delta_bev):
        """
        Args:
            samples: list of (meta_win, bev_boxes) tuples
                     meta_win  — T metadata dicts (fname, timestamp, velocity_gt, …)
                     bev_boxes — T BEVFormer box dicts (center_lidar, dims_lwh, …)
        """
        self.data_dir     = data_dir
        self.use_subframes = use_subframes
        self.delta_bev    = delta_bev
        self.T            = T
        self.add_kinematics = add_kinematics

        # Precompute box_params and gt_velocity (cheap); BEV loading stays lazy.
        self.meta_wins       = []
        self.box_params_list = []
        self.gt_velocities   = []

        for meta_win, bev_boxes in samples:
            bp  = _build_box_params(meta_win, bev_boxes, add_kinematics, T)
            gtv = torch.tensor(meta_win[-1]["velocity_gt"], dtype=torch.float32)
            self.meta_wins.append(meta_win)
            self.box_params_list.append(bp)
            self.gt_velocities.append(gtv)

    def __len__(self):
        return len(self.meta_wins)

    def __getitem__(self, idx):
        bev, crop = _build_bev_stack(
            self.meta_wins[idx], self.data_dir,
            self.use_subframes, self.delta_bev,
        )
        return bev, crop, self.box_params_list[idx], self.gt_velocities[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pretrained(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    T_mod = ckpt["T"]
    model = TemporalVelocityPredictor(
        T=T_mod,
        box_dim=ckpt["box_dim"],
        hidden_size=ckpt["hidden_size"],
        dropout=ckpt.get("dropout", 0.1),
        bev_encoder=ckpt.get("bev_encoder", "lightweight"),
        crop_encoder=ckpt.get("crop_encoder", "lightweight"),
        bev_channels=ckpt.get("bev_channels", 22),
        crop_channels=ckpt.get("crop_channels", 22),
        temporal_model=ckpt.get("temporal_model", "gru"),
        nhead=ckpt.get("transformer_nhead", 4),
        num_layers=ckpt.get("transformer_num_layers", 2),
        dim_feedforward=ckpt.get("transformer_dim_feedforward", 512),
    ).to(device)

    state = ckpt["model_state"]
    prefix = "_orig_mod."
    state  = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state)
    return model, ckpt


def build_windows(bev_data, token_to_meta, val_scenes, T):
    """Return list of (instance_token, meta_win, bev_boxes) from the match JSON."""
    groups = {}
    for rec in bev_data["matches"]:
        if val_scenes is not None and rec["scene"] not in val_scenes:
            continue
        key = (rec["scene"], rec["instance_token"])
        groups.setdefault(key, []).append(rec)
    for key in groups:
        groups[key].sort(key=lambda r: r["frame"])

    windows = []
    for (scene, inst_tok), frames in sorted(groups.items()):
        for i in range(len(frames) - T + 1):
            win = frames[i:i + T]
            if any(win[j + 1]["frame"] != win[j]["frame"] + 1
                   for j in range(T - 1)):
                continue
            if any(w["matched"] is None for w in win):
                continue
            if any(w["sample_token"] not in token_to_meta for w in win):
                continue
            meta_win  = [token_to_meta[w["sample_token"]] for w in win]
            bev_boxes = [w["matched"]["box"] for w in win]
            windows.append((inst_tok, meta_win, bev_boxes))
    return windows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_bevformer.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ft_cfg   = cfg["finetune"]
    tr_cfg   = cfg["training"]
    data_cfg = cfg["data"]

    CKPT_PATH    = ft_cfg["checkpoint"]
    BEV_BOXES    = ft_cfg["bevformer_boxes"]
    OUT_CKPT     = ft_cfg["output_ckpt"]
    VAL_FRAC     = float(ft_cfg.get("val_fraction", 0.2))
    FREEZE_ENC   = bool(ft_cfg.get("freeze_encoders", True))

    SEED         = int(tr_cfg.get("seed", 42))
    EPOCHS       = int(tr_cfg["epochs"])
    BATCH_SIZE   = int(tr_cfg["batch_size"])
    LR           = float(tr_cfg["lr"])
    GRAD_CLIP    = float(tr_cfg.get("grad_clip", 1.0))
    HUBER_DELTA  = float(tr_cfg.get("huber_delta", 0.5))
    PATIENCE     = int(tr_cfg.get("patience", 10))

    DATA_DIR  = data_cfg["data_dir"]
    META_PATH = data_cfg["meta_path"]

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained checkpoint
    print(f"Loading pretrained checkpoint: {CKPT_PATH}")
    model, base_ckpt = load_pretrained(CKPT_PATH, device)

    BOX_DIM        = base_ckpt["box_dim"]
    ADD_KINEMATICS = base_ckpt.get("add_kinematics", False)
    RESIDUAL_VEL   = base_ckpt.get("residual_velocity", False)
    USE_SUBFRAMES  = base_ckpt.get("use_subframes", False)
    DELTA_BEV      = base_ckpt.get("delta_bev", False)
    T              = base_ckpt["T"]

    print(f"Model: T={T}  box_dim={BOX_DIM}  residual={RESIDUAL_VEL}  "
          f"kinematics={ADD_KINEMATICS}")

    # Build metadata lookup
    with open(META_PATH) as f:
        all_meta = json.load(f)
    token_to_meta = {m["sample_token"]: m for m in all_meta if m["is_valid"]}

    # Load match JSON and restrict to official val scenes
    from nuscenes.utils.splits import create_splits_scenes
    val_scenes = set(create_splits_scenes()["val"])

    with open(BEV_BOXES) as f:
        bev_data = json.load(f)

    all_windows = build_windows(bev_data, token_to_meta, val_scenes, T)
    print(f"Total complete windows: {len(all_windows)}")

    # Instance-level train/val split (prevents leakage)
    all_instances = sorted(set(inst for inst, _, _ in all_windows))
    random.shuffle(all_instances)
    n_val    = max(1, int(len(all_instances) * VAL_FRAC))
    val_insts  = set(all_instances[:n_val])
    train_insts = set(all_instances[n_val:])

    train_samples = [(mw, bb) for inst, mw, bb in all_windows if inst in train_insts]
    val_samples   = [(mw, bb) for inst, mw, bb in all_windows if inst in val_insts]
    print(f"Instances — train: {len(train_insts)}  val: {len(val_insts)}")
    print(f"Windows   — train: {len(train_samples)}  val: {len(val_samples)}")

    train_ds = BEVFormerFinetuneDataset(
        train_samples, DATA_DIR, T, ADD_KINEMATICS, USE_SUBFRAMES, DELTA_BEV
    )
    val_ds = BEVFormerFinetuneDataset(
        val_samples, DATA_DIR, T, ADD_KINEMATICS, USE_SUBFRAMES, DELTA_BEV
    )

    n_workers = min(4, os.cpu_count() or 1)
    pin_mem   = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, pin_memory=pin_mem,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, pin_memory=pin_mem,
                              persistent_workers=True)

    # Recompute normalization stats from fine-tuning training data.
    # Critical: vx_hat/vy_hat (cols 8-9) have a different distribution from
    # training (BEVFormer velocity head vs FD from noisy GT positions), so we
    # must not reuse the pretrained checkpoint's box_mean/box_std directly.
    print("Computing normalization stats from fine-tuning training data …")
    all_box_rows = torch.cat(train_ds.box_params_list, dim=0)  # (N*T, 12+)
    box_mean = all_box_rows[:, :BOX_DIM].mean(0)
    box_std  = all_box_rows[:, :BOX_DIM].std(0).clamp(min=1e-6)
    print(f"  box_mean: {box_mean.tolist()}")
    print(f"  box_std:  {box_std.tolist()}")

    # Residual: GT_vel - vhat_at_last_frame
    residuals = []
    for i in range(len(train_ds)):
        gt_vel = train_ds.gt_velocities[i]
        vhat   = train_ds.box_params_list[i][-1, 8:10]  # last frame vhat
        residuals.append(gt_vel - vhat)
    all_residuals = torch.stack(residuals)
    residual_mean = all_residuals.mean(0)
    residual_std  = all_residuals.std(0).clamp(min=1e-6)
    print(f"  residual_mean: {residual_mean.tolist()}")
    print(f"  residual_std:  {residual_std.tolist()}")

    box_mean      = box_mean.to(device)
    box_std       = box_std.to(device)
    residual_mean = residual_mean.to(device)
    residual_std  = residual_std.to(device)

    # Freeze BEV/crop encoders
    if FREEZE_ENC:
        for name, param in model.named_parameters():
            if name.startswith("bev_encoder") or name.startswith("crop_encoder"):
                param.requires_grad_(False)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total  = sum(p.numel() for p in model.parameters())
        print(f"Frozen {frozen:,} / {total:,} params "
              f"({100.0 * frozen / total:.1f}%); fine-tuning the rest")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim   = torch.optim.Adam(trainable_params, lr=LR)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    # Training loop with early stopping
    best_val_loss  = float("inf")
    best_state     = None
    epochs_no_impr = 0
    train_losses   = []
    val_losses     = []

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch + 1:02d}/{EPOCHS} [train]",
                    unit="batch", leave=False)
        for bev, crop, box, label in pbar:
            bev, crop, box, label = (
                bev.to(device), crop.to(device),
                box.to(device), label.to(device),
            )
            box_norm = (box[:, :, :BOX_DIM] - box_mean) / box_std
            target   = (label - box[:, -1, 8:10] - residual_mean) / residual_std

            optim.zero_grad()
            pred = model(bev, crop, box_norm)
            loss = loss_fn(pred, target)
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optim.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bev, crop, box, label in val_loader:
                bev, crop, box, label = (
                    bev.to(device), crop.to(device),
                    box.to(device), label.to(device),
                )
                box_norm = (box[:, :, :BOX_DIM] - box_mean) / box_std
                target   = (label - box[:, -1, 8:10] - residual_mean) / residual_std
                pred     = model(bev, crop, box_norm)
                val_loss += loss_fn(pred, target).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
            print(f"  ✓ new best val_loss={val_loss:.4f}")
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= PATIENCE:
                print(f"Early stopping: no improvement for {PATIENCE} epochs.")
                break

    # Restore best weights and save
    model.load_state_dict(best_state)

    # Inherit all metadata from pretrained checkpoint; override what changed
    save_ckpt = dict(base_ckpt)
    save_ckpt.update({
        "model_state":    best_state,
        "val_loss":       best_val_loss,
        "epoch":          epoch,
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "box_mean":       box_mean.cpu(),
        "box_std":        box_std.cpu(),
        "residual_mean":  residual_mean.cpu(),
        "residual_std":   residual_std.cpu(),
        "finetuned":      True,
        "finetune_ckpt":  CKPT_PATH,
        "finetune_boxes": BEV_BOXES,
        "finetune_lr":    LR,
        "finetune_n_train_windows": len(train_samples),
        "finetune_n_val_windows":   len(val_samples),
        "finetune_val_instances":   sorted(val_insts),
    })
    # Remove optimizer/scheduler state (not needed for inference)
    for drop_key in ("optim_state", "sched_state"):
        save_ckpt.pop(drop_key, None)

    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save(save_ckpt, OUT_CKPT)
    print(f"\nFine-tuned model saved → {OUT_CKPT}")
    print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
