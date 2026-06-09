"""
Run our velocity model using BEVFormer-predicted 3D boxes as input.

The model was trained with synthetic camera noise on GT boxes.  This script
evaluates generalisation to real camera-based detections: BEVFormer boxes
replace the GT+noise boxes, and BEVFormer's own velocity prediction is used
as the kinematic estimate (vx̂/vŷ), which is what residual_velocity decodes
against.

Pipeline per T-frame window:
  1. Load LiDAR BEV maps from pre-generated files (same as normal inference).
  2. Extract crops using GT pixel positions from bev_data metadata.
  3. Build box_params[:,0:7]  = BEVFormer [x,y,z,l,w,h,yaw]
               box_params[:,7]   = dt between keyframes
               box_params[:,8:10] = BEVFormer velocity [vx,vy]  (kinematic estimate)
               box_params[:,10:12]= finite-diff from BEVFormer centres (display only)
  4. Run model; decode:  final = residual_output + BEVFormer_velocity
  5. Compare our prediction vs BEVFormer velocity vs GT.

Usage:
    # After running match_bevformer_instances.py:
    python scripts/infer_with_bevformer_boxes.py

    # Mini dataset (default):
    python scripts/infer_with_bevformer_boxes.py \
        --bevformer-boxes outputs/results/bevformer_instance_boxes.json \
        --config configs/train.yaml

    # Full dataset (needs camera blobs downloaded for val scenes):
    python scripts/infer_with_bevformer_boxes.py \
        --bevformer-boxes outputs/results/bevformer_instance_boxes_full.json \
        --no-val-filter
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from src.dataset import _extract_crop, _rotate_crop
from src.model import TemporalVelocityPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bevformer-boxes",
                        default="outputs/results/bevformer_instance_boxes.json")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--no-val-filter", action="store_true",
                        help="Evaluate all scenes in the JSON, not just official val scenes.")
    parser.add_argument("--output", default=None,
                        help="Override output JSON path (default: outputs/results/bevformer_comparison.json).")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path (default: ckpt_dir/best_model.pt from config).")
    return parser.parse_args()


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    T_model = ckpt["T"]
    T_kf    = ckpt.get("T_kf", T_model)

    model = TemporalVelocityPredictor(
        T=T_model,
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
    if any(k.startswith("_orig_mod.") for k in state):
        prefix = "_orig_mod."
        state = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt, T_kf


def build_token_to_meta(meta_path):
    with open(meta_path) as f:
        all_meta = json.load(f)
    return {m["sample_token"]: m for m in all_meta if m["is_valid"]}


def build_bev_stack(meta_win, data_dir, use_subframes, delta_bev):
    bev_list = []
    crops = []
    for meta_f in meta_win:
        bev_near = np.load(f"{data_dir}/full_bevs/{meta_f['fname']}.npy").astype(
            np.float32, copy=False
        )
        if use_subframes:
            bev_far = np.load(f"{data_dir}/full_bevs_far/{meta_f['fname']}.npy").astype(
                np.float32, copy=False
            )
            bev_list.extend([bev_far, bev_near])
        else:
            bev_list.append(bev_near)

        row_v, col_v = meta_f["box_3d"]["vehicle_bev_px"]
        crop = _extract_crop(bev_near, int(row_v), int(col_v))
        # Rotate using GT yaw_ref so the vehicle faces up — same as training
        crops.append(_rotate_crop(crop, meta_f["box_3d"]["yaw_ref"]))

    bev_tensor = torch.tensor(np.concatenate(bev_list, axis=0), dtype=torch.float32)

    if delta_bev:
        C = bev_list[0].shape[0]
        T_eff = len(bev_list)
        H, W = bev_tensor.shape[1], bev_tensor.shape[2]
        frames_t = bev_tensor.reshape(T_eff, C, H, W)
        deltas = torch.zeros_like(frames_t)
        deltas[1:] = frames_t[1:] - frames_t[:-1]
        bev_tensor = torch.cat([frames_t, deltas], dim=1).reshape(T_eff * 2 * C, H, W)

    crop_tensor = torch.tensor(np.concatenate(crops, axis=0), dtype=torch.float32)
    return bev_tensor, crop_tensor


def build_box_params(meta_win, bev_boxes, add_kinematics, T):
    """
    Assemble box_params tensor from BEVFormer predicted boxes.

    Columns:
      0-6  : [x, y, z, l, w, h, yaw]  — BEVFormer predicted box
      7    : dt                         — seconds between consecutive keyframes
      8-9  : [vx̂, vŷ]                 — BEVFormer velocity (kinematic estimate)
      10-11: [fd_vx, fd_vy]            — finite-diff from BEVFormer centres (display only)
    """
    timestamps = [m["timestamp"] for m in meta_win]
    centers = np.array([b["center_lidar"] for b in bev_boxes], dtype=np.float64)  # (T, 3)

    box_params = []
    for t, (meta_f, bev_box) in enumerate(zip(meta_win, bev_boxes)):
        c = bev_box["center_lidar"]   # [x, y, z]
        d = bev_box["dims_lwh"]       # [l, w, h]  — same convention as GT metadata
        yaw = bev_box["yaw_lidar"]
        params = [c[0], c[1], c[2], d[0], d[1], d[2], yaw]   # 7 base features

        if add_kinematics:
            # dt
            if t == 0:
                dt = max((timestamps[1] - timestamps[0]) / 1e6, 0.1) if T > 1 else 0.5
            else:
                dt = max((timestamps[t] - timestamps[t - 1]) / 1e6, 0.1)

            # BEVFormer velocity as kinematic estimate (indices 8-9)
            vx_hat, vy_hat = bev_box["velocity_lidar"]

            # Finite-diff from consecutive BEVFormer centres (indices 10-11, display only)
            if t == 0:
                dt_fd = max((timestamps[1] - timestamps[0]) / 1e6, 0.1) if T > 1 else 0.5
                fd_vx = (centers[1, 0] - centers[0, 0]) / dt_fd if T > 1 else 0.0
                fd_vy = (centers[1, 1] - centers[0, 1]) / dt_fd if T > 1 else 0.0
            else:
                dt_fd = max((timestamps[t] - timestamps[t - 1]) / 1e6, 0.1)
                fd_vx = (centers[t, 0] - centers[t - 1, 0]) / dt_fd
                fd_vy = (centers[t, 1] - centers[t - 1, 1]) / dt_fd

            params += [dt, vx_hat, vy_hat, fd_vx, fd_vy]

        box_params.append(params)

    return torch.tensor(box_params, dtype=torch.float32)  # (T, 7 or 12)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    DATA_DIR    = cfg["data"]["data_dir"]
    META_PATH   = cfg["data"]["meta_path"]
    CKPT_PATH   = args.checkpoint if args.checkpoint else os.path.join(cfg["output"]["ckpt_dir"], "best_model.pt")
    RESULTS_DIR = cfg["output"].get("results_dir", "outputs/results")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt, T = load_checkpoint(CKPT_PATH, device)
    BOX_DIM        = ckpt["box_dim"]           # 10 = 7 base + dt + vx̂ + vŷ
    ADD_KINEMATICS = ckpt.get("add_kinematics", False)
    RESIDUAL_VEL   = ckpt.get("residual_velocity", False)
    USE_SUBFRAMES  = ckpt.get("use_subframes", False)
    DELTA_BEV      = ckpt.get("delta_bev", False)

    box_mean      = ckpt["box_mean"].to(device)
    box_std       = ckpt["box_std"].to(device)
    residual_mean = ckpt.get("residual_mean", ckpt["label_mean"]).to(device)
    residual_std  = ckpt.get("residual_std",  ckpt["label_std"]).to(device)

    print(f"Model: T={T}  box_dim={BOX_DIM}  residual={RESIDUAL_VEL}  "
          f"kinematics={ADD_KINEMATICS}  device={device}")

    # If this is a fine-tuned checkpoint, restrict evaluation to the held-out
    # instances that were NOT seen during fine-tuning training.
    holdout_instances = None
    if ckpt.get("finetuned") and "finetune_val_instances" in ckpt:
        holdout_instances = set(ckpt["finetune_val_instances"])
        print(f"Fine-tuned model: restricting to {len(holdout_instances)} "
              f"held-out instances (excluded from fine-tuning training).")
    elif ckpt.get("finetuned"):
        print("WARNING: fine-tuned checkpoint has no 'finetune_val_instances' key. "
              "Evaluating on all instances — results may be inflated by leakage. "
              "Re-run finetune_bevformer.py to regenerate the checkpoint.")

    token_to_meta = build_token_to_meta(META_PATH)

    with open(args.bevformer_boxes) as f:
        bev_data = json.load(f)

    if not args.no_val_filter:
        from nuscenes.utils.splits import create_splits_scenes
        val_scenes = set(create_splits_scenes()["val"])
    else:
        val_scenes = None   # no filter

    # Group matches by (scene, instance_token), sorted by frame
    groups = {}
    for rec in bev_data["matches"]:
        if val_scenes is not None and rec["scene"] not in val_scenes:
            continue
        key = (rec["scene"], rec["instance_token"])
        groups.setdefault(key, []).append(rec)
    for key in groups:
        groups[key].sort(key=lambda r: r["frame"])

    print(f"Instances to evaluate: {len(groups)}")

    results = []
    latencies = []
    skipped_no_bev = 0
    skipped_unmatched = 0
    _first_window = True   # diagnostic flag — print convention check once then stop

    with torch.no_grad():
        for (scene, inst_tok), frames in sorted(groups.items()):
            if holdout_instances is not None and inst_tok not in holdout_instances:
                continue
            for i in range(len(frames) - T + 1):
                window = frames[i:i + T]

                # Require T consecutive frame indices
                if any(window[j + 1]["frame"] != window[j]["frame"] + 1
                       for j in range(T - 1)):
                    continue

                # All T frames must have a BEVFormer match
                if any(w["matched"] is None for w in window):
                    skipped_unmatched += 1
                    continue

                # All T frames must have pre-generated BEV data
                if any(w["sample_token"] not in token_to_meta for w in window):
                    skipped_no_bev += 1
                    continue

                meta_win  = [token_to_meta[w["sample_token"]] for w in window]
                bev_boxes = [w["matched"]["box"] for w in window]

                # ── First-window convention diagnostic ──────────────────────
                # Paste this output to verify BEVFormer box format matches GT.
                # Check: dims order, yaw sign, velocity frame, center accuracy.
                if _first_window:
                    _first_window = False
                    print("\n" + "="*65)
                    print("FIRST-WINDOW DIAGNOSTIC  (paste to Claude before trusting results)")
                    print(f"Scene: {scene}  instance: {inst_tok[:12]}  frames: "
                          f"{[w['frame'] for w in window]}")
                    print(f"{'':4s}  {'field':<22}  {'BEVFormer':>28}  {'GT (nuScenes)':>28}")
                    print("-"*65)
                    for t, (w, bbox) in enumerate(zip(window, bev_boxes)):
                        gt_b = w["gt_box"]
                        meta = meta_win[t]
                        tag  = " ← last" if t == T - 1 else ""
                        print(f"  t={t}{tag}")
                        print(f"    {'center_lidar':<22}  "
                              f"{str([round(x,2) for x in bbox['center_lidar']]):>28}  "
                              f"{str([round(x,2) for x in gt_b['center_lidar']]):>28}")
                        print(f"    {'dims (l,w,h)':<22}  "
                              f"{str([round(x,2) for x in bbox['dims_lwh']]):>28}  "
                              f"{str([round(x,2) for x in gt_b['wlh']]):>28}  ← GT is wlh")
                        print(f"    {'yaw_lidar (rad)':<22}  "
                              f"{bbox['yaw_lidar']:>28.4f}  "
                              f"{gt_b['yaw_lidar']:>28.4f}")
                        bev_v = bbox["velocity_lidar"]
                        gt_v  = meta["velocity_gt"]
                        print(f"    {'velocity (vx,vy)':<22}  "
                              f"{str([round(x,2) for x in bev_v]):>28}  "
                              f"{str([round(x,2) for x in gt_v]):>28}  ← GT vel")
                        print(f"    {'center_dist_m':<22}  "
                              f"{w['matched']['center_distance_m']:>28.3f}")
                    print("="*65 + "\n")
                # ────────────────────────────────────────────────────────────

                # BEV stack + crops
                bev_tensor, crop_tensor = build_bev_stack(
                    meta_win, DATA_DIR, USE_SUBFRAMES, DELTA_BEV
                )

                # Box params from BEVFormer
                box_tensor = build_box_params(meta_win, bev_boxes, ADD_KINEMATICS, T)

                # Batch dim + device
                bev_t  = bev_tensor.unsqueeze(0).to(device)
                crop_t = crop_tensor.unsqueeze(0).to(device)
                box_t  = box_tensor.unsqueeze(0).to(device)

                # Normalise first BOX_DIM features (fd at 10-11 are extra, not passed to model)
                box_norm = (box_t[:, :, :BOX_DIM] - box_mean) / box_std

                # Forward pass
                t0 = time.perf_counter()
                raw_pred = model(bev_t, crop_t, box_norm)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

                # Decode
                if RESIDUAL_VEL:
                    residual  = raw_pred.cpu() * residual_std.cpu() + residual_mean.cpu()
                    bev_vel_t = box_t.cpu()[0, -1, 8:10]   # BEVFormer vel at last keyframe
                    pred = (residual + bev_vel_t).numpy()[0]
                else:
                    pred = (raw_pred.cpu() * residual_std.cpu() + residual_mean.cpu()).numpy()[0]

                # GT velocity from bev_data metadata
                last_meta = meta_win[-1]
                gt = np.array(last_meta["velocity_gt"], dtype=np.float32)

                bev_vel = np.array(window[-1]["matched"]["box"]["velocity_lidar"],
                                   dtype=np.float32)

                our_err = float(np.linalg.norm(pred - gt))
                bev_err = float(np.linalg.norm(bev_vel - gt))

                # FD from BEVFormer centres (indices 10-11) for display
                fd_vel = box_t.cpu()[0, -1, 10:12].numpy() if ADD_KINEMATICS else None
                fd_err = float(np.linalg.norm(fd_vel - gt)) if fd_vel is not None else None

                dist_m = float(np.linalg.norm(last_meta["box_3d"]["center_lidar"][:2]))

                record = {
                    "scene": scene,
                    "instance_token": inst_tok,
                    "frame": window[-1]["frame"],
                    "sample_token": window[-1]["sample_token"],
                    "dist_m": dist_m,
                    "gt_vx": float(gt[0]),
                    "gt_vy": float(gt[1]),
                    "pred_vx": float(pred[0]),
                    "pred_vy": float(pred[1]),
                    "bevformer_vx": float(bev_vel[0]),
                    "bevformer_vy": float(bev_vel[1]),
                    "our_error": our_err,
                    "bevformer_error": bev_err,
                    "fd_error": fd_err,
                    "latency_ms": latencies[-1],
                }
                results.append(record)

                print(
                    f"[{scene}] fr={window[-1]['frame']:03d} "
                    f"dist={dist_m:5.1f}m  "
                    f"gt=({gt[0]:+.2f},{gt[1]:+.2f})  "
                    f"pred=({pred[0]:+.2f},{pred[1]:+.2f}) err={our_err:.3f}  "
                    f"bev=({bev_vel[0]:+.2f},{bev_vel[1]:+.2f}) err={bev_err:.3f}"
                )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.output if args.output else os.path.join(RESULTS_DIR, "bevformer_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSkipped {skipped_unmatched} windows (incomplete BEVFormer match), "
          f"{skipped_no_bev} windows (missing BEV data)")

    if not results:
        print("No results — check that bevformer_instance_boxes.json has matched frames "
              "and that bev_data metadata covers those sample tokens.")
        return

    our_errs = np.array([r["our_error"] for r in results])
    bev_errs = np.array([r["bevformer_error"] for r in results])

    lines = []
    evaluated_instances = len({r["instance_token"] for r in results})
    lines.append(f"Evaluated {len(results)} windows across {evaluated_instances} instances")
    lines.append("")
    lines.append("Our model (BEVFormer boxes + LiDAR BEV input):")
    lines.append(f"  Mean:   {our_errs.mean():.4f} m/s")
    lines.append(f"  Median: {np.median(our_errs):.4f} m/s")
    lines.append(f"  P90:    {np.percentile(our_errs, 90):.4f} m/s")
    lines.append(f"  P95:    {np.percentile(our_errs, 95):.4f} m/s")
    lines.append("")
    lines.append("BEVFormer velocity (no LiDAR, camera-only):")
    lines.append(f"  Mean:   {bev_errs.mean():.4f} m/s")
    lines.append(f"  Median: {np.median(bev_errs):.4f} m/s")
    lines.append(f"  P90:    {np.percentile(bev_errs, 90):.4f} m/s")
    lines.append(f"  P95:    {np.percentile(bev_errs, 95):.4f} m/s")

    delta = bev_errs.mean() - our_errs.mean()
    pct   = 100.0 * abs(delta) / bev_errs.mean()
    direction = "better" if delta > 0 else "worse"
    lines.append("")
    lines.append(f"Our model vs BEVFormer: {delta:+.4f} m/s  ({pct:.1f}% {direction})")
    lines.append(f"Mean latency: {np.mean(latencies):.1f} ms/sample")

    dists = np.array([r["dist_m"] for r in results])
    lines.append("\nError by distance bucket:")
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 100)]:
        mask = (dists >= lo) & (dists < hi)
        if not mask.any():
            continue
        lines.append(
            f"  {lo:3d}-{hi:3d}m  n={mask.sum():4d}"
            f"  our: mean={our_errs[mask].mean():.3f} median={np.median(our_errs[mask]):.3f}"
            f"  bev: mean={bev_errs[mask].mean():.3f} median={np.median(bev_errs[mask]):.3f}"
        )

    summary = "\n".join(lines)
    print(f"\n{'='*65}\n{summary}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(RESULTS_DIR, f"bevformer_comparison_summary_{ts}.txt")
    Path(summary_path).write_text(summary + "\n")
    print(f"\nSaved {len(results)} results → {out_path}")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
