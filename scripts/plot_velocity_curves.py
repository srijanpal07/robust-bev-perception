"""
plot_velocity_curves.py

Generates two velocity-over-time plots:

  1. Normal model  — GT / FD / pretrained model (predictions.json)
     Instance selected as the one with lowest prediction variance among those
     with >= 20 consecutive predicted windows.

  2. BEVFormer     — GT / FD / fine-tuned model / BEVFormer
                     (bevformer_comparison_ft.json)
     Instance with the longest consecutive frame span.

FD velocity is computed from consecutive center_ref positions in the metadata
(no box noise — clean GT positions).  All velocities are shown as speed
(L2 norm) in m/s.

Outputs:
  outputs/plots/velocity_curves_normal.png
  outputs/plots/velocity_curves_bevformer.png
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import BEVVelocityDataset


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_meta_lookup(meta_path):
    """Return dict: (scene, instance_token, frame) -> metadata entry."""
    with open(meta_path) as f:
        all_meta = json.load(f)
    lookup = {}
    for m in all_meta:
        if m["is_valid"]:
            lookup[(m["scene"], m["instance_token"], m["frame"])] = m
    return lookup


def compute_fd_speeds(meta_lookup, scene, instance_token, frames):
    """
    Compute FD speed at each frame using consecutive center_ref positions.
    Returns dict frame -> fd_speed (np.nan for the first frame in the list).
    """
    fd = {}
    sorted_frames = sorted(frames)
    for i, f in enumerate(sorted_frames):
        if i == 0:
            fd[f] = np.nan
            continue
        prev_f = sorted_frames[i - 1]
        m_cur  = meta_lookup.get((scene, instance_token, f))
        m_prev = meta_lookup.get((scene, instance_token, prev_f))
        if m_cur is None or m_prev is None:
            fd[f] = np.nan
            continue
        dt = (m_cur["timestamp"] - m_prev["timestamp"]) * 1e-6  # µs → s
        if dt <= 0:
            fd[f] = np.nan
            continue
        cr_cur  = np.array(m_cur["box_3d"]["center_ref"][:2])
        cr_prev = np.array(m_prev["box_3d"]["center_ref"][:2])
        fd[f] = float(np.linalg.norm((cr_cur - cr_prev) / dt))
    return fd


def longest_consecutive_run(frames_list):
    """Return (start_frame, length) of the longest consecutive run."""
    if not frames_list:
        return None, 0
    frames = sorted(set(frames_list))
    best_start, best_len = frames[0], 1
    cur_start, cur_len   = frames[0], 1
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            cur_len += 1
            if cur_len > best_len:
                best_len  = cur_len
                best_start = cur_start
        else:
            cur_start = frames[i]
            cur_len   = 1
    return best_start, best_len


def style_plot(ax, title, xlabel="Frame index", ylabel="Speed (m/s)"):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Normal model plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_normal_model(meta_lookup, meta_path, data_dir, ckpt_path, pred_path,
                      out_path, min_run=20):
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    T             = ckpt.get("T_kf", ckpt["T"])
    VAL_LAST_N    = ckpt.get("val_last_n", 5)
    VAL_SCENES    = ckpt.get("val_scenes", None)
    USE_SUBFRAMES = ckpt.get("use_subframes", False)
    DELTA_BEV     = ckpt.get("delta_bev", False)
    ADD_KIN       = ckpt.get("add_kinematics", False)
    USE_KALMAN    = ckpt.get("use_kalman", True)

    dataset = BEVVelocityDataset(
        meta_path=meta_path,
        data_dir=data_dir,
        T=T,
        split="val",
        val_last_n=VAL_LAST_N,
        split_scenes=VAL_SCENES,
        use_subframes=USE_SUBFRAMES,
        delta_bev=DELTA_BEV,
        add_kinematics=ADD_KIN,
        box_noise=False,
        use_kalman=USE_KALMAN,
        rng_seed=ckpt.get("seed", 42),
    )

    with open(pred_path) as f:
        preds = json.load(f)
    pred_by_idx = {p["sample_idx"]: p for p in preds}

    # Build per-instance summary: consecutive run length + pred variance
    sample_info = []
    for i, window in enumerate(dataset.samples):
        last = window[-1]
        sample_info.append({
            "sample_idx":     i,
            "scene":          last["scene"],
            "instance_token": last["instance_token"],
            "last_frame":     last["frame"],
        })

    runs      = collections.defaultdict(list)
    pred_spds = collections.defaultdict(list)
    for si in sample_info:
        key = (si["scene"], si["instance_token"])
        runs[key].append(si["last_frame"])
        p = pred_by_idx.get(si["sample_idx"])
        if p:
            pred_spds[key].append(float(np.linalg.norm([p["pred_vx"], p["pred_vy"]])))

    # Pick instance with lowest prediction variance among those with run >= min_run
    best_key, best_start, best_var = None, 0, float("inf")
    for key, lfs in runs.items():
        start, length = longest_consecutive_run(lfs)
        if length < min_run:
            continue
        spds = pred_spds[key]
        if not spds:
            continue
        var = float(np.var(spds))
        if var < best_var:
            best_var, best_start, best_key = var, start, key
            best_len = length

    best_scene, best_inst = best_key
    best_lfs = sorted(f for f in runs[best_key]
                      if best_start <= f < best_start + best_len)

    print(f"[normal] selected: {best_scene}, ...{best_inst[-4:]} "
          f"(run={best_len}, pred_var={best_var:.3f})")

    all_meta_frames = sorted(
        frame for (sc, inst, frame) in meta_lookup
        if sc == best_scene and inst == best_inst
    )
    fd_map = compute_fd_speeds(meta_lookup, best_scene, best_inst, all_meta_frames)

    frames, gt_spd, fd_spd, pred_spd = [], [], [], []
    for si in sample_info:
        if si["scene"] != best_scene or si["instance_token"] != best_inst:
            continue
        lf = si["last_frame"]
        if lf not in set(best_lfs):
            continue
        p = pred_by_idx.get(si["sample_idx"])
        if p is None:
            continue
        frames.append(lf)
        gt_spd.append(float(np.linalg.norm([p["gt_vx"],   p["gt_vy"]])))
        fd_spd.append(fd_map.get(lf, np.nan))
        pred_spd.append(float(np.linalg.norm([p["pred_vx"], p["pred_vy"]])))

    order    = np.argsort(frames)
    frames   = np.array(frames)[order]
    gt_spd   = np.array(gt_spd)[order]
    fd_spd   = np.array(fd_spd)[order]
    pred_spd = np.array(pred_spd)[order]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frames, gt_spd,   color="#2c7bb6", lw=2.0,
            linestyle="-",  label="Ground truth")
    ax.plot(frames, fd_spd,   color="#d7191c", lw=1.8,
            linestyle="--", label="FD baseline")
    ax.plot(frames, pred_spd, color="#1a9641", lw=1.8,
            linestyle="--", label="Our model")

    style_plot(ax,
               f"Velocity over time  |  Target vehicle ID: {best_inst[-4:]}\n"
               f"{best_scene}  ·  {best_len} consecutive windows  "
               f"(frames {best_lfs[0]}–{best_lfs[-1]})")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[normal] saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# BEVFormer comparison plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_bevformer(meta_lookup, bev_result_path, out_path):
    with open(bev_result_path) as f:
        data = json.load(f)

    runs = collections.defaultdict(list)
    for r in data:
        runs[(r["scene"], r["instance_token"])].append(r["frame"])

    best_key, best_start, best_len = None, 0, 0
    for key, frames in runs.items():
        start, length = longest_consecutive_run(frames)
        if length > best_len:
            best_len, best_start, best_key = length, start, key

    best_scene, best_inst = best_key
    best_frames = sorted(
        f for f in runs[best_key]
        if best_start <= f < best_start + best_len
    )

    print(f"[bevformer] selected: {best_scene}, ...{best_inst[-4:]} "
          f"(consecutive={best_len})")

    results_map = {
        r["frame"]: r
        for r in data
        if r["scene"] == best_scene and r["instance_token"] == best_inst
        and r["frame"] in set(best_frames)
    }

    all_meta_frames = sorted(
        frame for (sc, inst, frame) in meta_lookup
        if sc == best_scene and inst == best_inst
    )
    fd_map = compute_fd_speeds(meta_lookup, best_scene, best_inst, all_meta_frames)

    frames, gt_spd, fd_spd, pred_spd, bev_spd = [], [], [], [], []
    for f in sorted(results_map):
        r = results_map[f]
        frames.append(f)
        gt_spd.append(float(np.linalg.norm([r["gt_vx"],         r["gt_vy"]])))
        fd_spd.append(fd_map.get(f, np.nan))
        pred_spd.append(float(np.linalg.norm([r["pred_vx"],     r["pred_vy"]])))
        bev_spd.append(float(np.linalg.norm([r["bevformer_vx"], r["bevformer_vy"]])))

    frames   = np.array(frames)
    gt_spd   = np.array(gt_spd)
    fd_spd   = np.array(fd_spd)
    pred_spd = np.array(pred_spd)
    bev_spd  = np.array(bev_spd)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frames, gt_spd,   color="#2c7bb6", lw=2.0,
            linestyle="-",  label="Ground truth")
    ax.plot(frames, fd_spd,   color="#d7191c", lw=1.8,
            linestyle="--", label="FD baseline")
    ax.plot(frames, pred_spd, color="#1a9641", lw=1.8,
            linestyle="--", label="Our model (fine-tuned)")
    ax.plot(frames, bev_spd,  color="#ff7f00", lw=1.8,
            linestyle="--", label="BEVFormer")

    style_plot(ax,
               f"Velocity over time  |  Target vehicle ID: {best_inst[-4:]}\n"
               f"{best_scene}  ·  {best_len} consecutive windows  "
               f"(frames {best_frames[0]}–{best_frames[-1]})")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[bevformer] saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot velocity curves.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--ckpt",   default=None,
                        help="Pretrained checkpoint (default: ckpt_dir/best_model.pt).")
    parser.add_argument("--bev_results",
                        default="outputs/results/bevformer_comparison_ft.json")
    parser.add_argument("--pred_results",
                        default="outputs/results/predictions.json")
    parser.add_argument("--out_dir", default="outputs/plots")
    parser.add_argument("--min_run", type=int, default=20,
                        help="Min consecutive windows for normal model instance selection.")
    parser.add_argument("--skip_normal",    action="store_true")
    parser.add_argument("--skip_bevformer", action="store_true")
    args = parser.parse_args()

    cfg       = yaml.safe_load(open(args.config))
    meta_path = cfg["data"]["meta_path"]
    ckpt_path = args.ckpt or os.path.join(cfg["output"]["ckpt_dir"], "best_model.pt")

    print("Loading metadata lookup…")
    meta_lookup = load_meta_lookup(meta_path)
    print(f"  {len(meta_lookup)} valid entries loaded")

    if not args.skip_normal:
        plot_normal_model(
            meta_lookup,
            meta_path,
            cfg["data"]["data_dir"],
            ckpt_path,
            args.pred_results,
            os.path.join(args.out_dir, "velocity_curves_normal.png"),
            min_run=args.min_run,
        )

    if not args.skip_bevformer:
        plot_bevformer(
            meta_lookup,
            args.bev_results,
            os.path.join(args.out_dir, "velocity_curves_bevformer.png"),
        )


if __name__ == "__main__":
    main()
