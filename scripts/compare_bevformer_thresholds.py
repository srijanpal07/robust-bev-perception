"""
Aggregate and print the 2m / 5m / 10m BEVFormer matching-threshold comparison.

Run from project root:
    python scripts/compare_bevformer_thresholds.py

Reads per-threshold inference JSONs from outputs/results/ and writes
outputs/results/bevformer_threshold_comparison.txt.
"""

import json
import os
import numpy as np
from pathlib import Path

THRESHOLDS = [
    ("2m",  "outputs/results/bevformer_comparison_2m.json",
             "outputs/results/bevformer_instance_boxes.json"),
    ("5m",  "outputs/results/bevformer_comparison_5m.json",
             "outputs/results/bevformer_instance_boxes_5m.json"),
    ("10m", "outputs/results/bevformer_comparison_10m.json",
             "outputs/results/bevformer_instance_boxes_10m.json"),
]

OUT_PATH = "outputs/results/bevformer_threshold_comparison.txt"


def load(path):
    with open(path) as f:
        return json.load(f)


def stats(arr):
    return dict(mean=arr.mean(), median=np.median(arr),
                p90=np.percentile(arr, 90), p95=np.percentile(arr, 95))


def build_report():
    lines = []

    lines.append("=" * 72)
    lines.append("BEVFormer Matching Threshold Comparison")
    lines.append("Model: TemporalVelocityPredictor (residual_velocity=True, T=4)")
    lines.append("Metric: L2 velocity error (m/s) at the last frame of each window")
    lines.append("=" * 72)

    # ── per-threshold summary table ─────────────────────────────────────────
    lines.append("")
    lines.append("Matching summary (2133 total target frames across 53 val instances):")
    lines.append(
        f"  {'Threshold':<10} {'Matched frames':>15} {'Complete T=4 windows':>22}"
        f"  {'Match rate':>12}"
    )
    lines.append("  " + "-" * 62)

    rows = []
    for label, inf_path, match_path in THRESHOLDS:
        ms = load(match_path)["summary"]
        rows.append((label, ms["matched_frames"], ms["complete_windows_all_matched"],
                     inf_path, match_path))
        pct = 100.0 * ms["matched_frames"] / ms["frames_with_gt_instance"]
        lines.append(
            f"  {label:<10} {ms['matched_frames']:>15} {ms['complete_windows_all_matched']:>22}"
            f"  {pct:>11.1f}%"
        )

    lines.append("")
    lines.append("Note: 'complete windows' = consecutive T=4 frames all matched within threshold.")
    lines.append("      Larger threshold includes noisier / potentially wrong-car matches.")

    # ── velocity error comparison ────────────────────────────────────────────
    lines.append("")
    lines.append("-" * 72)
    lines.append("Velocity error (m/s) — all windows")
    lines.append("-" * 72)
    lines.append(
        f"  {'Threshold':<6}  {'n':>5}  {'Model mean':>11}  {'BEV mean':>10}"
        f"  {'Model median':>13}  {'BEV median':>11}  {'Δmean':>8}  {'%':>7}"
    )
    lines.append("  " + "-" * 70)

    for label, inf_path, _ in THRESHOLDS:
        data  = load(inf_path)
        our   = np.array([r["our_error"]       for r in data])
        bev   = np.array([r["bevformer_error"]  for r in data])
        delta = bev.mean() - our.mean()
        pct   = 100.0 * delta / bev.mean()
        lines.append(
            f"  {label:<6}  {len(data):>5}  {our.mean():>11.4f}  {bev.mean():>10.4f}"
            f"  {np.median(our):>13.4f}  {np.median(bev):>11.4f}"
            f"  {delta:>+8.4f}  {pct:>6.1f}%"
        )

    # ── percentile detail ───────────────────────────────────────────────────
    lines.append("")
    lines.append("-" * 72)
    lines.append("Percentile breakdown — Our model")
    lines.append("-" * 72)
    lines.append(f"  {'Threshold':<6}  {'Mean':>8}  {'Median':>8}  {'P90':>8}  {'P95':>8}")
    lines.append("  " + "-" * 42)
    for label, inf_path, _ in THRESHOLDS:
        data = load(inf_path)
        our  = np.array([r["our_error"] for r in data])
        s    = stats(our)
        lines.append(
            f"  {label:<6}  {s['mean']:>8.4f}  {s['median']:>8.4f}"
            f"  {s['p90']:>8.4f}  {s['p95']:>8.4f}"
        )

    lines.append("")
    lines.append("-" * 72)
    lines.append("Percentile breakdown — BEVFormer velocity")
    lines.append("-" * 72)
    lines.append(f"  {'Threshold':<6}  {'Mean':>8}  {'Median':>8}  {'P90':>8}  {'P95':>8}")
    lines.append("  " + "-" * 42)
    for label, inf_path, _ in THRESHOLDS:
        data = load(inf_path)
        bev  = np.array([r["bevformer_error"] for r in data])
        s    = stats(bev)
        lines.append(
            f"  {label:<6}  {s['mean']:>8.4f}  {s['median']:>8.4f}"
            f"  {s['p90']:>8.4f}  {s['p95']:>8.4f}"
        )

    # ── distance bucket breakdown ────────────────────────────────────────────
    lines.append("")
    lines.append("-" * 72)
    lines.append("Error by ego-distance bucket (mean / median m/s)")
    lines.append("-" * 72)
    buckets = [(0, 20), (20, 40), (40, 60), (60, 100)]

    for label, inf_path, _ in THRESHOLDS:
        data  = load(inf_path)
        our   = np.array([r["our_error"]      for r in data])
        bev   = np.array([r["bevformer_error"] for r in data])
        dists = np.array([r["dist_m"]          for r in data])
        lines.append(f"\n  Threshold = {label}:")
        lines.append(
            f"    {'Dist':>8}  {'n':>5}  {'Our mean':>10}  {'Our med':>9}"
            f"  {'BEV mean':>10}  {'BEV med':>9}"
        )
        for lo, hi in buckets:
            mask = (dists >= lo) & (dists < hi)
            if not mask.any():
                continue
            lines.append(
                f"    {lo:3d}-{hi:3d}m  {mask.sum():>5}"
                f"  {our[mask].mean():>10.3f}  {np.median(our[mask]):>9.3f}"
                f"  {bev[mask].mean():>10.3f}  {np.median(bev[mask]):>9.3f}"
            )

    lines.append("")
    lines.append("=" * 72)
    lines.append("Interpretation:")
    lines.append("  2m threshold : only genuine BEVFormer detections (correct car, <2m error).")
    lines.append("                 Cleanest comparison; smallest sample (188 windows, 12.2% match rate).")
    lines.append("  5m threshold : includes some mis-localised detections; broader coverage.")
    lines.append("  10m threshold: many wrong-car matches inflate errors for both models.")
    lines.append("")
    lines.append("  Our model is WORSE than BEVFormer velocity at all thresholds.")
    lines.append("  Root cause: the model was trained on GT+noise (synthetic camera noise)")
    lines.append("  but BEVFormer boxes have different characteristics — velocity comes")
    lines.append("  directly from the detector head, not differenced positions, and the")
    lines.append("  positional noise distribution differs from our synthetic model.")
    lines.append("  The residual corrections learned during training actively hurt")
    lines.append("  performance when applied to out-of-distribution BEVFormer inputs.")
    lines.append("")
    lines.append("  The 12.2% match rate at 2m also means BEVFormer detects only 12% of")
    lines.append("  target frames (median closest-car distance for unmatched = 12.7m).")
    lines.append("  These are fast-moving instances that camera-only detection struggles with.")
    lines.append("=" * 72)

    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    print(report)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(report + "\n")
    print(f"\nSaved → {OUT_PATH}")
