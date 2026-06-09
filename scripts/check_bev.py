"""
Sanity-checks the output of save_bev.py.

Verifies for every metadata entry:
  - full_bev, crop, crops_context, and label .npy files exist
  - array shapes are correct  : (3,500,500), (3,64,64), (3,64,64), (2,)
  - array dtypes are float32
  - BEV channels are finite and non-negative where expected
  - label [vx, vy] matches velocity_gt in metadata
  - is_valid=False entries have no saved arrays (as expected)
  - no duplicate fnames in metadata

Prints a summary and flags any issues found.

Usage:
    python scripts/check_bev.py
    python scripts/check_bev.py --data datasets/bev_data
"""

import argparse
import json
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Sanity-check save_bev.py output")
    parser.add_argument('--data', default=os.path.join('datasets', 'bev_data'),
                        help="Root bev_data directory (default: datasets/bev_data).")
    args = parser.parse_args()

    meta_path = os.path.join(args.data, 'metadata.json')
    if not os.path.exists(meta_path):
        print(f"ERROR: metadata.json not found at {meta_path}")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Loaded metadata.json — {len(metadata)} entries\n")

    issues      = []
    valid_count = 0
    skipped     = 0

    # Check for duplicate fnames
    fnames = [m['fname'] for m in metadata]
    dupes  = [f for f in set(fnames) if fnames.count(f) > 1]
    if dupes:
        issues.append(f"Duplicate fnames in metadata: {dupes}")

    for m in metadata:
        fname    = m['fname']
        is_valid = m['is_valid']

        bev_path     = os.path.join(args.data, 'full_bevs',     f"{fname}.npy")
        crop_path    = os.path.join(args.data, 'crops',         f"{fname}.npy")
        crop_ctx_path= os.path.join(args.data, 'crops_context', f"{fname}.npy")
        label_path   = os.path.join(args.data, 'labels',        f"{fname}.npy")

        if not is_valid:
            # Arrays should NOT exist for invalid frames
            for path in [bev_path, crop_path, crop_ctx_path, label_path]:
                if os.path.exists(path):
                    issues.append(f"{fname}: is_valid=False but file exists: {path}")
            skipped += 1
            continue

        # --- File existence ---
        for path in [bev_path, crop_path, crop_ctx_path, label_path]:
            if not os.path.exists(path):
                issues.append(f"{fname}: missing file {path}")
                continue

        try:
            bev      = np.load(bev_path)
            crop     = np.load(crop_path)
            crop_ctx = np.load(crop_ctx_path)
            label    = np.load(label_path)
        except Exception as e:
            issues.append(f"{fname}: failed to load arrays — {e}")
            continue

        # --- Shapes ---
        if bev.shape != (3, 500, 500):
            issues.append(f"{fname}: full_bev shape {bev.shape} != (3,500,500)")
        if crop.shape != (3, 64, 64):
            issues.append(f"{fname}: crop shape {crop.shape} != (3,64,64)")
        if crop_ctx.shape != (3, 64, 64):
            issues.append(f"{fname}: crops_context shape {crop_ctx.shape} != (3,64,64)")
        if label.shape != (2,):
            issues.append(f"{fname}: label shape {label.shape} != (2,)")

        # --- Dtypes ---
        for arr, name in [(bev, 'bev'), (crop, 'crop'), (crop_ctx, 'crops_context'), (label, 'label')]:
            if arr.dtype != np.float32:
                issues.append(f"{fname}: {name} dtype {arr.dtype} != float32")

        # --- Value checks ---
        if not np.all(np.isfinite(bev)):
            issues.append(f"{fname}: full_bev contains non-finite values")
        if bev[0].min() < 0:
            issues.append(f"{fname}: density channel has negative values")
        if np.any(np.isnan(label)):
            issues.append(f"{fname}: label has NaN but is_valid=True")

        # --- Label consistency with metadata ---
        meta_vel = m['velocity_gt']
        if not np.allclose(label, meta_vel, equal_nan=True):
            issues.append(f"{fname}: label {label} != metadata velocity_gt {meta_vel}")

        # --- BEV pixel bounds ---
        row, col = m['box_3d']['vehicle_bev_px']
        if not (0 <= row < 500 and 0 <= col < 500):
            issues.append(f"{fname}: vehicle_bev_px ({row},{col}) out of bounds")

        valid_count += 1

    # Summary
    print(f"Valid frames checked : {valid_count}")
    print(f"Invalid frames skipped (no arrays expected) : {skipped}")
    print(f"Issues found : {len(issues)}")

    if issues:
        print("\n--- ISSUES ---")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("\nAll checks passed.")

    # Quick stats on valid frames
    if valid_count > 0:
        speeds  = [m['speed_gt']        for m in metadata if m['is_valid']]
        dists   = [m['distance_to_ego'] for m in metadata if m['is_valid']]
        print(f"\n--- Stats (valid frames) ---")
        print(f"  speed_gt  : min={min(speeds):.2f}  max={max(speeds):.2f}  mean={np.mean(speeds):.2f} m/s")
        print(f"  dist      : min={min(dists):.1f}   max={max(dists):.1f}   mean={np.mean(dists):.1f} m")

        # Sample one frame and print its array info
        sample_m       = next(m for m in metadata if m['is_valid'])
        sample_bev     = np.load(os.path.join(args.data, 'full_bevs',     f"{sample_m['fname']}.npy"))
        sample_crop    = np.load(os.path.join(args.data, 'crops',         f"{sample_m['fname']}.npy"))
        sample_crop_ctx= np.load(os.path.join(args.data, 'crops_context', f"{sample_m['fname']}.npy"))
        print(f"\n--- Sample frame: {sample_m['fname']} ---")
        print(f"  full_bev      shape={sample_bev.shape}  min={sample_bev.min():.3f}  max={sample_bev.max():.3f}")
        print(f"  crop (fine)   shape={sample_crop.shape}  min={sample_crop.min():.3f}  max={sample_crop.max():.3f}")
        print(f"  crop (ctx)    shape={sample_crop_ctx.shape}  min={sample_crop_ctx.min():.3f}  max={sample_crop_ctx.max():.3f}")
        print(f"  label     {np.load(os.path.join(args.data, 'labels', sample_m['fname'] + '.npy'))}")
        print(f"  speed_gt  {sample_m['speed_gt']:.2f} m/s  dist={sample_m['distance_to_ego']:.1f} m")


if __name__ == '__main__':
    main()
