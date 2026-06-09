"""
Merges the extracted nuScenes trainval directory structure into the flat
layout expected by the nuScenes devkit.

Each blob and the metadata archive extract into their own top-level folder.
This script flattens them so the devkit can find everything under one root:

    raw_full/
      v1.0-trainval/    <- JSON metadata (from v1.0-trainval_meta/)
      maps/             <- map files     (from v1.0-trainval_meta/)
      samples/          <- sensor data   (merged from all v1.0-trainvalXX_blobs/)
      sweeps/           <- sweep data    (merged from all v1.0-trainvalXX_blobs/)

Usage:
    python scripts/prepare_dataset.py                     # merge only
    python scripts/prepare_dataset.py --prune             # merge + delete non-LiDAR sensor data
    python scripts/prepare_dataset.py --prune --dry-run   # preview what --prune would delete
    python scripts/prepare_dataset.py /path/to/raw_full   # explicit path

The merge step is idempotent — safe to re-run as you download more blob parts.
Already-merged files are left in place.

--prune deletes all sensor directories under samples/ and sweeps/ EXCEPT
LIDAR_TOP.  Camera images and radar data are not used by the BEV pipeline so
this can recover tens of GB of disk space.  THIS IS IRREVERSIBLE — only run it
after you have confirmed the merge is complete and you no longer need the raw
images.  Use --dry-run first to see what will be removed.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Only LiDAR is used to build BEV representations.
# Camera images are referenced by the devkit metadata but their actual files
# are never read in filter_dataset.py or save_bev.py.
KEEP_SENSORS = {"LIDAR_TOP"}


def merge_dir(src: Path, dst: Path) -> int:
    """Move all files from src into dst. Returns count of files moved."""
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in src.iterdir():
        target = dst / f.name
        if not target.exists():
            f.rename(target)
            moved += 1
    return moved


def remove_if_empty(path: Path):
    leftovers = [p for p in path.iterdir() if p.name != "LICENSE" and not p.name.startswith(".")]
    if not leftovers:
        shutil.rmtree(path)
        print(f"  Removed empty dir: {path.name}/")


def prune_unused_sensors(raw_full: Path, dry_run: bool = False):
    """Delete every sensor directory under samples/ and sweeps/ except LIDAR_TOP."""
    to_delete = []
    for split in ("samples", "sweeps"):
        split_dir = raw_full / split
        if not split_dir.exists():
            continue
        for sensor_dir in sorted(split_dir.iterdir()):
            if sensor_dir.is_dir() and sensor_dir.name not in KEEP_SENSORS:
                size = sum(f.stat().st_size for f in sensor_dir.rglob("*") if f.is_file())
                to_delete.append((sensor_dir, size))

    if not to_delete:
        print("\nPrune: nothing to remove (only LIDAR_TOP found or directories already clean).")
        return

    total_gb = sum(s for _, s in to_delete) / 1e9
    print(f"\nPrune: {len(to_delete)} sensor directories to delete  (~{total_gb:.1f} GB total):")
    for path, size in to_delete:
        print(f"  {path.relative_to(raw_full)}  ({size / 1e6:.0f} MB)")

    if dry_run:
        print("\n[dry-run] No files deleted. Re-run without --dry-run to actually delete.")
        return

    print()
    for path, size in to_delete:
        shutil.rmtree(path)
        print(f"  Deleted  {path.relative_to(raw_full)}  ({size / 1e6:.0f} MB)")
    print(f"\nFreed ~{total_gb:.1f} GB.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge nuScenes blob directories and optionally prune non-LiDAR sensor data.")
    parser.add_argument("dataroot", nargs="?", default=None,
                        help="Path to raw_full/ (default: datasets/raw_full relative to repo root)")
    parser.add_argument("--prune", action="store_true",
                        help="Delete all sensor data except LIDAR_TOP after merging (IRREVERSIBLE)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --prune: show what would be deleted without deleting anything")
    args = parser.parse_args()

    if args.dataroot:
        raw_full = Path(args.dataroot)
    else:
        raw_full = Path(__file__).parent.parent / "datasets" / "raw_full"

    if not raw_full.exists():
        print(f"ERROR: directory not found: {raw_full}")
        sys.exit(1)

    print(f"Target directory: {raw_full}\n")

    # ── 1. Metadata ────────────────────────────────────────────────────────
    meta_dir = raw_full / "v1.0-trainval_meta"
    if meta_dir.exists():
        print("Processing metadata...")
        for item in ("v1.0-trainval", "maps"):
            src = meta_dir / item
            dst = raw_full / item
            if dst.exists():
                print(f"  Already present: {item}/")
            elif src.exists():
                src.rename(dst)
                print(f"  Moved: {item}/")
        remove_if_empty(meta_dir)
    else:
        if (raw_full / "v1.0-trainval").exists():
            print("Metadata already merged.")
        else:
            print("WARNING: no v1.0-trainval_meta/ found — download the metadata archive first.")

    # ── 2. Blobs ───────────────────────────────────────────────────────────
    blob_dirs = sorted(raw_full.glob("v1.0-trainval*_blobs"))
    if not blob_dirs:
        print("\nNo blob directories found — nothing to merge.")
    else:
        for blob_dir in blob_dirs:
            print(f"\nProcessing {blob_dir.name}/")
            for split in ("samples", "sweeps"):
                src_split = blob_dir / split
                dst_split = raw_full / split
                if not src_split.exists():
                    continue
                dst_split.mkdir(exist_ok=True)
                for sensor_dir in sorted(src_split.iterdir()):
                    dst_sensor = dst_split / sensor_dir.name
                    if not dst_sensor.exists():
                        sensor_dir.rename(dst_sensor)
                        print(f"  Moved   {split}/{sensor_dir.name}/")
                    else:
                        n = merge_dir(sensor_dir, dst_sensor)
                        if n:
                            print(f"  Merged  {split}/{sensor_dir.name}/  ({n} files)")
                        else:
                            print(f"  Skipped {split}/{sensor_dir.name}/  (already up to date)")
                        if not any(sensor_dir.iterdir()):
                            sensor_dir.rmdir()
                try:
                    src_split.rmdir()
                except OSError:
                    pass
            remove_if_empty(blob_dir)

    # ── 3. Summary ─────────────────────────────────────────────────────────
    print("\nFinal layout:")
    for item in sorted(raw_full.iterdir()):
        if item.is_dir():
            print(f"  {item.name}/")
        else:
            print(f"  {item.name}")

    required = ["v1.0-trainval", "maps", "samples", "sweeps"]
    missing = [r for r in required if not (raw_full / r).exists()]
    if missing:
        print(f"\nWARNING: still missing: {missing}")
        print("Download and extract the corresponding archives, then re-run this script.")
    else:
        print("\nReady — run: python scripts/check_dataset.py --version full")

    # ── 4. Optional prune ──────────────────────────────────────────────────
    if args.prune or args.dry_run:
        prune_unused_sensors(raw_full, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
