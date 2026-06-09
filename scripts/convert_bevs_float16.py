"""
Convert full_bevs/ (and optionally full_bevs_far/) .npy files from float32 to
float16 in-place, halving disk I/O during training.

Usage:
    python scripts/convert_bevs_float16.py --data-dir /mnt/datasets/bev_data
    python scripts/convert_bevs_float16.py --data-dir /mnt/datasets/bev_data --also-far
    python scripts/convert_bevs_float16.py --data-dir /mnt/datasets/bev_data --dry-run
"""

import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def _convert_file(path: str) -> tuple[str, str]:
    """Load float32, save float16 in-place. Returns (path, status)."""
    arr = np.load(path)
    if arr.dtype == np.float16:
        return path, "already float16"
    if arr.dtype != np.float32:
        return path, f"skipped (dtype={arr.dtype})"
    tmp = path + ".tmp.npy"
    np.save(tmp, arr.astype(np.float16))
    os.replace(tmp, path)
    return path, "converted"


def convert_dir(directory: str, workers: int, dry_run: bool):
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".npy")
    )
    if not files:
        print(f"  No .npy files found in {directory}")
        return

    total = len(files)
    size_before = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"  {total} files  ({size_before:.1f} GB)  → projected {size_before/2:.1f} GB after")

    if dry_run:
        print("  Dry-run — no files written.")
        return

    converted = already = skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_convert_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            if status == "converted":
                converted += 1
            elif status == "already float16":
                already += 1
            else:
                skipped += 1
            if i % 500 == 0 or i == total:
                print(f"  [{i:>5}/{total}]  converted={converted}  already={already}  skipped={skipped}")

    size_after = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"  Done. {size_after:.1f} GB on disk (was {size_before:.1f} GB, saved {size_before-size_after:.1f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/datasets/bev_data")
    parser.add_argument("--also-far", action="store_true",
                        help="Also convert full_bevs_far/ (only needed if use_subframes=true)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing anything")
    args = parser.parse_args()

    dirs = [os.path.join(args.data_dir, "full_bevs")]
    if args.also_far:
        dirs.append(os.path.join(args.data_dir, "full_bevs_far"))

    for d in dirs:
        if not os.path.isdir(d):
            print(f"Skipping (not found): {d}")
            continue
        print(f"\nConverting {d} ...")
        convert_dir(d, workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
