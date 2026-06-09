"""
Quick sanity-check script for nuScenes datasets.
Prints the total number of scenes, then for the first 5 annotations in the
first sample: category name, 3D translation, velocity (vx, vy, vz), and
prev/next annotation links.

Usage:
    python check_dataset.py                  # mini dataset (default)
    python check_dataset.py --version full   # v1.0-trainval (part 1)

Datasets expected at:
    datasets/raw              — v1.0-mini
    datasets/raw_full_merged  — v1.0-trainval (symlinked merged layout)

Requirements: nuScenes devkit  (pip install nuscenes-devkit)
"""

import argparse
import os

from nuscenes.nuscenes import NuScenes

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

CONFIGS = {
    "mini": {
        "dataroot": os.path.join(DATASETS_DIR, "raw"),
        "version": "v1.0-mini",
    },
    "full": {
        "dataroot": os.path.join(DATASETS_DIR, "raw_full"),
        "version": "v1.0-trainval",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--version", choices=["mini", "full"], default="full", help="Dataset version to check")
args = parser.parse_args()

cfg = CONFIGS[args.version]
nusc = NuScenes(version=cfg["version"], dataroot=cfg["dataroot"], verbose=True)

print("Scenes:", len(nusc.scene))

sample = nusc.sample[0]
ann_tokens = sample["anns"]

for tok in ann_tokens[:5]:
    ann = nusc.get("sample_annotation", tok)
    vel = nusc.box_velocity(tok)
    print("category:", ann["category_name"])
    print("translation:", ann["translation"])
    print("velocity_xyz:", vel)
    print("prev:", ann["prev"], "next:", ann["next"])
    print("---")
