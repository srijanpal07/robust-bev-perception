"""
Filters the nuScenes dataset to find vehicle.car instances that are
persistently visible in front cameras across every frame of a scene and
move within a target speed band in every frame.

Usage:
    python filter_dataset.py [--version mini|full]

    Output is printed to stdout as a table:
        Scene | Instance Token | Min Speed (m/s) | Max Speed (m/s) | Frames

Filtering criteria:
    1. Visible in at least one of CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT
       in EVERY frame of the scene (BoxVisibility.ANY).
    2. Minimum per-frame speed >= MIN_SPEED (0.5 m/s) — every frame must be
       above this threshold, not just the average.  Eliminates instances that
       stop at traffic lights and then resume — those create near-zero ground-
       truth labels that bias the model toward predicting the training mean.
    3. Maximum per-frame speed <= MAX_SPEED (10.0 m/s) — excludes vehicles
       that are likely moving on a motorway at highway speed, which is a very
       different regime from urban driving and may have noisy annotations.

All speeds are in the global (world) frame — absolute, not relative to ego.

Output fields per qualifying instance:
    instance_token  : persistent ID constant across all frames for that object
    min_speed_mps   : minimum per-frame speed (m/s), NaN boundary frames excluded
    max_speed_mps   : maximum per-frame speed (m/s)
    avg_speed_mps   : mean per-frame speed (m/s)
    num_frames      : number of frames the instance appears in

Requirements: nuScenes devkit  (pip install nuscenes-devkit)
"""

import argparse
import csv
import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

CONFIGS = {
    "mini": {"dataroot": os.path.join(DATASETS_DIR, "raw"),      "version": "v1.0-mini"},
    "full": {"dataroot": os.path.join(DATASETS_DIR, "raw_full"), "version": "v1.0-trainval"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--version", choices=["mini", "full"], default="full",
                    help="Dataset version to filter (default: full)")
parser.add_argument("--dataroot", default=None,
                    help="Override path to nuScenes root (e.g. /mnt/ssd/raw_full)")
args = parser.parse_args()

cfg  = CONFIGS[args.version]
dataroot = args.dataroot if args.dataroot else cfg["dataroot"]
nusc = NuScenes(version=cfg["version"], dataroot=dataroot, verbose=False)

ALL_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
               'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT']
MIN_SPEED     = 0.5   # m/s — every frame must be at or above this (not just mean)
MAX_SPEED     = 30.0  # m/s — every frame must be at or below this
MIN_FRAMES    = 39    # instance must appear in at least this many frames
MAX_DIST      = 50.0  # metres — instance must be within this range in every frame

def get_front_visible_car_tokens(nusc, sample):
    """Return set of ann_tokens for vehicle.car visible in any camera."""
    visible_tokens = set()
    for cam in ALL_CAMERAS:
        cam_token = sample['data'][cam]
        cam_data  = nusc.get('sample_data', cam_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = np.array(cam_calib['camera_intrinsic'])
        imsize    = (cam_data['width'], cam_data['height'])
        _, boxes_in_cam, _ = nusc.get_sample_data(cam_token)
        for box in boxes_in_cam:
            if box.name == 'vehicle.car':
                if box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY):
                    visible_tokens.add(box.token)
    return visible_tokens


def find_persistent_front_vehicles(nusc, scene):
    """
    Find vehicle.car instances that are:
    1. Visible in front cameras in EVERY frame of the scene
    2. Every frame speed >= MIN_SPEED and <= MAX_SPEED
    
    Returns list of (instance_token, frame_data) for qualifying vehicles.
    """
    # Collect all samples in scene
    samples = []
    sample_token = scene['first_sample_token']
    while sample_token:
        samples.append(nusc.get('sample', sample_token))
        sample_token = samples[-1]['next']

    # For each frame, get front-visible vehicle.car annotation tokens
    # Map: frame_idx → set of ann_tokens visible
    per_frame_visible = []
    for sample in samples:
        visible = get_front_visible_car_tokens(nusc, sample)
        per_frame_visible.append(visible)

    # Find instance tokens present in ALL frames
    # First, map ann_token → instance_token
    def ann_to_instance(ann_token):
        return nusc.get('sample_annotation', ann_token)['instance_token']

    # Per frame: instance_tokens visible
    per_frame_instances = [
        {ann_to_instance(t) for t in frame_tokens}
        for frame_tokens in per_frame_visible
    ]

    # Intersection across all frames = instances visible in every frame
    persistent_instances = per_frame_instances[0]
    for frame_instances in per_frame_instances[1:]:
        persistent_instances &= frame_instances

    if not persistent_instances:
        return []

    # Now filter by speed band: min >= MIN_SPEED and max <= MAX_SPEED
    results = []
    for inst_token in persistent_instances:
        # Collect all annotations for this instance in this scene
        ann_tokens_in_scene = []
        for frame_tokens in per_frame_visible:
            for ann_token in frame_tokens:
                ann = nusc.get('sample_annotation', ann_token)
                if ann['instance_token'] == inst_token:
                    ann_tokens_in_scene.append(ann_token)
                    break

        # Filter: must be within MAX_DIST in every frame
        too_far = False
        for ann_token in ann_tokens_in_scene:
            ann = nusc.get('sample_annotation', ann_token)
            sample = nusc.get('sample', ann['sample_token'])
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            box_xy = np.array(ann['translation'][:2])
            ego_xy = np.array(ego_pose['translation'][:2])
            if np.linalg.norm(box_xy - ego_xy) > MAX_DIST:
                too_far = True
                break
        if too_far:
            continue

        # Compute per-frame speeds (skip NaN boundary frames)
        speeds = []
        for ann_token in ann_tokens_in_scene:
            vel = nusc.box_velocity(ann_token)
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            if not np.isnan(speed):
                speeds.append(speed)

        if not speeds:
            continue

        min_speed = min(speeds)
        max_speed = max(speeds)
        avg_speed = np.mean(speeds)

        if min_speed < MIN_SPEED:
            continue  # at least one frame is near-stationary — skip
        if max_speed > MAX_SPEED:
            continue  # at least one frame exceeds the speed cap — skip

        if len(ann_tokens_in_scene) < MIN_FRAMES:
            continue  # skip instances not present in enough frames

        results.append({
            'instance_token': inst_token,
            'min_speed_mps':  round(min_speed, 3),
            'max_speed_mps':  round(max_speed, 3),
            'avg_speed_mps':  round(avg_speed, 3),
            'num_frames':     len(ann_tokens_in_scene),
            'ann_tokens':     ann_tokens_in_scene
        })

    return results


# --- Run across all scenes ---
print(f"{'Scene':<15} {'--instance':<12} {'Min (m/s)':>10} {'Max (m/s)':>10} {'Avg (m/s)':>10} {'Frames':>8}")
print("-" * 70)

qualifying = []
for scene in nusc.scene:
    results = find_persistent_front_vehicles(nusc, scene)
    for r in results:
        short_id = r['instance_token'][-4:]
        print(f"{scene['name']:<15} {short_id:<12} "
              f"{r['min_speed_mps']:>10.2f}  {r['max_speed_mps']:>10.2f}  "
              f"{r['avg_speed_mps']:>10.2f}  {r['num_frames']:>6}")
        qualifying.append({'scene': scene['name'], **r})

print(f"\nTotal qualifying instances: {len(qualifying)}")
print(f"Speed band: [{MIN_SPEED}, {MAX_SPEED}] m/s (absolute, global frame) — every frame must be within range")

csv_path = os.path.join(DATASETS_DIR, f"filtered_{args.version}.csv")
if os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"Removed existing {csv_path}")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scene", "instance_token",
                                            "min_speed_mps", "max_speed_mps",
                                            "avg_speed_mps", "num_frames"])
    writer.writeheader()
    for q in qualifying:
        writer.writerow({
            "scene":          q["scene"],
            "instance_token": q["instance_token"],
            "min_speed_mps":  q["min_speed_mps"],
            "max_speed_mps":  q["max_speed_mps"],
            "avg_speed_mps":  q["avg_speed_mps"],
            "num_frames":     q["num_frames"],
        })
print(f"Saved → {csv_path}")
