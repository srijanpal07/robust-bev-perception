"""
Saves per-frame BEV data for a specified vehicle instance across all frames
of its scene. Intended to build the training dataset for BEV-based velocity
estimation.

Output directory structure:
    datasets/bev_data/
    ├── full_bevs/        scene-0061_frame000.npy  → (22, 500, 500) float16
    ├── crops/            scene-0061_frame000.npy  → (22, 64, 64)   float32  [fine, 10 m]
    ├── crops_context/    scene-0061_frame000.npy  → (22, 64, 64)   float32  [context, 20 m]
    ├── labels/           scene-0061_frame000.npy  → (2,)            float32  [vx, vy]
    ├── bev_channel_stats.npz                      → per-channel mean/std (#5)
    └── metadata.json                              → list of per-frame dicts

Improvements applied over baseline:
    #6  Ego-motion compensation — every frame's point cloud is transformed to the
        last (reference) frame's LiDAR coordinate system before building the BEV.
        Static objects are therefore fixed across frames; only moving objects shift.
    #9  Rotation-aligned crop — both crops are rotated by −yaw (vehicle's heading
        in the reference LiDAR frame) so the vehicle always faces a canonical
        upward direction in every crop.
    #10 Fixed-reference crop — all crops are centered on the reference frame's
        vehicle BEV position rather than each frame's own vehicle center.
        The vehicle therefore appears offset in earlier crops, directly encoding
        the trajectory.
    #11 Multi-scale crop — two crops per frame are saved: fine (10 m × 10 m)
        in crops/ and context (20 m × 20 m) in crops_context/.

New improvements in this version:
    #1  Intensity channel — mean LiDAR return intensity per BEV cell.
        Reflectivity differs between asphalt, vehicle metal, and vegetation.
    #2  Min height channel — minimum z per BEV cell. Paired with max z it
        characterises cells with ground-only returns vs. elevated objects.
    #3  Height spread channel — (max z − min z) per cell. Vehicles have large
        spread; flat ground has near-zero spread.
    #4  Z-range filtering — points outside [Z_MIN, Z_MAX] are discarded before
        building any channel, removing sky/below-ground noise.
    #5  Per-channel BEV normalization stats — mean and std are accumulated over
        all saved BEVs and written to bev_channel_stats.npz. Use in train.py /
        infer.py to normalize inputs before passing to the model.
    #7  LiDAR sweep accumulation — N_SWEEPS consecutive LiDAR scans (going back
        in time from the keyframe, ~0.5 s window) are merged into one point
        cloud after per-sweep ego-motion compensation. Produces a denser BEV,
        especially at long range where individual scans are sparse.
    #26 Height-sliced BEV — the vertical range [HEIGHT_BIN_MIN, HEIGHT_BIN_MAX]
        is divided into N_HEIGHT_BINS equal-width bins; each bin contributes one
        log-occupancy channel. Preserves vertical structure (where points are
        distributed across height) that scalar statistics collapse.

BEV array channels (22 total = 6 scalar + 16 height bins):
    Channel 0  : log1p point density
    Channel 1  : mean height (z)
    Channel 2  : max  height (z)
    Channel 3  : mean intensity              (#1)
    Channel 4  : min  height (z)             (#2)
    Channel 5  : height spread (max−min z)   (#3)
    Channels 6–21 : log1p occupancy per height bin, bins evenly spaced
                    from HEIGHT_BIN_MIN to HEIGHT_BIN_MAX (#26)

BEV coordinate convention:
    row = (BEV_RANGE - x) / VOXEL_SIZE   x forward  → top of image (row 0)
    col = (BEV_RANGE - y) / VOXEL_SIZE   y left      → left of image
    reference ego sits at pixel (BEV_SIZE//2, BEV_SIZE//2) = (250, 250)

Usage:
    python scripts/save_bev.py --instance a3f2c1
    python scripts/save_bev.py --instance a3f2c1 --scene scene-0061
    python scripts/save_bev.py --instance a3f2c1 --output datasets/bev_data

Arguments:
    --instance  (required) Instance token or suffix from filter_dataset.py output.
    --scene     Scene name (e.g. 'scene-0061'). Auto-detected from instance if omitted.
    --output    Root output directory (default: datasets/bev_data).
"""

import argparse
import shutil
import csv
import json
import os

import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# ---------------------------------------------------------------------------
# BEV parameters
# ---------------------------------------------------------------------------

BEV_RANGE              = 50.0                          # metres each side of ego
BEV_SIZE               = 500                           # pixels per side
VOXEL_SIZE             = (BEV_RANGE * 2) / BEV_SIZE   # 0.2 m/px
CROP_METRIC_SIZE       = 16.0                          # fine crop window (metres) — 80 px → downsample to 64 px (#8)
CROP_CONTEXT_METRIC_SIZE = 20.0                        # context crop window (metres)
CROP_SIZE              = 64                            # output crop size (pixels)

MIN_FRAMES = 39   # warn if instance is visible in fewer frames than this

# --- New channel parameters ---

Z_MIN          = -3.0   # z-range filter lower bound (#4)
Z_MAX          =  7.0   # z-range filter upper bound (#4)

N_HEIGHT_BINS  = 16     # number of height-slice bins per BEV cell (#26)
HEIGHT_BIN_MIN = -1.0   # metres — bottom of first bin
HEIGHT_BIN_MAX =  3.0   # metres — top of last bin

N_SWEEPS       = 10     # LiDAR sweeps to accumulate per keyframe (#7)
N_SWEEPS_NEAR  = N_SWEEPS // 2   # near sub-frame: most recent 5 sweeps (#27)
N_SWEEPS_CROP  = 3               # sweeps for crop extraction — keeps target sharp (#29)

BEV_N_CHANNELS = 6 + N_HEIGHT_BINS   # 22 total channels

# ---------------------------------------------------------------------------
# BEV construction
# ---------------------------------------------------------------------------

def build_bev(pts: np.ndarray) -> np.ndarray:
    """
    Convert (N, 4) LiDAR points [x, y, z, intensity] to a
    (BEV_N_CHANNELS, BEV_SIZE, BEV_SIZE) float32 array.

    Channel layout (BEV_N_CHANNELS = 6 + N_HEIGHT_BINS):
        0  log1p point density
        1  mean height (z)
        2  max  height (z)
        3  mean intensity              (#1)
        4  min  height (z)             (#2)
        5  height spread (max−min z)   (#3)
        6..6+N_HEIGHT_BINS-1  log1p occupancy per height bin (#26)

    #4 z-range filter: points outside [Z_MIN, Z_MAX] are dropped first.
    """
    # --- #4: z-range filter + XY range filter ---
    mask = (
        (np.abs(pts[:, 0]) < BEV_RANGE) &
        (np.abs(pts[:, 1]) < BEV_RANGE) &
        (pts[:, 2] >= Z_MIN) &
        (pts[:, 2] <= Z_MAX)
    )
    pts = pts[mask]

    rows = np.clip(((BEV_RANGE - pts[:, 0]) / VOXEL_SIZE).astype(int), 0, BEV_SIZE - 1)
    cols = np.clip(((BEV_RANGE - pts[:, 1]) / VOXEL_SIZE).astype(int), 0, BEV_SIZE - 1)
    z         = pts[:, 2]
    intensity = pts[:, 3] if pts.shape[1] >= 4 else np.zeros(len(pts), dtype=np.float32)

    density    = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    height_sum = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    height_cnt = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    max_height = np.full( (BEV_SIZE, BEV_SIZE), -np.inf, dtype=np.float32)
    min_height = np.full( (BEV_SIZE, BEV_SIZE),  np.inf, dtype=np.float32)  # #2
    intens_sum = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)            # #1

    np.add.at(density,    (rows, cols), 1.0)
    np.add.at(height_sum, (rows, cols), z)
    np.add.at(height_cnt, (rows, cols), 1.0)
    np.maximum.at(max_height, (rows, cols), z)
    np.minimum.at(min_height, (rows, cols), z)   # #2
    np.add.at(intens_sum, (rows, cols), intensity)  # #1

    occupied = height_cnt > 0

    mean_height   = np.where(occupied, height_sum / height_cnt, 0.0)
    max_height    = np.where(occupied, max_height, 0.0)
    min_height    = np.where(occupied, min_height, 0.0)          # #2
    mean_intens   = np.where(occupied, intens_sum / height_cnt, 0.0)  # #1
    height_spread = np.where(occupied, max_height - min_height, 0.0)  # #3

    # --- #26: height-bin occupancy channels ---
    # Assign each point to a bin via digitize, then scatter into a flat array.
    bin_edges = np.linspace(HEIGHT_BIN_MIN, HEIGHT_BIN_MAX, N_HEIGHT_BINS + 1)
    bin_idx   = np.clip(np.digitize(z, bin_edges) - 1, 0, N_HEIGHT_BINS - 1)
    height_bins = np.zeros((N_HEIGHT_BINS, BEV_SIZE, BEV_SIZE), dtype=np.float32)
    flat_idx    = bin_idx * (BEV_SIZE * BEV_SIZE) + rows * BEV_SIZE + cols
    np.add.at(height_bins.ravel(), flat_idx, 1.0)
    height_bins = np.log1p(height_bins)   # log-scale like density channel

    scalar_channels = [
        np.log1p(density),  # 0
        mean_height,        # 1
        max_height,         # 2
        mean_intens,        # 3  (#1)
        min_height,         # 4  (#2)
        height_spread,      # 5  (#3)
    ]
    return np.concatenate(
        [np.stack(scalar_channels, axis=0), height_bins], axis=0
    ).astype(np.float32)   # (BEV_N_CHANNELS, BEV_SIZE, BEV_SIZE)


def lidar_to_bev_px(x: float, y: float):
    """
    Convert LiDAR (x, y) coordinates to BEV (row, col) pixel indices.
    Returns (row, col) or None if outside BEV range.
    """
    row = int((BEV_RANGE - x) / VOXEL_SIZE)
    col = int((BEV_RANGE - y) / VOXEL_SIZE)
    if 0 <= row < BEV_SIZE and 0 <= col < BEV_SIZE:
        return row, col
    return None


def extract_crop(bev: np.ndarray, row: int, col: int,
                 metric_size: float = CROP_METRIC_SIZE) -> np.ndarray:
    """
    Extract a metric_size × metric_size metre window centred on (row, col)
    from the BEV, then resize to (C, CROP_SIZE, CROP_SIZE).

    Zero-pads if the crop window extends outside the BEV boundary.
    """
    half_px = int((metric_size / 2) / VOXEL_SIZE)

    r0, r1 = row - half_px, row + half_px
    c0, c1 = col - half_px, col + half_px

    pad_top    = max(0, -r0)
    pad_bottom = max(0, r1 - BEV_SIZE)
    pad_left   = max(0, -c0)
    pad_right  = max(0, c1 - BEV_SIZE)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        bev = np.pad(bev, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
        row += pad_top
        col += pad_left

    patch = bev[:, row - half_px: row + half_px,
                   col - half_px: col + half_px]

    channels = []
    for c in range(patch.shape[0]):
        img = Image.fromarray(patch[c]).resize(
            (CROP_SIZE, CROP_SIZE), Image.BILINEAR
        )
        channels.append(np.array(img, dtype=np.float32))
    return np.stack(channels, axis=0)   # (C, CROP_SIZE, CROP_SIZE)


# ---------------------------------------------------------------------------
# Ego-motion compensation (#6)
# ---------------------------------------------------------------------------

def compensate_ego_motion(pts: np.ndarray,
                          lidar_data_cur: dict,
                          lidar_data_ref: dict,
                          nusc) -> np.ndarray:
    """
    Transform (N, 3) points from the current LiDAR frame to the reference
    (last frame's) LiDAR frame.

    Pipeline: cur_lidar → cur_ego → global → ref_ego → ref_lidar
    """
    calib_cur = nusc.get('calibrated_sensor', lidar_data_cur['calibrated_sensor_token'])
    ego_cur   = nusc.get('ego_pose',           lidar_data_cur['ego_pose_token'])
    calib_ref = nusc.get('calibrated_sensor', lidar_data_ref['calibrated_sensor_token'])
    ego_ref   = nusc.get('ego_pose',           lidar_data_ref['ego_pose_token'])

    p = pts.T.copy()   # (3, N)

    # cur_lidar → cur_ego
    p = Quaternion(calib_cur['rotation']).rotation_matrix @ p
    p += np.array(calib_cur['translation'])[:, None]

    # cur_ego → global
    p = Quaternion(ego_cur['rotation']).rotation_matrix @ p
    p += np.array(ego_cur['translation'])[:, None]

    # global → ref_ego
    p -= np.array(ego_ref['translation'])[:, None]
    p = Quaternion(ego_ref['rotation']).rotation_matrix.T @ p

    # ref_ego → ref_lidar
    p -= np.array(calib_ref['translation'])[:, None]
    p = Quaternion(calib_ref['rotation']).rotation_matrix.T @ p

    return p.T   # (N, 3)


# ---------------------------------------------------------------------------
# LiDAR sweep accumulation (#7)
# ---------------------------------------------------------------------------

def accumulate_sweeps(nusc, lidar_data: dict, ref_lidar_data: dict,
                      n_sweeps: int = N_SWEEPS, skip: int = 0) -> np.ndarray:
    """
    Accumulate up to n_sweeps consecutive LiDAR scans into a single point cloud
    in the reference LiDAR frame.

    skip: number of most-recent sweeps to skip before collecting. Used to build
    the 'far' sub-frame BEV (sweeps N_SWEEPS_NEAR..N_SWEEPS-1).
    """
    all_pts = []
    current = lidar_data
    step = 0
    collected = 0

    while collected < n_sweeps:
        if step >= skip:
            pc   = LidarPointCloud.from_file(nusc.dataroot + '/' + current['filename'])
            pts4 = pc.points[:4].T
            xyz_comp = compensate_ego_motion(pts4[:, :3], current, ref_lidar_data, nusc)
            pts_comp = np.concatenate([xyz_comp, pts4[:, 3:4]], axis=1)
            all_pts.append(pts_comp)
            collected += 1
        step += 1
        if not current['prev']:
            break
        current = nusc.get('sample_data', current['prev'])

    if not all_pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(all_pts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Rotation alignment (#9)
# ---------------------------------------------------------------------------

def get_yaw_in_ref(box_orientation: Quaternion,
                   lidar_data_cur: dict,
                   lidar_data_ref: dict,
                   nusc) -> float:
    """
    Return the vehicle's yaw angle in the reference LiDAR frame.

    Computes the compound rotation from cur_lidar to ref_lidar and applies it
    to the box's own orientation quaternion.
    """
    calib_cur = nusc.get('calibrated_sensor', lidar_data_cur['calibrated_sensor_token'])
    ego_cur   = nusc.get('ego_pose',           lidar_data_cur['ego_pose_token'])
    calib_ref = nusc.get('calibrated_sensor', lidar_data_ref['calibrated_sensor_token'])
    ego_ref   = nusc.get('ego_pose',           lidar_data_ref['ego_pose_token'])

    # Rotation: cur_lidar → global → ref_lidar
    q_cur_to_ref = (
        Quaternion(calib_ref['rotation']).inverse
        * Quaternion(ego_ref['rotation']).inverse
        * Quaternion(ego_cur['rotation'])
        * Quaternion(calib_cur['rotation'])
    )
    q_box_in_ref = q_cur_to_ref * box_orientation
    return float(q_box_in_ref.yaw_pitch_roll[0])


def rotate_crop(crop: np.ndarray, yaw_rad: float) -> np.ndarray:
    """
    Rotate a (C, H, W) crop so the vehicle faces a canonical upward direction.

    In the BEV image, yaw=0 already means "facing up" (+x = row 0).
    Rotating by −yaw removes the vehicle's heading from the representation.
    PIL.rotate uses positive = CCW on screen, so angle = −yaw_degrees.
    """
    angle_deg = -np.degrees(yaw_rad)
    result = []
    for c in range(crop.shape[0]):
        img = Image.fromarray(crop[c])
        rotated = img.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
        result.append(np.array(rotated, dtype=np.float32))
    return np.stack(result, axis=0)


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

def velocity_to_lidar_frame(nusc, ann_token: str, lidar_data: dict):
    """
    Transform annotation velocity from global frame → LiDAR frame.
    Returns [vx, vy] as a list, or None if velocity is NaN.
    """
    vel_global = nusc.box_velocity(ann_token)
    if np.any(np.isnan(vel_global)):
        return None

    ego_pose = nusc.get('ego_pose',          lidar_data['ego_pose_token'])
    calib    = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    vel_ego   = Quaternion(ego_pose['rotation']).inverse.rotate(vel_global)
    vel_lidar = Quaternion(calib['rotation']).inverse.rotate(vel_ego)
    return vel_lidar[:2].tolist()   # [vx, vy]

# ---------------------------------------------------------------------------
# nuScenes helpers
# ---------------------------------------------------------------------------

def find_instance_token(nusc, suffix: str) -> str:
    """Return the full instance token matching the given suffix (or exact token)."""
    matches = [inst['token'] for inst in nusc.instance if inst['token'].endswith(suffix)]
    if not matches:
        raise ValueError(f"No instance token ending with '{suffix}' found.")
    if len(matches) > 1:
        raise ValueError(f"Multiple tokens end with '{suffix}': {matches}. Use a longer suffix.")
    return matches[0]


def get_scene_for_instance(nusc, instance_token: str) -> dict:
    """Return the scene dict containing the given instance."""
    inst      = nusc.get('instance', instance_token)
    first_ann = nusc.get('sample_annotation', inst['first_annotation_token'])
    sample    = nusc.get('sample', first_ann['sample_token'])
    return nusc.get('scene', sample['scene_token'])


def collect_ann_map(nusc, instance_token: str) -> dict:
    """
    Build {sample_token: ann_token} for every annotation of this instance.
    Traverses the annotation chain from first to last.
    """
    ann_map   = {}
    inst      = nusc.get('instance', instance_token)
    ann_token = inst['first_annotation_token']
    while ann_token:
        ann = nusc.get('sample_annotation', ann_token)
        ann_map[ann['sample_token']] = ann_token
        ann_token = ann['next']
    return ann_map

# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def _run_instance(nusc, instance_suffix: str, scene_name_override, output_dir: str):
    """Process one instance: build BEVs, crops, labels, and accumulate stats."""

    instance_token = find_instance_token(nusc, instance_suffix)

    if scene_name_override:
        matches = [s for s in nusc.scene if s['name'] == scene_name_override]
        if not matches:
            raise ValueError(f"Scene '{scene_name_override}' not found.")
        scene = matches[0]
    else:
        scene = get_scene_for_instance(nusc, instance_token)

    scene_name = scene['name']
    print(f"Instance : {instance_token}")
    print(f"Scene    : {scene_name}")

    samples, token = [], scene['first_sample_token']
    while token:
        s = nusc.get('sample', token)
        samples.append(s)
        token = s['next']
    print(f"Frames   : {len(samples)}")

    # Check whether the LiDAR data for this scene is actually on disk.
    # The metadata covers all 850 scenes but only downloaded blobs have sensor files.
    first_lidar = nusc.get('sample_data', samples[0]['data']['LIDAR_TOP'])
    first_lidar_path = os.path.join(nusc.dataroot, first_lidar['filename'])
    if not os.path.exists(first_lidar_path):
        print(f"  SKIP: LiDAR blob not downloaded for this scene ({first_lidar['filename']})")
        return False

    ann_map       = collect_ann_map(nusc, instance_token)
    visible_count = sum(1 for s in samples if s['token'] in ann_map)

    if visible_count < MIN_FRAMES:
        print(f"WARNING: instance visible in only {visible_count}/{len(samples)} frames "
              f"(expected >= {MIN_FRAMES}). Frames without the instance will be skipped.")

    full_bev_dir     = os.path.join(output_dir, 'full_bevs')
    full_bev_far_dir = os.path.join(output_dir, 'full_bevs_far')
    crop_dir         = os.path.join(output_dir, 'crops')
    crop_ctx_dir     = os.path.join(output_dir, 'crops_context')
    label_dir        = os.path.join(output_dir, 'labels')
    for d in [full_bev_dir, full_bev_far_dir, crop_dir, crop_ctx_dir, label_dir]:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------
    # Pass 1 — find the reference frame (#6, #10)
    # -------------------------------------------------------------------
    ref_lidar_data = None
    ref_row = ref_col = None

    for sample in reversed(samples):
        if sample['token'] not in ann_map:
            continue
        lidar_token_cand = sample['data']['LIDAR_TOP']
        lidar_data_cand  = nusc.get('sample_data', lidar_token_cand)
        _, boxes_cand, _ = nusc.get_sample_data(lidar_token_cand)
        ann_token_cand   = ann_map[sample['token']]
        box_cand         = next((b for b in boxes_cand if b.token == ann_token_cand), None)
        if box_cand is None:
            continue
        bev_px_cand = lidar_to_bev_px(box_cand.center[0], box_cand.center[1])
        if bev_px_cand is None:
            continue
        ref_lidar_data = lidar_data_cand
        ref_row, ref_col = bev_px_cand
        break

    if ref_lidar_data is None:
        print("WARNING: no reference frame found — falling back to BEV centre.")
        ref_row = ref_col = BEV_SIZE // 2
        ref_lidar_data = nusc.get('sample_data', samples[-1]['data']['LIDAR_TOP'])

    print(f"Reference: lidar_token={ref_lidar_data['token']}  "
          f"crop_center=({ref_row}, {ref_col})")

    # -------------------------------------------------------------------
    # Per-channel stats accumulation (#5) — loads existing file if present
    # so stats accumulate correctly across multiple instances / runs.
    # -------------------------------------------------------------------
    bev_stats_path = os.path.join(output_dir, 'bev_channel_stats.npz')
    if os.path.exists(bev_stats_path):
        _s = np.load(bev_stats_path)
        stats_pixel_count = int(_s['pixel_count'])
        stats_ch_sum      = _s['channel_sum'].astype(np.float64)
        stats_ch_sq_sum   = _s['channel_sq_sum'].astype(np.float64)
    else:
        stats_pixel_count = 0
        stats_ch_sum      = np.zeros(BEV_N_CHANNELS, dtype=np.float64)
        stats_ch_sq_sum   = np.zeros(BEV_N_CHANNELS, dtype=np.float64)

    # -------------------------------------------------------------------
    # Pass 2 — process every frame
    # -------------------------------------------------------------------
    metadata    = []
    saved_count = 0

    for frame_idx, sample in enumerate(samples):

        if sample['token'] not in ann_map:
            print(f"  frame {frame_idx:03d}: instance not present — skipping")
            continue

        ann_token = ann_map[sample['token']]
        ann       = nusc.get('sample_annotation', ann_token)

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data  = nusc.get('sample_data', lidar_token)
        pts_near = accumulate_sweeps(nusc, lidar_data, ref_lidar_data, N_SWEEPS_NEAR)
        pts_far  = accumulate_sweeps(nusc, lidar_data, ref_lidar_data, N_SWEEPS_NEAR,
                                     skip=N_SWEEPS_NEAR)
        pts_crop = accumulate_sweeps(nusc, lidar_data, ref_lidar_data, N_SWEEPS_CROP)

        bev_near = build_bev(pts_near)
        bev_far  = build_bev(pts_far)
        bev_crop = build_bev(pts_crop)

        _, boxes, _ = nusc.get_sample_data(lidar_token)
        box = next((b for b in boxes if b.token == ann_token), None)
        if box is None:
            print(f"  frame {frame_idx:03d}: box not found in LiDAR data — skipping")
            continue

        center_lidar = box.center.tolist()
        dimensions   = box.wlh[[1, 0, 2]].tolist()
        yaw_cur      = float(box.orientation.yaw_pitch_roll[0])

        center_ref = compensate_ego_motion(
            np.array(box.center[:3]).reshape(1, 3),
            lidar_data, ref_lidar_data, nusc
        )[0]
        bev_px_ref = lidar_to_bev_px(center_ref[0], center_ref[1])
        is_valid   = True  # only NaN velocity invalidates a frame

        if bev_px_ref is not None:
            row_ref, col_ref = bev_px_ref
        else:
            row_ref = col_ref = BEV_SIZE // 2
            print(f"  frame {frame_idx:03d}: vehicle outside compensated BEV range (crop still valid)")

        yaw_ref      = get_yaw_in_ref(box.orientation, lidar_data, ref_lidar_data, nusc)
        crop_row, crop_col = ref_row, ref_col

        crop_fine    = extract_crop(bev_crop, crop_row, crop_col, CROP_METRIC_SIZE)
        crop_context = extract_crop(bev_crop, crop_row, crop_col, CROP_CONTEXT_METRIC_SIZE)
        crop_fine    = rotate_crop(crop_fine,    yaw_ref)
        crop_context = rotate_crop(crop_context, yaw_ref)

        vel = velocity_to_lidar_frame(nusc, ann_token, ref_lidar_data)
        if vel is None:
            is_valid = False
            vel      = [float('nan'), float('nan')]
            print(f"  frame {frame_idx:03d}: NaN velocity — is_valid=False")

        label    = np.array(vel, dtype=np.float32)
        speed_gt = float(np.linalg.norm(label)) if is_valid else float('nan')

        ego_pose   = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        box_global = np.array(ann['translation'][:2])
        ego_global = np.array(ego_pose['translation'][:2])
        dist       = float(np.linalg.norm(box_global - ego_global))

        fname = f"{scene_name}_frame{frame_idx:03d}"

        if is_valid:
            np.save(os.path.join(full_bev_dir,     f"{fname}.npy"), bev_near.astype(np.float16))
            np.save(os.path.join(full_bev_far_dir, f"{fname}.npy"), bev_far.astype(np.float16))
            np.save(os.path.join(crop_dir,         f"{fname}.npy"), crop_fine)
            np.save(os.path.join(crop_ctx_dir,     f"{fname}.npy"), crop_context)
            np.save(os.path.join(label_dir,        f"{fname}.npy"), label)
            saved_count += 1

            flat = bev_near.reshape(BEV_N_CHANNELS, -1).astype(np.float64)
            stats_ch_sum    += flat.sum(axis=1)
            stats_ch_sq_sum += (flat ** 2).sum(axis=1)
            stats_pixel_count += flat.shape[1]

        metadata.append({
            "scene":          scene_name,
            "frame":          frame_idx,
            "fname":          fname,
            "sample_token":   sample['token'],
            "ann_token":      ann_token,
            "instance_token": instance_token,
            "box_3d": {
                "center_lidar":       center_lidar,
                "center_ref":         center_ref.tolist(),
                "dimensions":         dimensions,
                "yaw":                yaw_cur,
                "yaw_ref":            yaw_ref,
                "vehicle_bev_px":     [row_ref, col_ref],
                "crop_center_bev_px": [crop_row, crop_col],
            },
            "distance_to_ego": dist,
            "velocity_gt":     vel,
            "speed_gt":        speed_gt,
            "timestamp":       sample['timestamp'],
            "is_valid":        is_valid,
        })

        print(f"  frame {frame_idx:03d} → {fname}  "
              f"speed={speed_gt:.2f} m/s  dist={dist:.1f} m  valid={is_valid}")

    if stats_pixel_count > 0:
        ch_mean = (stats_ch_sum / stats_pixel_count).astype(np.float32)
        ch_var  = (stats_ch_sq_sum / stats_pixel_count - (stats_ch_sum / stats_pixel_count) ** 2)
        ch_std  = np.sqrt(np.maximum(ch_var, 1e-8)).astype(np.float32)
        np.savez(bev_stats_path,
                 mean=ch_mean, std=ch_std,
                 pixel_count=np.array(stats_pixel_count, dtype=np.int64),
                 channel_sum=stats_ch_sum.astype(np.float64),
                 channel_sq_sum=stats_ch_sq_sum.astype(np.float64))
        print(f"\nBEV channel stats saved → {bev_stats_path}")
        print(f"  Channels : {BEV_N_CHANNELS}  Pixels accumulated : {stats_pixel_count:,}")

    meta_path = os.path.join(output_dir, 'metadata.json')
    existing  = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)

    existing_fnames = {e['fname'] for e in existing}
    new_entries     = [m for m in metadata if m['fname'] not in existing_fnames]
    duplicates      = len(metadata) - len(new_entries)

    with open(meta_path, 'w') as f:
        json.dump(existing + new_entries, f, indent=2)

    print(f"\nDone. {saved_count} valid frames saved to {output_dir}/")
    if duplicates:
        print(f"Skipped {duplicates} duplicate entries already in metadata.json")
    print(f"metadata.json: {len(existing) + len(new_entries)} total entries → {meta_path}")
    print(f"BEV channels  : {BEV_N_CHANNELS}  (6 scalar + {N_HEIGHT_BINS} height bins)")
    print(f"Sweeps/frame (near): {N_SWEEPS_NEAR}  (far): {N_SWEEPS_NEAR}  (crop): {N_SWEEPS_CROP}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Save per-frame BEV data for a vehicle instance")
    parser.add_argument('--instance', default=None,
                        help="Instance token or suffix (from filter_dataset.py). Omit when using --batch.")
    parser.add_argument('--scene',   default=None,
                        help="Scene name (e.g. 'scene-0061'). Auto-detected from instance if omitted.")
    parser.add_argument('--output',  default=os.path.join('datasets', 'bev_data'),
                        help="Root output directory (default: datasets/bev_data).")
    parser.add_argument('--version', choices=['mini', 'full'], default='full',
                        help="Dataset version to use (default: full).")
    parser.add_argument('--batch',   action='store_true',
                        help="Process all instances from datasets/filtered_{version}.csv.")
    parser.add_argument('--resume',  action='store_true',
                        help="Skip instances already in output/metadata.json; resume an interrupted --batch run.")
    parser.add_argument('--dataroot', default=None,
                        help="Override path to nuScenes root (e.g. /mnt/ssd/raw_full).")
    args = parser.parse_args()

    if not args.instance and not args.batch:
        parser.error("Provide --instance <token> for a single instance, or --batch to process all from the filtered CSV.")
    if args.instance and args.batch:
        parser.error("--instance and --batch are mutually exclusive.")

    _datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    _configs = {
        'mini': {'dataroot': os.path.join(_datasets_dir, 'raw'),      'version': 'v1.0-mini'},
        'full': {'dataroot': os.path.join(_datasets_dir, 'raw_full'), 'version': 'v1.0-trainval'},
    }
    _cfg = _configs[args.version]
    dataroot = args.dataroot if args.dataroot else _cfg['dataroot']
    nusc = NuScenes(version=_cfg['version'], dataroot=dataroot, verbose=True)

    if args.batch:
        csv_path = os.path.join(_datasets_dir, f"filtered_{args.version}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"No filtered CSV found at {csv_path}.\n"
                f"Run: python scripts/filter_dataset.py --version {args.version}"
            )
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        print(f"Batch mode: {len(rows)} instances from {csv_path}")

        done_instances = set()
        if args.resume:
            meta_path = os.path.join(args.output, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    done_instances = {e['instance_token'] for e in json.load(f)}
                print(f"Resume: {len(done_instances)} instances already done, skipping them.\n")
            else:
                print("Resume: no metadata.json found, starting fresh.\n")
        elif os.path.exists(args.output):
            print(f"Removing existing output directory: {args.output}")
            shutil.rmtree(args.output)
            print("Removed.\n")

        processed = skipped = resumed = 0
        for i, row in enumerate(rows):
            if row['instance_token'] in done_instances:
                resumed += 1
                print(f"[{i+1}/{len(rows)}] Already done — skipping instance=...{row['instance_token'][-6:]}")
                continue
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(rows)}] scene={row['scene']}  "
                  f"instance=...{row['instance_token'][-6:]}  "
                  f"speed={row['avg_speed_mps']} m/s  frames={row['num_frames']}")
            print('='*60)
            result = _run_instance(nusc, row['instance_token'], row['scene'], args.output)
            if result is False:
                skipped += 1
            else:
                processed += 1
        print(f"\nBatch complete: {processed} processed, {skipped} skipped (blob not downloaded), {resumed} already done.")
    else:
        _run_instance(nusc, args.instance, args.scene, args.output)


if __name__ == '__main__':
    main()
