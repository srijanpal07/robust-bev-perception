"""
Create nuscenes_infos_temporal_val.pkl for BEVFormer inference.

Bypasses two limitations of BEVFormer's stock create_data.py:
  1. No nuScenes CAN bus data required — uses np.zeros(18) as fallback
     (same path BEVFormer takes for "server scenes" without CAN bus).
  2. Only processes val scenes — skips the full train pass entirely,
     reducing runtime from ~30 min to ~5 min.

Must be run from inside the bevformer/ directory so that BEVFormer's
own data_converter module is importable:

    cd bevformer
    conda activate detr3d
    python ../scripts/create_bevformer_val_infos.py \
        --root-path ./data/nuscenes \
        --out-dir   ./data/nuscenes \
        --version   v1.0-trainval
"""

import argparse
import os
import sys

# Must run from bevformer/ — add it to path so data_converter imports work
sys.path.insert(0, os.path.abspath('.'))

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nuscenes_splits
from pyquaternion import Quaternion
from mmdet3d.datasets import NuScenesDataset

from tools.data_converter.nuscenes_converter import obtain_sensor2top


class _FakeCanBus:
    """Stand-in for NuScenesCanBus that always raises, so _get_can_bus_info
    falls through to its except branch and returns np.zeros(18)."""
    def get_messages(self, *args, **kwargs):
        raise RuntimeError("no canbus data")


def _get_can_bus_info_safe(sample):
    """Returns zeros(18) — same fallback as BEVFormer uses for server scenes."""
    return np.zeros(18)


def create_val_infos(root_path, out_dir, version='v1.0-trainval', max_sweeps=10):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    if version == 'v1.0-trainval':
        val_scene_names = set(nuscenes_splits.val)
    elif version == 'v1.0-mini':
        val_scene_names = set(nuscenes_splits.mini_val)
    else:
        raise ValueError(f"Unsupported version: {version}")

    # Build val scene token set for fast lookup
    val_scene_tokens = {
        scene['token']
        for scene in nusc.scene
        if scene['name'] in val_scene_names
    }
    print(f"Val scenes: {len(val_scene_tokens)}")

    val_infos = []
    frame_idx = 0

    for sample in mmcv.track_iter_progress(nusc.sample):
        if sample['scene_token'] not in val_scene_tokens:
            continue

        # Reset frame counter at the first sample of each new scene
        if sample['prev'] == '':
            frame_idx = 0

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec      = nusc.get('sample_data', lidar_token)
        cs_record   = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        l2e_r     = cs_record['rotation']
        l2e_t     = cs_record['translation']
        e2g_r     = pose_record['rotation']
        e2g_t     = pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        info = {
            'lidar_path':           str(lidar_path),
            'token':                sample['token'],
            'prev':                 sample['prev'],
            'next':                 sample['next'],
            'can_bus':              _get_can_bus_info_safe(sample),  # zeros(18)
            'frame_idx':            frame_idx,
            'sweeps':               [],
            'cams':                 {},
            'scene_token':          sample['scene_token'],
            'lidar2ego_translation': l2e_t,
            'lidar2ego_rotation':    l2e_r,
            'ego2global_translation': e2g_t,
            'ego2global_rotation':    e2g_r,
            'timestamp':            sample['timestamp'],
        }

        frame_idx += 1
        if sample['next'] == '':
            frame_idx = 0   # will be reset at next scene's first sample

        # ── Camera calibration info ──────────────────────────────────────
        for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            cam_token = sample['data'][cam]
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'][cam] = cam_info

        # ── LiDAR sweeps ────────────────────────────────────────────────
        sd_sweep = nusc.get('sample_data', lidar_token)
        sweeps   = []
        while len(sweeps) < max_sweeps:
            if not sd_sweep['prev']:
                break
            sweep = obtain_sensor2top(
                nusc, sd_sweep['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar'
            )
            sweeps.append(sweep)
            sd_sweep = nusc.get('sample_data', sd_sweep['prev'])
        info['sweeps'] = sweeps

        # ── Annotations (needed for val metrics; same as BEVFormer converter) ──
        annotations = [nusc.get('sample_annotation', t) for t in sample['anns']]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh   for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        velocity = np.array(
            [nusc.box_velocity(t)[:2] for t in sample['anns']]
        )
        valid_flag = np.array(
            [(a['num_lidar_pts'] + a['num_radar_pts']) > 0 for a in annotations],
            dtype=bool
        ).reshape(-1)

        # Convert velocity: global → LiDAR frame (identical to BEVFormer converter)
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        # Category name mapping (same as BEVFormer converter)
        names = [b.name for b in boxes]
        for i, name in enumerate(names):
            if name in NuScenesDataset.NameMapping:
                names[i] = NuScenesDataset.NameMapping[name]
        names = np.array(names)

        # Yaw to SECOND format (same sign convention as BEVFormer converter)
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

        info['gt_boxes']      = gt_boxes
        info['gt_names']      = names
        info['gt_velocity']   = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
        info['valid_flag']    = valid_flag

        val_infos.append(info)

    print(f"Val samples collected: {len(val_infos)}")

    out_path = os.path.join(out_dir, 'nuscenes_infos_temporal_val.pkl')
    mmcv.dump(dict(infos=val_infos, metadata=dict(version=version)), out_path)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', default='./data/nuscenes')
    parser.add_argument('--out-dir',   default='./data/nuscenes')
    parser.add_argument('--version',   default='v1.0-trainval')
    parser.add_argument('--max-sweeps', type=int, default=10)
    args = parser.parse_args()

    create_val_infos(args.root_path, args.out_dir, args.version, args.max_sweeps)
