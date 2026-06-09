import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as scipy_rotate
from torch.utils.data import Dataset
from src.box_noise import add_camera_detector_noise
from src.kalman    import kalman_velocity

# Crop extraction constants — must match save_bev.py
_BEV_SIZE     = 500
_CROP_HALF_PX = 40    # 16 m window / 2 / 0.2 m·px⁻¹ → 80 px → resize to 64
_CROP_SIZE    = 64


def _extract_crop(bev: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract a 16 m × 16 m window from (C, H, W) BEV and resize to (C, 64, 64)."""
    hp = _CROP_HALF_PX
    pad_top    = max(0, hp - row)
    pad_bottom = max(0, row + hp - bev.shape[1])
    pad_left   = max(0, hp - col)
    pad_right  = max(0, col + hp - bev.shape[2])
    if pad_top or pad_bottom or pad_left or pad_right:
        bev = np.pad(bev, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
        row += pad_top
        col += pad_left
    patch = bev[:, row - hp: row + hp, col - hp: col + hp]
    t = torch.from_numpy(patch).unsqueeze(0)  # (1, C, 80, 80)
    t = F.interpolate(t, size=(_CROP_SIZE, _CROP_SIZE), mode='bilinear', align_corners=False)
    return t.squeeze(0).numpy()


def _rotate_crop(crop: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate (C, H, W) crop so the vehicle faces canonical upward direction."""
    angle_deg = -np.degrees(yaw_rad)
    return scipy_rotate(crop, angle_deg, axes=(1, 2), reshape=False,
                        order=1, mode='constant', cval=0.0).astype(np.float32)

class BEVVelocityDataset(Dataset):
    """PyTorch dataset that serves sliding windows of T consecutive BEV frames for one vehicle instance.

    Each sample returns (bev_stack, crop_stack, box_params, label):
      bev_stack  — (T_eff * bev_ch, 500, 500) stacked full-scene BEV frames
      crop_stack — (T * 22, 64, 64)           stacked vehicle-centred crop frames
      box_params — (T, box_dim)               [x,y,z,l,w,h,yaw] + optional [dt,vx̂,vŷ] per keyframe
      label      — (2,)                        [vx, vy] ground-truth velocity at the last frame

    T_eff = T * 2 when use_subframes=True (interleaved far+near per keyframe), else T.
    bev_ch = 22 * 2 when delta_bev=True (original + frame diff channels), else 22.
    box_dim = 7 normally; 10 when add_kinematics=True (appends [dt, vx̂, vŷ] per frame, #21/#22).

    box_noise=True injects per-frame, distance-dependent, heavy-tailed noise into box_params
    to simulate detections from a camera-only 3-D bounding-box detector (see src/box_noise.py).
    Noise is re-sampled independently on every __getitem__ call so the model sees different
    perturbations each epoch.  Applied to train split only by convention; pass box_noise=False
    for val/test.
    """

    def __init__(self, meta_path, data_dir, T=3, split='train', val_last_n=10,
                 split_scenes=None, use_subframes=False, delta_bev=False,
                 add_kinematics=False, box_noise=False, box_noise_params=None,
                 use_kalman=True, rng_seed=None):
        with open(meta_path) as f:
            all_meta = json.load(f)

        all_meta = [m for m in all_meta if m['is_valid']]

        self.data_dir       = data_dir
        self.T              = T
        self.use_subframes  = use_subframes
        self.delta_bev      = delta_bev
        self.add_kinematics = add_kinematics
        self.box_noise        = box_noise
        self.box_noise_params = box_noise_params or {}
        self.use_kalman       = use_kalman and add_kinematics
        self.rng_seed         = rng_seed

        groups = {}
        for m in all_meta:
            key = (m['scene'], m['instance_token'])
            groups.setdefault(key, []).append(m)
        for key in groups:
            groups[key].sort(key=lambda x: x['frame'])

        self.samples = []

        if split_scenes is not None:
            target = set(split_scenes)
            for (scene_name, _), frames in groups.items():
                if scene_name not in target:
                    continue
                for i in range(len(frames) - T + 1):
                    window = frames[i:i + T]
                    if all(window[j+1]['frame'] == window[j]['frame'] + 1
                           for j in range(T - 1)):
                        self.samples.append(window)
        else:
            for key, frames in groups.items():
                max_frame   = frames[-1]['frame']
                split_frame = max_frame - val_last_n + 1

                for i in range(len(frames) - T + 1):
                    window = frames[i:i + T]
                    consecutive = all(
                        window[j+1]['frame'] == window[j]['frame'] + 1
                        for j in range(T - 1)
                    )
                    if not consecutive:
                        continue
                    if split == 'train' and window[-1]['frame'] < split_frame:
                        self.samples.append(window)
                    elif split == 'val' and window[0]['frame'] >= split_frame:
                        self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]

        # Full BEV + dynamic per-frame crop extraction (fix: crops now centered at
        # each frame's own vehicle position instead of the track's global last frame,
        # so the vehicle is always visible in the crop).
        bev_list = []
        crops    = []
        for frame in window:
            bev_near = np.load(f"{self.data_dir}/full_bevs/{frame['fname']}.npy").astype(np.float32, copy=False)
            if self.use_subframes:
                bev_far = np.load(f"{self.data_dir}/full_bevs_far/{frame['fname']}.npy").astype(np.float32, copy=False)
                bev_list.extend([bev_far, bev_near])  # far first (older), near second
            else:
                bev_list.append(bev_near)
            row_v, col_v = frame['box_3d']['vehicle_bev_px']
            crop = _extract_crop(bev_near, int(row_v), int(col_v))
            crops.append(_rotate_crop(crop, frame['box_3d']['yaw_ref']))

        bev_stack = np.concatenate(bev_list, axis=0)  # (T_eff * C, H, W)
        bev_tensor = torch.tensor(bev_stack, dtype=torch.float32)

        # Delta-BEV: append frame-to-frame differences as extra channels (#28)
        if self.delta_bev:
            C     = bev_list[0].shape[0]
            T_eff = len(bev_list)
            H, W  = bev_tensor.shape[1], bev_tensor.shape[2]
            frames_t = bev_tensor.reshape(T_eff, C, H, W)
            deltas   = torch.zeros_like(frames_t)
            deltas[1:] = frames_t[1:] - frames_t[:-1]
            bev_tensor = torch.cat([frames_t, deltas], dim=1).reshape(T_eff * 2 * C, H, W)

        crop_tensor = torch.tensor(np.concatenate(crops, axis=0), dtype=torch.float32)

        # Box params — (T, 7+): [x, y, z, l, w, h, yaw] + optional [dt, vx̂, vŷ] (#21/#22)
        #
        # Build in two steps:
        #   1. Noise the 7 base features (position, size, yaw) independently per frame.
        #   2. Re-derive kinematic velocity from the noisy positions so that vx̂/vŷ
        #      reflect the actual position errors a camera detector would produce.
        #      Noisy center_ref ≈ clean_center_ref + (noisy_lidar − clean_lidar),
        #      i.e. the same spatial delta is applied in the reference frame.

        # Step 1 — build and noise the 7-feature base array
        base_list = [frame['box_3d']['center_lidar'] + frame['box_3d']['dimensions']
                     + [frame['box_3d']['yaw']] for frame in window]
        base_arr = np.array(base_list, dtype=np.float32)   # (T, 7)

        if self.box_noise:
            rng = (np.random.default_rng([self.rng_seed, idx])
                   if self.rng_seed is not None
                   else np.random.default_rng())
            base_arr = add_camera_detector_noise(base_arr, rng=rng, **self.box_noise_params)

        # Step 2 — compute noisy center_ref for every keyframe
        #          noisy_ref ≈ clean_ref + (noisy_lidar − clean_lidar)
        if self.add_kinematics:
            noisy_refs = np.array([
                np.array(frame['box_3d']['center_ref'][:2])
                + (base_arr[t, :2] - np.array(frame['box_3d']['center_lidar'][:2]))
                for t, frame in enumerate(window)
            ], dtype=np.float64)                                     # (T, 2)

            timestamps = np.array([frame['timestamp'] for frame in window])

            if self.use_kalman:
                # Kalman-filtered velocity: optimal estimate given all T noisy positions.
                # When box_noise is off (val/inference), use small R (near-GT positions).
                kf_scale = (self.box_noise_params.get('noise_scale', 1.0)
                            if self.box_noise else 0.05)
                kf_vels  = kalman_velocity(noisy_refs, timestamps,
                                           noise_scale=kf_scale)  # (T, 2)
            else:
                # Fallback: naive per-step finite-diff
                kf_vels = np.zeros((len(window), 2))
                for t in range(len(window)):
                    if t == 0:
                        if len(window) > 1:
                            dt = max((window[1]['timestamp'] - window[0]['timestamp']) / 1e6, 0.1)
                            kf_vels[0] = (noisy_refs[1] - noisy_refs[0]) / dt
                    else:
                        dt = max((window[t]['timestamp'] - window[t-1]['timestamp']) / 1e6, 0.1)
                        kf_vels[t] = (noisy_refs[t] - noisy_refs[t-1]) / dt

            # Naive per-step FD from the same noisy reference-frame positions (cols 10-11).
            # Stored for display in infer.py; not passed to the model.
            fd_vels = np.zeros((len(window), 2), dtype=np.float64)
            for t in range(len(window)):
                if t == 0:
                    if len(window) > 1:
                        dt_fd = max((timestamps[1] - timestamps[0]) / 1e6, 0.1)
                        fd_vels[0] = (noisy_refs[1] - noisy_refs[0]) / dt_fd
                else:
                    dt_fd = max((timestamps[t] - timestamps[t - 1]) / 1e6, 0.1)
                    fd_vels[t] = (noisy_refs[t] - noisy_refs[t - 1]) / dt_fd

        # Step 3 — build box_params list
        box_params = []
        for t, frame in enumerate(window):
            params = base_arr[t].tolist()   # 7 noisy features

            if self.add_kinematics:
                if t == 0 and len(window) > 1:
                    dt = max((window[1]['timestamp'] - frame['timestamp']) / 1e6, 0.1)
                elif t > 0:
                    dt = max((frame['timestamp'] - window[t-1]['timestamp']) / 1e6, 0.1)
                else:
                    dt = 0.5
                params += [dt, float(kf_vels[t, 0]), float(kf_vels[t, 1]),
                           float(fd_vels[t, 0]), float(fd_vels[t, 1])]

            box_params.append(params)

        box_tensor = torch.tensor(box_params, dtype=torch.float32)

        # Label — [vx, vy] at last frame
        label = np.load(f"{self.data_dir}/labels/{window[-1]['fname']}.npy")
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return bev_tensor, crop_tensor, box_tensor, label_tensor
