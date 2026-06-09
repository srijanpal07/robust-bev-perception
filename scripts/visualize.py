"""
Visualizes a nuScenes mini dataset scene using Rerun.
Logs LiDAR point clouds, all 6 camera images, 3D bounding boxes, and
velocity arrows for each sample frame in the scene.

Usage:
    # Annotate ALL vehicles (default, 360°, LiDAR-space):
    python visualize.py

    # Annotate only front-visible vehicles (CAM_FRONT / FRONT_LEFT / FRONT_RIGHT):
    python visualize.py --mode front

    # Annotate only the single closest moving vehicle in front of the ego:
    python visualize.py --mode closest_front

    # Visualize a specific scene by name:
    python visualize.py --scene scene-0061

    # Annotate a single vehicle by its instance token (full or suffix):
    python visualize.py --instance a3f2c1
    python visualize.py --mode front --instance a3f2c1

    # Isolate a single vehicle — only that box + ego marker + cropped LiDAR:
    python visualize.py --instance-only a3f2c1
    python visualize.py --instance-only a3f2c1 --scene scene-0061

Arguments:
    --mode          'all' (default) : draw 3D boxes + velocity for every vehicle in the scene
                    'front'         : draw 3D boxes + velocity only for front-camera-visible vehicles
                    'closest_front' : draw 3D boxes + velocity for the nearest moving vehicle
                                      visible in a front camera (speed > 0.5 m/s)
    --scene         Scene name to visualize (e.g. 'scene-0061'). Defaults to scene index 7.
    --instance      Instance token (or suffix) of a specific vehicle to annotate.
                    Get the token from filter_dataset.py output. Applied on top of --mode.
                    Recommend using --mode all with --instance so the vehicle is tracked
                    across all frames regardless of camera visibility. Using --mode front
                    with --instance may cause the vehicle to disappear in frames where it
                    is not visible in a front camera.
    --instance-only Instance token (or suffix) to highlight a single vehicle. All 6
                    camera feeds and the full LiDAR scene are kept. LiDAR points inside
                    the vehicle's 3D bounding box are shown in bright orange (larger);
                    background points are shown in dim grey (smaller). Ego vehicle marker
                    is logged at the LiDAR origin. Uses geometric box masking —
                    nuScenes mini does not include point-level instance labels.
                    Overrides --mode and --instance.

Velocity:
    nuScenes provides per-annotation velocity (m/s) estimated from consecutive
    frames.  It is logged as 3D arrows in LiDAR space and appended to each
    box label as "[<id>] <class>  X.X m/s".  The short ID (last 4 chars of the
    instance_token) is constant for the same vehicle across all frames in a scene.
    Stationary or NaN velocities are skipped.

Requirements: nuScenes devkit, Rerun  (pip install rerun-sdk nuscenes-devkit)
Dataset must be present at:
    <project_root>/datasets/raw/  (version v1.0-mini)
"""

import argparse
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
from pyquaternion import Quaternion
import numpy as np
from PIL import Image
import rerun as rr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRONT_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
ALL_CAMERAS   = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT']

VEHICLE_CLASSES = {
    'vehicle.car',
    'vehicle.truck',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.motorcycle',
    'vehicle.bicycle',
    'vehicle.trailer',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_front_visible_vehicle_tokens(nusc, sample):
    """Return annotation tokens for vehicles visible in any front camera."""
    visible_tokens = set()
    for cam in FRONT_CAMERAS:
        cam_token = sample['data'][cam]
        cam_data  = nusc.get('sample_data', cam_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        intrinsic = np.array(cam_calib['camera_intrinsic'])
        imsize    = (cam_data['width'], cam_data['height'])

        _, boxes_in_cam, _ = nusc.get_sample_data(cam_token)
        for box in boxes_in_cam:
            if not box.name.startswith('vehicle.'):
                continue
            if box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY):
                visible_tokens.add(box.token)

    return visible_tokens


def filter_boxes(boxes, mode, nusc=None, sample=None, lidar_data=None):
    """Return only the boxes that should be annotated for the given mode."""
    if mode == 'all':
        return [b for b in boxes if b.name in VEHICLE_CLASSES]

    visible_tokens = get_front_visible_vehicle_tokens(nusc, sample)
    front_boxes = [b for b in boxes if b.token in visible_tokens]

    if mode == 'front':
        return front_boxes

    # mode == 'closest_front': single nearest moving vehicle in front of ego
    MOVING_THRESHOLD = 0.9  # m/s — below this a vehicle is considered stationary

    best_box  = None
    best_dist = float('inf')
    for box in front_boxes:
        vel = velocity_in_lidar_frame(nusc, box.token, lidar_data)
        if vel is None:
            continue
        if np.linalg.norm(vel[:2]) < MOVING_THRESHOLD:
            continue
        # horizontal distance from LiDAR origin (= ego position in LiDAR frame)
        dist = float(np.linalg.norm(box.center[:2]))
        if dist < best_dist:
            best_dist = dist
            best_box  = box

    return [best_box] if best_box is not None else []


def velocity_in_lidar_frame(nusc, ann_token, lidar_data):
    """
    Return the annotation velocity rotated into the LiDAR sensor frame.

    nusc.box_velocity() gives velocity in the global/world frame.
    We apply the inverse ego-pose rotation then the inverse lidar-sensor
    rotation to bring it into LiDAR space (translation is irrelevant for
    a pure velocity vector).  Returns None when velocity is NaN.
    """
    vel_global = nusc.box_velocity(ann_token)  # (vx, vy, vz) in m/s, global frame
    if np.any(np.isnan(vel_global)):
        return None

    ego_pose   = nusc.get('ego_pose',          lidar_data['ego_pose_token'])
    calib      = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    # global → ego frame
    vel_ego   = Quaternion(ego_pose['rotation']).inverse.rotate(vel_global)
    # ego → lidar frame
    vel_lidar = Quaternion(calib['rotation']).inverse.rotate(vel_ego)

    return vel_lidar

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="nuScenes Rerun visualizer")
    parser.add_argument(
        '--mode',
        choices=['all', 'front', 'closest_front'],
        default='all',
        help=(
            "'all' (default) — annotate every vehicle in the scene (360°); "
            "'front'         — annotate only front-camera-visible vehicles; "
            "'closest_front' — annotate only the nearest moving vehicle in front of ego"
        ),
    )
    parser.add_argument(
        '--scene',
        default=None,
        help="Scene name to visualize (e.g. 'scene-0061'). Defaults to scene index 7.",
    )
    parser.add_argument(
        '--instance',
        default=None,
        help=(
            "Instance token of a specific vehicle to annotate (e.g. from filter_dataset.py output). "
            "Can be a full token or a suffix match (last N chars). "
            "Applied on top of --mode. Recommended: use --mode all (default) so the vehicle "
            "is tracked across all frames. Using --mode front may cause it to disappear "
            "in frames where it is not front-visible."
        ),
    )
    parser.add_argument(
        '--instance-only',
        default=None,
        metavar='INSTANCE',
        help=(
            "Instance token (or suffix) of a single vehicle to isolate. "
            "Shows only that vehicle's box + the ego vehicle marker. "
            "LiDAR points are cropped to a 20 m radius around the vehicle. "
            "Camera feeds and all other vehicles are hidden. "
            "Overrides --mode and --instance."
        ),
    )
    args = parser.parse_args()

    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "raw"),
        verbose=True,
    )

    rr.init("nuscenes", spawn=True)

    if args.scene:
        matches = [s for s in nusc.scene if s['name'] == args.scene]
        if not matches:
            available = [s['name'] for s in nusc.scene]
            raise ValueError(f"Scene '{args.scene}' not found. Available: {available}")
        scene = matches[0]
    else:
        scene = nusc.scene[7]
    print(f"Visualizing scene: {scene['name']}")
    sample_token = scene['first_sample_token']
    i = 0

    while sample_token:
        sample = nusc.get('sample', sample_token)
        rr.set_time_sequence("frame", i)

        # --- LiDAR ---
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data  = nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(nusc.dataroot + '/' + lidar_data['filename'])
        pts = pc.points[:3].T  # (N, 3)

        # --- All 6 cameras ---
        for cam in ALL_CAMERAS:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            img = np.array(Image.open(nusc.dataroot + '/' + cam_data['filename']))
            rr.log(f"camera/{cam}", rr.Image(img))

        # --- 3D bounding boxes + velocity arrows ---
        _, boxes, _ = nusc.get_sample_data(lidar_token)

        if args.instance_only:
            selected = [
                b for b in boxes
                if nusc.get('sample_annotation', b.token)['instance_token'].endswith(args.instance_only)
            ]

            # Build vehicle-point mask using the oriented 3D bounding box
            vehicle_mask = np.zeros(len(pts), dtype=bool)
            if selected:
                box = selected[0]
                pts_local = pts - box.center
                pts_local = (box.orientation.inverse.rotation_matrix @ pts_local.T).T
                w, l, h = box.wlh
                vehicle_mask = (
                    (np.abs(pts_local[:, 0]) <= l / 2) &
                    (np.abs(pts_local[:, 1]) <= w / 2) &
                    (np.abs(pts_local[:, 2]) <= h / 2)
                )

            # Vehicle points — bright orange, larger
            if vehicle_mask.any():
                rr.log("lidar/vehicle", rr.Points3D(
                    pts[vehicle_mask],
                    colors=[[255, 140, 0]] * int(vehicle_mask.sum()),
                    radii=0.08,
                ))
            # Background points — dim grey, smaller
            rr.log("lidar/background", rr.Points3D(
                pts[~vehicle_mask],
                colors=[[80, 80, 80]] * int((~vehicle_mask).sum()),
                radii=0.02,
            ))

            # Ego vehicle — fixed box at LiDAR origin (approx car size)
            rr.log("ego", rr.Boxes3D(
                centers=[[0.0, 0.0, 0.0]],
                sizes=[[2.0, 4.5, 1.5]],
                labels=["ego"],
            ))
        else:
            selected = filter_boxes(boxes, args.mode, nusc=nusc, sample=sample, lidar_data=lidar_data)
            if args.instance:
                selected = [
                    b for b in selected
                    if nusc.get('sample_annotation', b.token)['instance_token'].endswith(args.instance)
                ]
            rr.log("lidar", rr.Points3D(pts))

        box_centers, box_sizes, box_quats, box_labels = [], [], [], []
        arrow_origins, arrow_vectors = [], []

        for box in selected:
            center = box.center.tolist()
            box_centers.append(center)
            box_sizes.append(box.wlh[[1, 0, 2]].tolist())
            q = box.orientation
            box_quats.append([q.x, q.y, q.z, q.w])

            # Persistent vehicle ID (constant across frames for the same object)
            ann = nusc.get('sample_annotation', box.token)
            vid = ann['instance_token'][-4:]  # last 4 chars as a short readable ID

            # Velocity
            vel = velocity_in_lidar_frame(nusc, box.token, lidar_data)
            if vel is not None:
                speed = float(np.linalg.norm(vel[:2]))  # horizontal speed
                box_labels.append(f"[{vid}] {box.name}  {speed:.1f} m/s")
                arrow_origins.append(center)
                arrow_vectors.append(vel.tolist())
            else:
                box_labels.append(f"[{vid}] {box.name}")

        if box_centers:
            rr.log("boxes", rr.Boxes3D(
                centers=box_centers,
                sizes=box_sizes,
                rotations=[rr.Quaternion(xyzw=q) for q in box_quats],
                labels=box_labels,
            ))

        if arrow_origins:
            rr.log("velocity", rr.Arrows3D(
                origins=arrow_origins,
                vectors=arrow_vectors,
            ))

        sample_token = sample['next']
        i += 1


if __name__ == '__main__':
    main()
