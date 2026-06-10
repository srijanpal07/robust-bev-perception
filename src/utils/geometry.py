"""Geometric utilities: coordinate transforms, BEV projection, ego-motion compensation."""

import numpy as np


def ego_to_bev_px(x: float, y: float, bev_size: int = 500,
                   resolution: float = 0.2) -> tuple[int, int]:
    """Convert ego-frame (x, y) metres to BEV pixel (row, col).

    nuScenes BEV convention: x forward, y left, origin at ego centre.
    BEV image convention: row 0 = top (forward), col 0 = left.
    """
    col = int(round(bev_size / 2 - y / resolution))
    row = int(round(bev_size / 2 - x / resolution))
    return row, col


def rotation_matrix_2d(yaw_rad: float) -> np.ndarray:
    """2×2 rotation matrix for yaw angle (counter-clockwise positive)."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s], [s, c]])


def compensate_ego_motion(points: np.ndarray, ego_delta: np.ndarray) -> np.ndarray:
    """Subtract ego-vehicle translation from point cloud positions.

    Args:
        points:    (N, 3+) point cloud in world frame
        ego_delta: (3,) translation [dx, dy, dz] of ego vehicle

    Returns:
        (N, 3+) points in ego-compensated frame
    """
    out = points.copy()
    out[:, :3] -= ego_delta
    return out


def yaw_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    """Extract yaw (rotation about z-axis) from a unit quaternion."""
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
