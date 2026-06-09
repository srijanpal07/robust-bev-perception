"""
Constant-velocity Kalman smoother for vehicle velocity estimation.

State:       x = [px, py, vx, vy]
Motion model: constant velocity  (px += vx*dt, py += vy*dt)
Measurement:  noisy [px, py] positions in the ego-motion-compensated
              reference LiDAR frame.

Measurement noise R is built per-step from the object's current range,
matching the anisotropic depth/lateral profile used in box_noise.py:
  - Radial (depth) variance  ≈ (d · noise_scale · depth_sigma)²
    log-normal approximation: Var ≈ (d · σ_log)² for small σ_log
  - Lateral variance         = 2 · (noise_scale · lateral_lap · d)²
    Laplace(0, b):  Var = 2b²

Both variances are rotated from polar into Cartesian (x, y) by the
azimuth angle θ = atan2(py, px).

A Rauch-Tung-Striebel (RTS) backward pass follows the forward filter so
that every frame receives an optimal velocity estimate that incorporates
both past and future position observations.  Without the backward pass,
frames 0 and 1 carry identical or near-identical velocities when the
forward FD is used to seed the initial state (zero innovation at t=1).
"""

import numpy as np

# ── defaults match box_noise.py constants ────────────────────────────────────
_DEPTH_SIGMA   = 0.07    # σ in log-depth space  (_LOG_DEPTH_SIGMA)
_LATERAL_LAP   = 0.006   # Laplace scale per metre (_LATERAL_ANGLE_LAP)
_SIGMA_A       = 0.5     # m/s² — acceleration noise for process model Q
# ─────────────────────────────────────────────────────────────────────────────

_H  = np.array([[1., 0., 0., 0.],
                [0., 1., 0., 0.]])    # position-only observation matrix
_I4 = np.eye(4)


def _transition(dt: float) -> np.ndarray:
    return np.array([[1., 0., dt, 0.],
                     [0., 1., 0., dt],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])


def _process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """Discrete white-noise acceleration model for Q."""
    q = sigma_a ** 2
    d2, d3, d4 = dt**2, dt**3, dt**4
    return q * np.array([[d4/4, 0.,   d3/2, 0.  ],
                          [0.,   d4/4, 0.,   d3/2],
                          [d3/2, 0.,   d2,   0.  ],
                          [0.,   d3/2, 0.,   d2  ]])


def _meas_noise(px: float, py: float,
                noise_scale: float,
                depth_sigma: float,
                lateral_lap: float) -> np.ndarray:
    """
    Anisotropic 2×2 measurement noise covariance in Cartesian (x, y).

    Built in polar coordinates then rotated:
      var_r   = (d · noise_scale · depth_sigma)²   — radial (depth)
      var_lat = 2 · (noise_scale · lateral_lap · d)²  — lateral (Laplace)
    """
    d     = max(float(np.sqrt(px*px + py*py)), 1.0)
    theta = float(np.arctan2(py, px))
    c, s  = np.cos(theta), np.sin(theta)

    var_r   = (d * noise_scale * depth_sigma) ** 2
    var_lat = 2.0 * (noise_scale * lateral_lap * d) ** 2

    return np.array([
        [c*c*var_r + s*s*var_lat,  c*s*(var_r - var_lat)],
        [c*s*(var_r - var_lat),    s*s*var_r + c*c*var_lat],
    ])


def kalman_velocity(
    positions:   np.ndarray,
    timestamps:  np.ndarray,
    noise_scale: float = 1.0,
    sigma_a:     float = _SIGMA_A,
    depth_sigma: float = _DEPTH_SIGMA,
    lateral_lap: float = _LATERAL_LAP,
) -> np.ndarray:
    """
    Estimate velocity at every keyframe using a forward KF + RTS backward smoother.

    Forward pass:
      t=0  — initialise position from first measurement; velocity prior = (0, 0)
             with large uncertainty (10000 m²/s²) so the filter converges quickly.
      t≥1  — predict with constant-velocity model, update with noisy position.

    Backward pass (RTS):
      t=T-2..0 — smoother gain G propagates the future-informed state backward,
                 giving each frame an estimate that uses all T observations.
                 This ensures frames 0 and 1 carry distinct, meaningful velocities
                 rather than both collapsing to the same forward FD.

    Args:
        positions:   (T, 2) float array — noisy [px, py] in the reference
                     LiDAR frame (ego-motion compensated).
        timestamps:  (T,)  integer array — UNIX-microsecond timestamps.
        noise_scale: multiplier on R  (should equal box_noise_params.noise_scale;
                     use a small value like 0.05 for near-GT positions).
        sigma_a:     acceleration noise magnitude (m/s²) for Q.
        depth_sigma: log-depth σ (default matches box_noise._LOG_DEPTH_SIGMA).
        lateral_lap: lateral Laplace scale coefficient (matches _LATERAL_ANGLE_LAP).

    Returns:
        velocities: (T, 2) float array — RTS-smoothed [vx, vy] at each step.
    """
    T = positions.shape[0]
    velocities = np.zeros((T, 2), dtype=np.float64)

    # ── Initialise at t=0: zero velocity prior, large uncertainty ────────────
    p0 = positions[0].astype(np.float64)
    R0 = _meas_noise(p0[0], p0[1], noise_scale, depth_sigma, lateral_lap)
    x  = np.array([p0[0], p0[1], 0.0, 0.0])
    P  = np.diag([R0[0, 0], R0[1, 1], 100.0, 100.0])  # very loose velocity prior → fast convergence

    # Storage for RTS backward pass
    xs      = [x.copy()]   # filtered states
    Ps      = [P.copy()]   # filtered covariances
    As      = []           # transition matrices  (len = T-1)
    P_preds = []           # predicted covariances before update (len = T-1)

    # ── Forward filter ───────────────────────────────────────────────────────
    for t in range(1, T):
        dt = max((int(timestamps[t]) - int(timestamps[t - 1])) / 1e6, 0.05)

        A = _transition(dt)
        Q = _process_noise(dt, sigma_a)
        x_pred = A @ xs[-1]
        P_pred = A @ Ps[-1] @ A.T + Q

        meas  = positions[t].astype(np.float64)
        R     = _meas_noise(meas[0], meas[1], noise_scale, depth_sigma, lateral_lap)
        innov = meas - _H @ x_pred
        S     = _H @ P_pred @ _H.T + R
        K     = P_pred @ _H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ innov
        P_upd = (_I4 - K @ _H) @ P_pred

        xs.append(x_upd.copy())
        Ps.append(P_upd.copy())
        As.append(A)
        P_preds.append(P_pred.copy())

        velocities[t] = x_upd[2:4]

    # ── RTS backward smoother ────────────────────────────────────────────────
    x_smooth = xs[-1].copy()
    for t in range(T - 2, -1, -1):
        G        = Ps[t] @ As[t].T @ np.linalg.inv(P_preds[t])
        x_smooth = xs[t] + G @ (x_smooth - As[t] @ xs[t])
        velocities[t] = x_smooth[2:4]

    return velocities
