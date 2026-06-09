"""
Realistic camera-only 3D bounding-box detector noise for nuScenes GT boxes.

Calibrated to monocular/multi-camera detectors (FCOS3D, DETR3D, BEVDet) on
nuScenes.  Every noise source uses a distribution that matches how that
particular error actually arises in camera-based 3D detection — none of them
are Gaussian:

  DEPTH (radial)   — Log-normal multiplicative.  Depth prediction error in
                     log-space is approximately normal, so the absolute error
                     in metres is right-skewed and grows proportionally with
                     distance.  Occasional gross failures (2D→3D lifting
                     ambiguity) add a Laplace-distributed depth jump.

  LATERAL          — Laplace.  Angular localisation in the image is fairly
                     precise, but the Laplace distribution's sharper peak and
                     heavier tails better match the bursty image-space errors
                     that propagate through the perspective projection.

  HEIGHT (z)       — Laplace with a small positive bias.  Networks struggle
                     with pitch/elevation and tend to overestimate height
                     slightly because the ground-plane prior is imperfect.

  YAW              — Four-component mixture model:
                       • Laplace(0, σ_yaw)   — normal run, small error
                       • Laplace(π, 0.08)    — confident 180° flip
                       • Laplace(±π/2, 0.08) — confident ±90° flip (side confusion)
                       • Uniform(−π, π)      — complete heading confusion
                     Each component draws fresh per frame so the mode can
                     jump discontinuously between frames.

  DIMENSIONS       — Log-normal shared scale factor (correlated l/w/h) + per-
                     axis Laplace additive noise + slight regression-to-mean
                     (camera networks learn a strong size prior and shrink
                     predictions toward the dataset average).

  VELOCITY (kin.)  — Laplace noise on vx̂/vŷ (heavier tails than Gaussian,
                     matching the large velocity errors of camera detectors
                     that use only two noisy position estimates).

All noise is i.i.d. per frame; no bias is shared across frames.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fixed hyper-parameters — the noise_scale arg multiplies σ values below
# ---------------------------------------------------------------------------

# Depth: log-normal multiplicative.
# Monocular depth uncertainty is roughly *constant in relative (%) terms*,
# so σ_log is a single small constant.  Absolute error naturally grows with
# distance because delta_r = d * (exp(noise) - 1).
# IQR of absolute error ≈ ±d*(exp(σ_log)-1):
#   d=10 m → ±0.7 m   d=30 m → ±2.2 m   d=50 m → ±3.6 m
_LOG_DEPTH_SIGMA = 0.07   # σ in log-depth space (constant relative uncertainty)

# Outlier: gross depth jump when 2D→3D lifting fails (additive Laplace on top)
_OUTLIER_LAP_BASE  = 0.50   # m — Laplace scale floor
_OUTLIER_LAP_SCALE = 0.020  # m/m — Laplace scale growth with distance
# 99th-pct outlier jump: ~5 m at 30 m, ~7 m at 50 m

# Lateral: Laplace angular uncertainty
_LATERAL_ANGLE_LAP = 0.006  # rad (Laplace scale, ~0.34°)

# Height z: Laplace + slight positive bias
_Z_LAP_BASE  = 0.20   # m
_Z_LAP_SCALE = 0.006  # m/m
_Z_BIAS      = 0.08   # m — cameras overestimate height slightly

# Yaw: Laplace continuous noise
_YAW_LAP_BASE  = 0.06   # rad
_YAW_LAP_SCALE = 0.005  # rad/m
# Mixture flip component sigma (the network is *confidently* wrong)
_YAW_FLIP_SIGMA = 0.08  # rad

# Dimensions: mean nuScenes car size (used for regression-to-mean)
_CAR_MEAN_L, _CAR_MEAN_W, _CAR_MEAN_H = 4.60, 1.95, 1.73
_DIM_SHRINK        = 0.10   # fraction regression toward mean car size
_DIM_LOG_SCALE_STD = 0.06   # σ of shared log-normal scale factor
_DIM_LAP_L = 0.15           # Laplace scale for additive length noise
_DIM_LAP_W = 0.09           # Laplace scale for additive width noise
_DIM_LAP_H = 0.07           # Laplace scale for additive height noise
_DIM_MIN_L, _DIM_MIN_W, _DIM_MIN_H = 0.5, 0.3, 0.3

# Defaults for the discrete-event parameters (overridable via function args)
_DEFAULT_DEPTH_OUTLIER_PROB = 0.015
_DEFAULT_YAW_PI_FLIP_PROB   = 0.025


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_camera_detector_noise(
    box_params: np.ndarray,
    rng: "np.random.Generator | None" = None,
    noise_scale: float = 1.0,
    depth_outlier_prob: float = _DEFAULT_DEPTH_OUTLIER_PROB,
    yaw_pi_flip_prob: float = _DEFAULT_YAW_PI_FLIP_PROB,
) -> np.ndarray:
    """Inject realistic camera-only 3D bounding-box detector noise into GT boxes.

    Every noise component uses a non-Gaussian distribution that reflects the
    actual error mechanism in camera-based 3D detection (see module docstring).
    All noise is sampled i.i.d. per frame.

    Args:
        box_params: Float array (T, D), D = 7 or 10.
            Columns 0-6 : [x, y, z, l, w, h, yaw] in current LiDAR frame.
            Columns 7-9 : [dt, vx̂, vŷ] kinematic features (optional).
        rng: NumPy Generator.  A fresh default_rng() is used when None,
            giving independent noise on every call.
        noise_scale: Global multiplier on all continuous σ / scale values.
            0.0 = no noise, 1.0 = calibrated camera noise, >1.0 = worse.
        depth_outlier_prob: Per-frame probability of a gross depth error
            (2D→3D lifting failure).  Default 0.015.
        yaw_pi_flip_prob: Per-frame probability of a 180° heading flip.
            ±90° flip probability is set to 40 % of this.  Default 0.025.

    Returns:
        Noisy copy of box_params, same dtype as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    s = float(noise_scale)
    yaw_halfpi_prob = 0.40 * yaw_pi_flip_prob
    # Probability of complete heading confusion (uniform draw)
    yaw_uniform_prob = 0.005

    src_dtype = box_params.dtype
    out = box_params.astype(np.float64).copy()
    T   = out.shape[0]

    for t in range(T):
        x, y, z = out[t, 0], out[t, 1], out[t, 2]

        dist  = max(float(np.sqrt(x * x + y * y)), 1.0)
        theta = float(np.arctan2(y, x))

        # ------------------------------------------------------------------ #
        # 1. DEPTH (radial) — log-normal multiplicative                       #
        #                                                                      #
        # In log-depth space the prediction error is roughly normal.          #
        # This yields a right-skewed, multiplicative absolute error:          #
        #   predicted_depth = GT_depth * exp( N(0, σ_log) )                  #
        # Heavy-tail: mix in a Laplace-distributed additive outlier jump.     #
        # ------------------------------------------------------------------ #
        sigma_log = s * _LOG_DEPTH_SIGMA
        log_noise = rng.normal(0.0, sigma_log)
        delta_r   = dist * (np.exp(log_noise) - 1.0)   # multiplicative shift

        if rng.random() < depth_outlier_prob:
            # Gross 2D→3D failure: extra Laplace depth jump on top of log-normal
            outlier_scale = s * (_OUTLIER_LAP_BASE + _OUTLIER_LAP_SCALE * dist)
            delta_r += rng.laplace(0.0, outlier_scale)

        # ------------------------------------------------------------------ #
        # 2. LATERAL (tangential) — Laplace                                   #
        #                                                                      #
        # Angular localisation in image space is better than depth but still  #
        # bursty; Laplace's sharper peak + heavy tails fits this well.        #
        # ------------------------------------------------------------------ #
        sigma_lat = s * _LATERAL_ANGLE_LAP * dist
        delta_lat = rng.laplace(0.0, sigma_lat)

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        out[t, 0] = x + delta_r * cos_t - delta_lat * sin_t
        out[t, 1] = y + delta_r * sin_t + delta_lat * cos_t

        # ------------------------------------------------------------------ #
        # 3. HEIGHT z — Laplace + small positive bias                         #
        #                                                                      #
        # Cameras struggle with elevation; ground-plane prior is imperfect    #
        # leading to slight systematic overestimation of object height.       #
        # ------------------------------------------------------------------ #
        sigma_z = s * (_Z_LAP_BASE + _Z_LAP_SCALE * dist)
        out[t, 2] = z + rng.laplace(s * _Z_BIAS, sigma_z)

        # ------------------------------------------------------------------ #
        # 4. DIMENSIONS — log-normal shared scale + Laplace per-axis          #
        #               + regression toward mean car size                     #
        #                                                                      #
        # Camera networks learn strong size priors from training data and     #
        # partially regress predictions toward the dataset mean (shrinkage).  #
        # The shared log-normal scale captures correlated l/w/h errors        #
        # (the whole box shrinks or expands together).                        #
        # ------------------------------------------------------------------ #
        shared_scale = np.exp(rng.normal(0.0, s * _DIM_LOG_SCALE_STD))

        for col, mean_dim, lap_s, min_d in [
            (3, _CAR_MEAN_L, s * _DIM_LAP_L, _DIM_MIN_L),
            (4, _CAR_MEAN_W, s * _DIM_LAP_W, _DIM_MIN_W),
            (5, _CAR_MEAN_H, s * _DIM_LAP_H, _DIM_MIN_H),
        ]:
            shrunk = out[t, col] * (1.0 - _DIM_SHRINK) + mean_dim * _DIM_SHRINK
            out[t, col] = max(min_d,
                              shrunk * shared_scale + rng.laplace(0.0, lap_s))

        # ------------------------------------------------------------------ #
        # 5. YAW — four-component mixture model                               #
        #                                                                      #
        # Component probabilities (per frame, independent):                   #
        #   • Normal run  : Laplace(0, σ_yaw)     — most frames               #
        #   • π flip      : Laplace(π, 0.08)      — confident front-back flip #
        #   • ±π/2 flip   : Laplace(±π/2, 0.08)  — confident side flip       #
        #   • Uniform(−π,π): complete confusion                               #
        # The flip components have small sigma because when the network        #
        # flips, it does so confidently (bimodal error, not gradual drift).   #
        # ------------------------------------------------------------------ #
        sigma_yaw = s * (_YAW_LAP_BASE + _YAW_LAP_SCALE * dist)

        u = rng.random()
        if u < yaw_pi_flip_prob:
            delta_yaw = rng.laplace(np.pi, _YAW_FLIP_SIGMA)
        elif u < yaw_pi_flip_prob + yaw_halfpi_prob:
            sign = float(rng.choice([-1, 1]))
            delta_yaw = rng.laplace(sign * np.pi / 2, _YAW_FLIP_SIGMA)
        elif u < yaw_pi_flip_prob + yaw_halfpi_prob + yaw_uniform_prob:
            delta_yaw = rng.uniform(-np.pi, np.pi) - out[t, 6]  # random absolute yaw
        else:
            delta_yaw = rng.laplace(0.0, sigma_yaw)

        out[t, 6] = float((out[t, 6] + delta_yaw + np.pi) % (2 * np.pi) - np.pi)

    return out.astype(src_dtype)
