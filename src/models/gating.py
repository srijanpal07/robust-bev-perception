import torch
import torch.nn as nn


class MultiAgentModalityGating(nn.Module):
    """Phase 2: per-agent modality gating for (B, N, feat_dim) batches.

    Same gating logic as ModalityGating but processes N agents simultaneously.
    Input features are (B, N, feat_dim); q_scores are (B, N, 1).

    Used with ResNet18BEVEncoderWithFeatures + RoIAgentEncoder.
    """

    def __init__(self, lidar_feat_dim: int = 256, camera_feat_dim: int = 256,
                 stats_dim: int = 16):
        super().__init__()
        self.quality_estimator = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, lidar_feat: torch.Tensor, camera_feat: torch.Tensor,
                pc_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lidar_feat:  (B, N, lidar_feat_dim)
            camera_feat: (B, N, camera_feat_dim)
            pc_stats:    (B, N, stats_dim) — per-agent point cloud statistics

        Returns:
            fused: (B, N, lidar_feat_dim)
            q:     (B, N, 1) — per-agent quality score in [0, 1]
        """
        q = self.quality_estimator(pc_stats)          # (B, N, 1)
        fused = q * lidar_feat + (1.0 - q) * camera_feat
        return fused, q


class ModalityGating(nn.Module):
    """Soft per-modality quality gating from point cloud statistics.

    TODO (C3): estimates a quality score q ∈ [0, 1] for each active modality
    (LiDAR, camera, radar) from lightweight per-frame statistics, then produces
    a soft weighted combination of modality BEV features:

        fused = q_lidar * f_lidar + (1 - q_lidar) * f_camera

    The score is derived from point cloud density statistics (beam count proxy,
    intensity variance, range histogram) rather than hand-engineered thresholds,
    so the gating adapts to partial degradation not just binary failure.

    Calibrated uncertainty: q is also passed to the trajectory head so that
    predicted uncertainty can widen as q → 0 (LiDAR degraded → higher positional
    uncertainty → wider Gaussian over future waypoints).

    Ablation (exploratory): compare this geometric-statistics gating against a
    CLIP ViT-L/14 gating signal derived from front-camera frames. Only promote
    CLIP gating to a contribution if it clearly outperforms this baseline.
    """

    def __init__(self, lidar_feat_dim: int = 256, camera_feat_dim: int = 256,
                 stats_dim: int = 16):
        super().__init__()
        self.quality_estimator = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, lidar_feat: torch.Tensor, camera_feat: torch.Tensor,
                pc_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lidar_feat:  (B, lidar_feat_dim)
            camera_feat: (B, camera_feat_dim)
            pc_stats:    (B, stats_dim) — point cloud statistics (density, range hist, etc.)

        Returns:
            fused: (B, lidar_feat_dim)  — soft-gated feature blend
            q:     (B, 1)               — LiDAR quality score in [0, 1]
        """
        q = self.quality_estimator(pc_stats)          # (B, 1)
        fused = q * lidar_feat + (1.0 - q) * camera_feat
        return fused, q
