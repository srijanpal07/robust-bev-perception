"""Phase 1 single-agent trajectory model.

Wires the existing components for Phase 1 (vehicle.car, no camera):

    BEV (22,500,500) ──► ResNet18BEVEncoderWithFeatures ──► pooled (256)
    crop (22,64,64)  ──► CropEncoder                    ──► crop_feat (128)
    box_params (7)   ──► Linear(7→32) + ReLU            ──► box_feat (32)
                                    │ cat (416)
                                    ▼
                          Linear(416→256) + ReLU + Dropout ──► lidar_feat (256)
                                    │
                          q_score gating: fused = q_i · lidar_feat
                          (no camera features in Phase 1; when q→0 the head
                          receives near-zero input and must learn to output
                          high σ to stay calibrated)
                                    │
                          TrajectoryHead(256) ──► μ (B,6,2), log σ (B,6,2)

q_score is the geometric quality score from local LiDAR density (computed by
quality_score.py and injected by the DataLoader).  No separate learned gate is
needed in Phase 1 — the geometric score IS the ground-truth beam quality signal
and is directly interpretable.  Phase 2 will replace this with MultiAgentModalityGating.
"""

import torch
import torch.nn as nn

from src.models.encoders import ResNet18BEVEncoderWithFeatures, CropEncoder
from src.models.heads import TrajectoryHead


class Phase1Model(nn.Module):
    def __init__(
        self,
        bev_channels:  int   = 22,
        box_dim:       int   = 7,
        box_proj_dim:  int   = 32,
        lidar_feat_dim: int  = 256,
        t_future:      int   = 6,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.bev_encoder  = ResNet18BEVEncoderWithFeatures(in_ch=bev_channels)
        self.crop_encoder = CropEncoder(in_ch=bev_channels, dropout=dropout)
        self.box_proj     = nn.Sequential(
            nn.Linear(box_dim, box_proj_dim),
            nn.ReLU(inplace=True),
        )
        combined_dim = 256 + 128 + box_proj_dim   # 416
        self.feat_proj = nn.Sequential(
            nn.Linear(combined_dim, lidar_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.traj_head = TrajectoryHead(hidden_size=lidar_feat_dim, T_future=t_future)

    def forward(
        self,
        bev:        torch.Tensor,   # (B, C, 500, 500)
        crop:       torch.Tensor,   # (B, C,  64,  64)
        box_params: torch.Tensor,   # (B, 7)
        q_score:    torch.Tensor,   # (B, 1)  — geometric LiDAR quality in [0,1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns mu (B, T, 2) and log_sigma (B, T, 2)."""
        bev_pooled, _ = self.bev_encoder(bev)    # (B, 256)
        crop_feat     = self.crop_encoder(crop)  # (B, 128)
        box_feat      = self.box_proj(box_params)  # (B, 32)

        lidar_feat = self.feat_proj(
            torch.cat([bev_pooled, crop_feat, box_feat], dim=-1)
        )  # (B, 256)

        # Direct q_i gating — no camera in Phase 1
        fused = q_score * lidar_feat   # (B, 256)

        return self.traj_head(fused)   # (B, T, 2), (B, T, 2)
