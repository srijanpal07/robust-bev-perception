import torch
import torch.nn as nn


class _DSBlock(nn.Module):
    """Depthwise-separable conv block with BatchNorm and residual skip.

    Replaces a standard k×k Conv2d with:
      depthwise  — 3×3, groups=in_ch  (spatial mixing, one filter per channel)
      pointwise  — 1×1               (channel projection)
      BatchNorm  — after pointwise
      skip       — 1×1 conv if in_ch != out_ch, else identity

    Multiply-adds are ~8–9× fewer than an equivalent standard Conv2d(in_ch, out_ch, 3).
    bias=False on both convs because BatchNorm's learnable β makes the conv bias redundant.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch,  3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1,                           bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class BEVEncoder(nn.Module):
    """Lightweight BEV encoder: (in_ch, 500, 500) → (256,).

    Architecture (depthwise-separable + skip connections):
      stage1: DSBlock(in_ch→32) + MaxPool2 + Dropout2d  → (32, 250, 250)
      stage2: DSBlock(32→64)    + MaxPool2 + Dropout2d  → (64, 125, 125)
      stage3: DSBlock(64→128)   + MaxPool2 + Dropout2d  → (128,  62,  62)
      stage4: DSBlock(128→256)  + AdaptiveAvgPool(4,4)  → (256,   4,   4)
      fc:     4096 → 256
    """

    def __init__(self, dropout: float = 0.1, in_ch: int = 3):
        super().__init__()
        self.stage1 = nn.Sequential(
            _DSBlock(in_ch, 32),  nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.stage2 = nn.Sequential(
            _DSBlock(32,  64),  nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.stage3 = nn.Sequential(
            _DSBlock(64,  128), nn.MaxPool2d(2), nn.Dropout2d(dropout))
        self.stage4 = nn.Sequential(
            _DSBlock(128, 256), nn.AdaptiveAvgPool2d((4, 4)))
        self.fc = nn.Linear(256 * 4 * 4, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.fc(x.flatten(1))   # (B, 256)


class ResNet18BEVEncoder(nn.Module):
    """ResNet18 BEV encoder: (in_ch, 500, 500) → (256,).

    Uses torchvision ResNet18 (no pretrained weights — BEV is not natural images).
    Strips the original FC layer and replaces it with Linear(512, 256).
    When in_ch != 3 the stem conv is replaced to match the new channel count.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        import torchvision.models as tvm
        resnet = tvm.resnet18(weights=None)
        if in_ch != 3:
            resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x).flatten(1))   # (B, 256)


class CropEncoder(nn.Module):
    """Lightweight crop encoder: (in_ch, 64, 64) → (128,).

    Architecture:
      Conv(in_ch→32) + BN + ReLU + MaxPool2 + Dropout2d  → (32, 32, 32)
      Conv(32→64)    + BN + ReLU + MaxPool2 + Dropout2d  → (64, 16, 16)
      Conv(64→128)   + BN + ReLU + AdaptiveAvgPool(2,2)  → (128, 2,  2)
      fc: 512 → 128
    """

    def __init__(self, dropout: float = 0.1, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32,  3, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),     nn.Dropout2d(dropout),

            nn.Conv2d(32, 64,  3, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),     nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(128 * 2 * 2, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))   # (B, 128)


class ResNet18BEVEncoderWithFeatures(nn.Module):
    """ResNet18 BEV encoder that returns BOTH the feature map AND the pooled embedding.

    Phase 2 replacement for ResNet18BEVEncoder — required for RoI Align-based
    per-agent feature extraction.  The feature map (B, 512, H_feat, W_feat) is
    passed to RoIAgentEncoder; the pooled embedding (B, 256) can serve as optional
    global scene context.

    Spatial scale for a 500×500 input through ResNet18:
        500 → 250 (stem, stride 2) → 125 (layer1, no stride) → 63 (layer2, stride 2)
            → 32 (layer3, stride 2) → 16 (layer4, stride 2)
    spatial_scale = 16 / 500 = 0.032  (feature pixels per BEV pixel)
    """

    SPATIAL_SCALE = 16.0 / 500.0   # used by RoIAgentEncoder

    def __init__(self, in_ch: int = 22):
        super().__init__()
        import torchvision.models as tvm
        resnet = tvm.resnet18(weights=None)
        if in_ch != 3:
            resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)
        self.backbone  = nn.Sequential(*list(resnet.children())[:-2])   # → (B, 512, H, W)
        self.avgpool   = resnet.avgpool
        self.fc        = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.backbone(x)                      # (B, 512, H_feat, W_feat)
        pooled   = self.avgpool(feat_map).flatten(1)     # (B, 512)
        return self.fc(pooled), feat_map                 # (B, 256), (B, 512, H, W)


class RoIAgentEncoder(nn.Module):
    """Phase 2 replacement for CropEncoder.

    Extracts per-agent local features from the shared BEV feature map via
    torchvision RoI Align — one BEV encoder pass regardless of N agents.

    Why this replaces CropEncoder:
    - CropEncoder required N separate 64×64 forward passes (one per agent per scene).
    - RoIAgentEncoder samples from the feature map already computed by the BEV encoder.
    - O(1) BEV encoder cost regardless of N agents.
    - Per-agent spatial features are still preserved via the RoI window.
    - q_i (per-agent quality score) is computed from point density within the RoI region.

    Interface with ResNet18BEVEncoderWithFeatures:
        _, feat_map = bev_encoder(bev_stack)         # (B, 512, H_feat, W_feat)
        agent_feats = roi_encoder(feat_map, boxes)   # (B, N, out_dim)

    Agent boxes must be in BEV pixel coordinates [x1, y1, x2, y2], yaw-aligned
    by rotating the crop window before sampling (or via deformable sampling).
    torchvision.ops.roi_align handles fractional pixel coordinates and bilinear
    interpolation; spatial_scale converts BEV pixel coords to feature map coords.
    """

    def __init__(self, in_channels: int = 512, output_size: int = 7, out_dim: int = 256):
        super().__init__()
        self.output_size = output_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * output_size * output_size, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_map: torch.Tensor, agent_boxes: list,
                spatial_scale: float = ResNet18BEVEncoderWithFeatures.SPATIAL_SCALE,
                ) -> torch.Tensor:
        """
        feat_map:     (B, 512, H_feat, W_feat) from ResNet18BEVEncoderWithFeatures
        agent_boxes:  list of length B, each entry (N_i, 4) — [x1,y1,x2,y2]
                      in BEV pixel coordinates (yaw-aligned window recommended)
        spatial_scale: feat_map pixel / BEV pixel  (default: 16/500 = 0.032)
        returns:      (B, N_max, out_dim) — zero-padded to N_max agents

        Implementation note:
            torchvision.ops.roi_align expects boxes as (K, 5) with
            [batch_idx, x1, y1, x2, y2].  Pad to the maximum agent count
            across the batch and return a padded tensor.
        """
        from torchvision.ops import roi_align

        rois = []
        agent_counts = []
        for b_idx, boxes in enumerate(agent_boxes):
            if boxes.shape[0] == 0:
                agent_counts.append(0)
                continue
            batch_col = torch.full((boxes.shape[0], 1), b_idx,
                                   dtype=boxes.dtype, device=boxes.device)
            rois.append(torch.cat([batch_col, boxes], dim=1))   # (N_i, 5)
            agent_counts.append(boxes.shape[0])

        if not rois:
            B = feat_map.shape[0]
            return feat_map.new_zeros(B, 0, self.head[-2].out_features if hasattr(
                self.head[-2], 'out_features') else 256)

        all_rois  = torch.cat(rois, dim=0)                       # (sum_N, 5)
        pooled    = roi_align(feat_map, all_rois,
                              output_size=self.output_size,
                              spatial_scale=spatial_scale,
                              aligned=True)                      # (sum_N, 512, K, K)
        feats     = self.head(pooled)                            # (sum_N, out_dim)

        # Re-assemble into (B, N_max, out_dim) with zero padding
        N_max  = max(agent_counts) if agent_counts else 0
        B      = feat_map.shape[0]
        out    = feats.new_zeros(B, N_max, feats.shape[-1])
        ptr    = 0
        for b_idx, n in enumerate(agent_counts):
            if n > 0:
                out[b_idx, :n] = feats[ptr:ptr + n]
                ptr += n
        return out                                               # (B, N_max, out_dim)


class EfficientNetCropEncoder(nn.Module):
    """EfficientNet-B0 crop encoder: (in_ch, 64, 64) → (128,).

    Uses ImageNet-pretrained EfficientNet-B0. When in_ch != 3 the first conv
    is replaced (randomly initialised) to match the new channel count.
    Final classifier replaced with Linear(1280, 128).
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        import torchvision.models as tvm
        eff = tvm.efficientnet_b0(weights='IMAGENET1K_V1')
        if in_ch != 3:
            old = eff.features[0][0]
            eff.features[0][0] = nn.Conv2d(
                in_ch, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False,
            )
        self.features = eff.features
        self.avgpool  = eff.avgpool
        self.fc       = nn.Linear(1280, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.avgpool(self.features(x)).flatten(1)
        return self.fc(feat)   # (B, 128)
