import torch
import torch.nn as nn


class _DSBlock(nn.Module):
    """Depthwise-separable conv block with BatchNorm and residual skip (#13, #16).

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
            nn.Conv2d(in_ch, in_ch,  3, padding=1, groups=in_ch, bias=False),  # depthwise
            nn.Conv2d(in_ch, out_ch, 1,                           bias=False),  # pointwise
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class BEVEncoder(nn.Module):
    """Lightweight BEV encoder: (in_ch, 500, 500) → (256,).

    Architecture (#13 revised — depthwise-separable + skip connections):
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
        # Drop final FC; keep stem + 4 residual stages + avgpool → (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x).flatten(1))   # (B, 256)


class CropEncoder(nn.Module):
    """Lightweight crop encoder: (in_ch, 64, 64) → (128,).

    Architecture (#14 revised — original lightweight CNN + BN + Dropout):
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
            nn.MaxPool2d(2),     nn.Dropout2d(dropout),          # → 32×32

            nn.Conv2d(32, 64,  3, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),     nn.Dropout2d(dropout),          # → 16×16

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),                         # → 2×2
        )
        self.fc = nn.Linear(128 * 2 * 2, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))   # (B, 128)


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
        self.features = eff.features   # outputs (B, 1280, H', W')
        self.avgpool  = eff.avgpool    # AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(1280, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.avgpool(self.features(x)).flatten(1)   # (B, 1280)
        return self.fc(feat)   # (B, 128)


# ---------------------------------------------------------------------------
# Temporal aggregation: GRU or Transformer (#17, #18)
# ---------------------------------------------------------------------------

_COMBINED_DIM = 256 + 128 + 64   # bev_feat + crop_feat + box_feat = 448


class _LearnedPositionalEncoding(nn.Module):
    """Learnable positional embedding added to the sequence before the Transformer (#18).

    Uses nn.Embedding so positions are learned end-to-end rather than fixed sinusoids.
    max_len covers any T we'd realistically use (T * 2 with subframes ≤ 28 for T=14).
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        pos = torch.arange(x.shape[1], device=x.device)
        return x + self.pe(pos)   # broadcasts over batch


class TemporalVelocityPredictor(nn.Module):
    """Predicts vehicle velocity [vx, vy] from a temporal window of T BEV frames, crops, and box parameters.

    Per-frame features from a BEV encoder (→256), crop encoder (→128), and box MLP (→32) are
    concatenated into a (T, 416) sequence and processed by either:
      • a GRU  whose final hidden state  → head → [vx, vy]
      • a 2-layer Transformer encoder whose last-token output → projection → head → [vx, vy]

    Encoder variants (bev_encoder / crop_encoder):
      'lightweight' — depthwise-separable CNN / 3-layer CNN  (fast, fewer params)
      'resnet18'    — torchvision ResNet18 (no pretrained)
      'efficientnet'— torchvision EfficientNet-B0 (ImageNet pretrained, crop only)

    Temporal variants (temporal_model):
      'gru'         — GRU, hidden_size controls output dim
      'transformer' — TransformerEncoder with learnable positional encoding (#17/#18);
                      depth/width controlled by num_layers/nhead/dim_feedforward

    Outputs are in normalized label space — denormalize with label_mean/label_std from checkpoint.
    """

    def __init__(self, T: int = 3, box_dim: int = 7,
                 hidden_size: int = 256, dropout: float = 0.1,
                 bev_encoder: str = 'lightweight',
                 crop_encoder: str = 'lightweight',
                 bev_channels: int = 3,
                 crop_channels: int = 3,
                 temporal_model: str = 'gru',
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 512):
        super().__init__()
        self.T              = T
        self.bev_channels   = bev_channels
        self.crop_channels  = crop_channels
        self.temporal_model = temporal_model

        if bev_encoder == 'resnet18':
            self.bev_encoder = ResNet18BEVEncoder(in_ch=bev_channels)
        else:
            self.bev_encoder = BEVEncoder(dropout=dropout, in_ch=bev_channels)

        if crop_encoder == 'efficientnet':
            self.crop_encoder = EfficientNetCropEncoder(in_ch=crop_channels)
        else:
            self.crop_encoder = CropEncoder(dropout=dropout, in_ch=crop_channels)

        self.box_proj = nn.Linear(box_dim, 64)

        # Temporal aggregation — GRU or Transformer
        if temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=_COMBINED_DIM,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.pos_enc     = _LearnedPositionalEncoding(_COMBINED_DIM)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # Project last-token output to hidden_size so the head is identical for both paths
            self.temporal_proj = nn.Linear(_COMBINED_DIM, hidden_size)
        else:  # gru (default)
            self.gru = nn.GRU(
                input_size=_COMBINED_DIM,
                hidden_size=hidden_size,
                batch_first=True,
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),   # [vx, vy]
        )

        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, bev_stack: torch.Tensor,
                crop_stack: torch.Tensor,
                box_params: torch.Tensor) -> torch.Tensor:
        """
        bev_stack:  (B, T*bev_channels, 500, 500)
        crop_stack: (B, T*crop_channels, 64, 64)
        box_params: (B, T, box_dim)
        """
        B    = bev_stack.shape[0]
        T_kf = box_params.shape[1]   # keyframe count (= T, not T*2 when use_subframes)

        # BEV: self.T may equal T_kf*2 when use_subframes interleaves far+near BEVs.
        bev_feats = self.bev_encoder(
            bev_stack.view(B * self.T, self.bev_channels,
                           bev_stack.shape[2], bev_stack.shape[3])
        ).view(B, self.T, -1)                                         # (B, T_bev, 256)

        # Crops and box params are keyed to keyframes only.
        crop_feats = self.crop_encoder(
            crop_stack.view(B * T_kf, self.crop_channels, 64, 64)
        ).view(B, T_kf, -1)                                           # (B, T_kf, 128)
        box_feats  = self.box_proj(box_params)                        # (B, T_kf,  32)

        # When T_bev > T_kf (subframes), repeat-interleave crop/box so every BEV
        # step has a matching crop and box feature: [kf0,kf0,kf1,kf1,...].
        if self.T != T_kf:
            ratio      = self.T // T_kf
            crop_feats = crop_feats.repeat_interleave(ratio, dim=1)   # (B, T_bev, 128)
            box_feats  = box_feats.repeat_interleave(ratio, dim=1)    # (B, T_bev,  32)

        combined = torch.cat([bev_feats, crop_feats, box_feats], dim=-1)  # (B, T_bev, 416)

        if self.temporal_model == 'transformer':
            seq = self.pos_enc(combined)                   # (B, T, 416) + learned PE
            out = self.transformer(seq)                    # (B, T, 416)
            ctx = self.temporal_proj(out.mean(1))           # (B, hidden_size) — mean pool
        else:
            _, hidden = self.gru(combined)                 # (1, B, hidden_size)
            ctx = hidden.squeeze(0)                        # (B, hidden_size)

        return self.head(ctx)   # (B, 2)
