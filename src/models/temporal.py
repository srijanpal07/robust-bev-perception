import torch
import torch.nn as nn

from src.models.encoders import (
    BEVEncoder, ResNet18BEVEncoder,
    CropEncoder, EfficientNetCropEncoder,
)
from src.models.heads import VelocityHead

_COMBINED_DIM = 256 + 128 + 64   # bev_feat + crop_feat + box_feat = 448


class _LearnedPositionalEncoding(nn.Module):
    """Learnable positional embedding added to the sequence before the Transformer.

    Uses nn.Embedding so positions are learned end-to-end rather than fixed sinusoids.
    max_len covers any T we'd realistically use.
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        pos = torch.arange(x.shape[1], device=x.device)
        return x + self.pe(pos)


class TemporalVelocityPredictor(nn.Module):
    """Predicts vehicle velocity [vx, vy] from a temporal window of T BEV frames, crops, and box parameters.

    Per-frame features from a BEV encoder (→256), crop encoder (→128), and box MLP (→32) are
    concatenated into a (T, 448) sequence and processed by either:
      • a GRU  whose final hidden state  → VelocityHead → [vx, vy]
      • a 2-layer Transformer encoder whose mean-pooled output → VelocityHead → [vx, vy]

    Encoder variants (bev_encoder / crop_encoder):
      'lightweight' — depthwise-separable CNN / 3-layer CNN
      'resnet18'    — torchvision ResNet18 (no pretrained)
      'efficientnet'— torchvision EfficientNet-B0 (ImageNet pretrained, crop only)

    Temporal variants (temporal_model):
      'gru'         — GRU, hidden_size controls output dim
      'transformer' — TransformerEncoder with learnable positional encoding

    Outputs are in normalised label space — denormalise with label_mean/label_std from checkpoint.

    Note: T_MODEL (BEV steps) vs T_kf (keyframe count) distinction matters when use_subframes=True.
    A mismatch caused an inference crash in the baseline — see dataset.py for the interleaving logic.
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

        if temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=_COMBINED_DIM,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.pos_enc       = _LearnedPositionalEncoding(_COMBINED_DIM)
            self.transformer   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.temporal_proj = nn.Linear(_COMBINED_DIM, hidden_size)
        else:
            self.gru = nn.GRU(
                input_size=_COMBINED_DIM,
                hidden_size=hidden_size,
                batch_first=True,
            )

        self.head = VelocityHead(hidden_size)

    def forward(self, bev_stack: torch.Tensor,
                crop_stack: torch.Tensor,
                box_params: torch.Tensor) -> torch.Tensor:
        """
        bev_stack:  (B, T*bev_channels, 500, 500)
        crop_stack: (B, T*crop_channels, 64, 64)
        box_params: (B, T, box_dim)
        Returns:    (B, 2) — [vx, vy]
        """
        B    = bev_stack.shape[0]
        T_kf = box_params.shape[1]

        bev_feats = self.bev_encoder(
            bev_stack.view(B * self.T, self.bev_channels,
                           bev_stack.shape[2], bev_stack.shape[3])
        ).view(B, self.T, -1)

        crop_feats = self.crop_encoder(
            crop_stack.view(B * T_kf, self.crop_channels, 64, 64)
        ).view(B, T_kf, -1)
        box_feats  = self.box_proj(box_params)

        if self.T != T_kf:
            ratio      = self.T // T_kf
            crop_feats = crop_feats.repeat_interleave(ratio, dim=1)
            box_feats  = box_feats.repeat_interleave(ratio, dim=1)

        combined = torch.cat([bev_feats, crop_feats, box_feats], dim=-1)

        if self.temporal_model == 'transformer':
            seq = self.pos_enc(combined)
            out = self.transformer(seq)
            ctx = self.temporal_proj(out.mean(1))
        else:
            _, hidden = self.gru(combined)
            ctx = hidden.squeeze(0)

        return self.head(ctx)   # (B, 2)
