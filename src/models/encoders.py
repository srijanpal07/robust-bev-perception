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
