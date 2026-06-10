"""Full ADE / FDE computation for nuScenes motion forecasting benchmark.

Matches the nuScenes motion forecasting evaluation protocol:
  - minADE_k: minimum ADE over k trajectory modes
  - minFDE_k: minimum FDE over k trajectory modes
  - MissRate_k: fraction of samples where all k modes miss by > 2 m at T_future

Horizons evaluated: 1 s (2 steps), 2 s (4 steps), 3 s (6 steps) at 0.5 s / step.
"""

import numpy as np
import torch


def ade_k(pred: torch.Tensor, target: torch.Tensor, k: int = 1) -> torch.Tensor:
    """minADE_k: minimum ADE over k predicted modes.

    Args:
        pred:   (B, k, T, 2) k trajectory modes
        target: (B, T, 2)    ground-truth trajectory

    Returns:
        scalar minADE_k
    """
    target_exp = target.unsqueeze(1).expand_as(pred)       # (B, k, T, 2)
    per_mode   = (pred - target_exp).norm(dim=-1).mean(-1)  # (B, k)
    return per_mode.min(dim=1).values.mean()                # scalar


def fde_k(pred: torch.Tensor, target: torch.Tensor, k: int = 1) -> torch.Tensor:
    """minFDE_k: minimum FDE over k predicted modes (at final timestep only)."""
    target_final = target[:, -1].unsqueeze(1).expand(target.shape[0], k, 2)
    per_mode     = (pred[:, :, -1] - target_final).norm(dim=-1)   # (B, k)
    return per_mode.min(dim=1).values.mean()


def miss_rate_k(pred: torch.Tensor, target: torch.Tensor,
                k: int = 1, threshold: float = 2.0) -> torch.Tensor:
    """MissRate_k: fraction where best mode FDE > threshold metres."""
    target_final = target[:, -1].unsqueeze(1).expand(target.shape[0], k, 2)
    per_mode     = (pred[:, :, -1] - target_final).norm(dim=-1)
    best_fde     = per_mode.min(dim=1).values
    return (best_fde > threshold).float().mean()


def evaluate_forecasting(model: torch.nn.Module, dataloader,
                          T_future: int = 6, k: int = 1,
                          device: str = 'cuda') -> dict:
    """Run full forecasting evaluation and return metric dict.

    Returns:
        {'minADE_1': ..., 'minFDE_1': ..., 'MissRate_1': ...,
         'minADE_1s': ..., 'minADE_2s': ..., 'minADE_3s': ...}
    """
    raise NotImplementedError
