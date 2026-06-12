"""Lightweight training-time metrics computed during the validation loop.

These are fast approximations used for monitoring during training.
Full evaluation (degradation curves, ECE reliability diagrams) lives in src/eval/.
"""

import torch


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average Displacement Error: mean L2 distance over all timesteps and batch.

    Args:
        pred:   (B, T, 2) or (B, 2) predicted positions / velocities
        target: same shape as pred

    Returns:
        scalar ADE in metres (assuming inputs are in metres).
    """
    return (pred - target).norm(dim=-1).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Final Displacement Error: L2 distance at the last predicted timestep.

    Args:
        pred:   (B, T, 2) predicted trajectory
        target: (B, T, 2) ground-truth trajectory

    Returns:
        scalar FDE in metres.
    """
    return (pred[:, -1] - target[:, -1]).norm(dim=-1).mean()


def mean_nll(mu: torch.Tensor, log_sigma: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL for monitoring calibration quality during training."""
    mu        = mu.float()
    log_sigma = log_sigma.float().clamp(-1.0, 4.0)
    target    = target.float()
    var = torch.exp(2 * log_sigma)
    return (log_sigma + 0.5 * (target - mu) ** 2 / var).mean()
