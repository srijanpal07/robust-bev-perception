"""Calibration monitoring during the validation loop.

Computes a lightweight Expected Calibration Error (ECE) approximation during
training to track whether predicted uncertainty is well-calibrated across
degradation levels. Full reliability diagrams are in src/eval/calibration_metrics.py.
"""

import torch
import numpy as np


def interval_coverage(mu: torch.Tensor, log_sigma: torch.Tensor,
                       target: torch.Tensor, alpha: float = 0.9) -> float:
    """Fraction of targets falling within the predicted (1-alpha) credible interval.

    A well-calibrated model should report coverage ≈ alpha.
    Under-confident: coverage > alpha (intervals too wide).
    Over-confident: coverage < alpha (intervals too narrow — dangerous under degradation).

    Args:
        mu:         (B, T, 2) or (B, 2) predicted mean
        log_sigma:  same shape — predicted log std
        target:     same shape — ground truth
        alpha:      target coverage level (default 0.9 → 90% interval)

    Returns:
        empirical coverage fraction in [0, 1]
    """
    sigma = torch.exp(log_sigma)
    z = 1.645  # 90% normal quantile
    in_interval = ((target - mu).abs() <= z * sigma).float()
    return float(in_interval.mean())


def ece_1d(mu: torch.Tensor, log_sigma: torch.Tensor,
           target: torch.Tensor, n_bins: int = 10) -> float:
    """1D ECE approximation via confidence-accuracy binning.

    TODO: replace with proper multivariate ECE for (x, y) trajectory outputs.
    """
    raise NotImplementedError
