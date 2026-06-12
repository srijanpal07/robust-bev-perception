"""Training losses for velocity prediction and probabilistic trajectory forecasting."""

import torch
import torch.nn.functional as F


def velocity_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss for point-estimate velocity prediction (baseline)."""
    return F.mse_loss(pred, target)


def trajectory_nll_loss(mu: torch.Tensor, log_sigma: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL loss for probabilistic trajectory prediction (C4).

    Args:
        mu:         (B, T_future, 2) predicted mean waypoints
        log_sigma:  (B, T_future, 2) predicted log standard deviation
        target:     (B, T_future, 2) ground-truth waypoints

    Returns:
        scalar NLL loss, averaged over batch and time.
    """
    # Cast to float32: AMP returns fp16 outside autocast; exp(2*log_sigma) can
    # underflow in fp16.  Clamp matches TrajectoryHead floor: sigma in [0.37, 55] m.
    mu        = mu.float()
    log_sigma = log_sigma.float().clamp(-1.0, 4.0)
    target    = target.float()
    var = torch.exp(2 * log_sigma)   # in [0.135, 2981] — stable in fp32, no extra clamp
    nll = log_sigma + 0.5 * (target - mu) ** 2 / var
    return nll.mean()


def combined_trajectory_loss(mu: torch.Tensor, log_sigma: torch.Tensor,
                              target: torch.Tensor,
                              nll_weight: float = 1.0,
                              ade_weight: float = 0.5) -> torch.Tensor:
    """NLL + ADE composite loss for trajectory training.

    NLL encourages calibrated uncertainty; ADE regularises against degenerate
    distributions with very wide sigma and poor mean accuracy.
    """
    nll = trajectory_nll_loss(mu, log_sigma, target)
    ade = (mu - target).norm(dim=-1).mean()
    return nll_weight * nll + ade_weight * ade
