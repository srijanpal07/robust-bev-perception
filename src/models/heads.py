import torch
import torch.nn as nn


class VelocityHead(nn.Module):
    """2-layer MLP: context vector → [vx, vy].

    Predicts instantaneous velocity as a point estimate.
    Replaced by TrajectoryHead for the C4 trajectory extension.
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)   # (B, 2)


class TrajectoryHead(nn.Module):
    """Probabilistic trajectory head: context → Gaussian over T_future waypoints.

    TODO (C4): outputs (mu, log_sigma) per waypoint so NLL and ECE can be computed.
    At each degradation level we measure whether predicted uncertainty widens
    correctly as LiDAR beam count drops — this is the calibrated graceful degradation claim.

    Output shape: mu (B, T_future, 2), log_sigma (B, T_future, 2)
    Loss: NLL = sum_t [ log_sigma_t + 0.5*(y_t - mu_t)^2 / exp(2*log_sigma_t) ]
    """

    def __init__(self, hidden_size: int = 256, T_future: int = 6):
        super().__init__()
        self.T_future = T_future
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, T_future * 2),   # (B, T_future*2) → reshape to (B, T_future, 2)
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, T_future * 2),
        )
        # clamp log_sigma to avoid degenerate distributions
        self._log_sigma_min = -4.0
        self._log_sigma_max =  4.0

    def forward(self, ctx: torch.Tensor):
        B = ctx.shape[0]
        mu        = self.mu_head(ctx).view(B, self.T_future, 2)
        log_sigma = self.log_sigma_head(ctx).view(B, self.T_future, 2)
        log_sigma = log_sigma.clamp(self._log_sigma_min, self._log_sigma_max)
        return mu, log_sigma   # (B, T_future, 2), (B, T_future, 2)
