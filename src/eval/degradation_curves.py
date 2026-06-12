"""Sweep LiDAR beam counts and collect ADE / FDE / ECE — Figure 1 of the paper.

Runs the model at each beam level {32, 16, 8, 4, 2, 0} and collects:
  - ADE at 1 s / 2 s / 3 s
  - FDE at 1 s / 2 s / 3 s
  - ECE (interval coverage at 90%)
  - NLL

Produces one curve per ablation variant (baseline, +C2, +C2+C3) so the
additive gain from each contribution is visible in a single figure.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


BEAM_LEVELS = [32, 16, 8, 4, 2, 0]


@dataclass
class DegradationResult:
    beam_count:  int
    ade_1s:  float = 0.0
    ade_2s:  float = 0.0
    ade_3s:  float = 0.0
    fde_1s:  float = 0.0
    fde_2s:  float = 0.0
    fde_3s:  float = 0.0
    ece:     float = 0.0
    nll:     float = 0.0


def run_degradation_sweep(model: torch.nn.Module,
                          dataset_factory,
                          beam_levels: List[int] = BEAM_LEVELS,
                          device: str = 'cuda') -> List[DegradationResult]:
    """Evaluate model at each beam level and return a list of DegradationResult.

    Args:
        model:           trained Phase1Model
        dataset_factory: callable(n_beams) → Dataset — builds the degraded eval set
        beam_levels:     beam counts to sweep
        device:          torch device string

    Returns:
        List of DegradationResult, one per beam level.
    """
    raise NotImplementedError


def plot_degradation_curves(results_by_variant: Dict[str, List[DegradationResult]],
                             metric: str = 'ade_3s',
                             save_path: str | None = None) -> None:
    """Plot ADE/FDE/ECE vs beam count for multiple model variants.

    Args:
        results_by_variant: {'Baseline': [...], '+Dropout': [...], '+Gating': [...]}
        metric:             which metric to plot on y-axis
        save_path:          if provided, save figure to this path
    """
    raise NotImplementedError
