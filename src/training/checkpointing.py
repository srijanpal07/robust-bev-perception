"""Checkpoint save / load utilities."""

import torch
import os
from pathlib import Path


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: dict, path: str | Path) -> None:
    """Save model + optimiser state with metadata."""
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'metrics':    metrics,
    }, path)


def load_checkpoint(model: torch.nn.Module, path: str | Path,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: str = 'cpu') -> dict:
    """Load checkpoint into model (and optionally optimizer). Returns metadata dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return {k: v for k, v in ckpt.items() if k not in ('model', 'optimizer')}


def best_checkpoint_path(run_dir: str | Path, metric: str = 'val_ade') -> Path:
    """Return path to the checkpoint with the lowest value of `metric`."""
    raise NotImplementedError
