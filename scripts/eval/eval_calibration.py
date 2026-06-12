"""Full calibration evaluation: ECE, coverage, and reliability diagrams.

Measures whether predicted uncertainty is calibrated at each degradation level.
Requires a model with a probabilistic TrajectoryHead (C3/C4).

Usage:
    python scripts/eval/eval_calibration.py \
        --checkpoint outputs/run_gating/ckpt_best.pt \
        --config configs/eval_degradation.yaml \
        --output outputs/figures/reliability_diagram.png
"""

import argparse
import yaml
import torch

from src.models.phase1_model import Phase1Model
from src.eval.calibration_metrics import calibration_sweep
from src.training.checkpointing import load_checkpoint
from src.data.beam_degradation import BEAM_LEVELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--output',     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Phase1Model(**cfg["model"]).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()

    # TODO: wire up once TrajectoryHead and calibration_sweep are implemented
    # results = calibration_sweep(model, dataset_factory, BEAM_LEVELS, device=str(device))
    # for beam, metrics in results.items():
    #     print(f"beams={beam:2d}  ECE={metrics['ece']:.4f}  NLL={metrics['nll']:.4f}  "
    #           f"cov90={metrics['coverage_90']:.3f}")
    raise NotImplementedError("Requires TrajectoryHead (C4) and calibration_sweep (C3).")


if __name__ == '__main__':
    main()
