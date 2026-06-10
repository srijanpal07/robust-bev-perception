"""Generate degradation curves: ADE / FDE / ECE vs. LiDAR beam count.

This is Figure 1 of the paper. Runs the model at each beam level
{32, 16, 8, 4, 2, 0} and plots one curve per ablation variant.

Usage:
    python scripts/eval/eval_degradation_curve.py \
        --checkpoint outputs/run_dropout/ckpt_best.pt \
        --config configs/eval_degradation.yaml \
        --output outputs/figures/degradation_curve.png
"""

import argparse
import yaml
import torch

from src.models.temporal import TemporalVelocityPredictor
from src.eval.degradation_curves import run_degradation_sweep, plot_degradation_curves
from src.training.checkpointing import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--output',     default=None)
    p.add_argument('--metric',     default='ade_3s',
                   choices=['ade_1s', 'ade_2s', 'ade_3s',
                            'fde_1s', 'fde_2s', 'fde_3s', 'ece'])
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalVelocityPredictor(**cfg['model']).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()

    # TODO: build dataset_factory that applies beam subsampling at each level
    # dataset_factory = lambda n_beams: BEVVelocityDataset(..., n_beams=n_beams)
    # results = run_degradation_sweep(model, dataset_factory, device=str(device))
    # plot_degradation_curves({'Model': results}, metric=args.metric, save_path=args.output)
    raise NotImplementedError("Wire up dataset_factory once beam_degradation.py is implemented.")


if __name__ == '__main__':
    main()
