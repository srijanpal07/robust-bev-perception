"""Cross-domain robustness evaluation on Isaac Sim physics-accurate degraded scenes.

The model is trained exclusively on real nuScenes data. This script evaluates
it on Isaac Sim outputs — no Isaac Sim data seen during training.

This is NOT sim-to-real transfer. It is cross-domain robustness validation:
does stochastic beam-dropout training (C2) generalise to physics-accurate
degradation (fog, rain, 4-beam hardware LiDAR) beyond artificially subsampled beams?

Usage:
    python scripts/eval/eval_isaac_sim.py \
        --checkpoint outputs/run_dropout/ckpt_best.pt \
        --isaac-data data/isaac_sim_scenes/ \
        --config configs/eval_isaac_sim.yaml \
        --output outputs/eval_isaac_sim_results.json
"""

import argparse
import json
import yaml
import torch

from src.models.temporal import TemporalVelocityPredictor
from src.data.isaac_sim import IsaacSimDataset
from src.training.checkpointing import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--isaac-data',  required=True, dest='isaac_data')
    p.add_argument('--config',      required=True)
    p.add_argument('--output',      default='eval_isaac_sim_results.json')
    p.add_argument('--condition',   default='all',
                   help='Degradation condition filter: all | fog | rain | 4beam | lidar_fail')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalVelocityPredictor(**cfg['model']).to(device)
    load_checkpoint(model, args.checkpoint, device=str(device))
    model.eval()

    # TODO: implement once IsaacSimDataset is ready (Month 1, Week 4)
    # dataset = IsaacSimDataset(args.isaac_data, degradation_condition=args.condition)
    # loader  = DataLoader(dataset, batch_size=cfg['eval']['batch_size'], shuffle=False)
    # results = run_eval(model, loader, device)
    # with open(args.output, 'w') as f:
    #     json.dump(results, f, indent=2)
    raise NotImplementedError("Requires IsaacSimDataset — pending Isaac Sim scene generation.")


if __name__ == '__main__':
    main()
