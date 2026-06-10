"""Unified training entrypoint for velocity and trajectory prediction.

Usage:
    python scripts/train/train.py --config configs/train_baseline.yaml
    python scripts/train/train.py --config configs/train_dropout.yaml

The task (velocity | trajectory) and all hyperparameters are set in the config file.
This replaces the separate train_velocity.py / train_trajectory.py split.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BEVVelocityDataset
from src.models.temporal import TemporalVelocityPredictor
from src.training.losses import velocity_mse_loss
from src.training.metrics import ade
from src.training.checkpointing import save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to YAML config file')
    p.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    return p.parse_args()


def train(cfg: dict, resume: str | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_ds = BEVVelocityDataset(
        meta_path=cfg['data']['meta_path'],
        data_dir=cfg['data']['data_dir'],
        T=cfg['model']['T'],
        split='train',
        **cfg['data'].get('dataset_kwargs', {}),
    )
    val_ds = BEVVelocityDataset(
        meta_path=cfg['data']['meta_path'],
        data_dir=cfg['data']['data_dir'],
        T=cfg['model']['T'],
        split='val',
    )
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=True,  num_workers=cfg['training'].get('num_workers', 4))
    val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=cfg['training'].get('num_workers', 4))

    # Model
    model = TemporalVelocityPredictor(**cfg['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                                   weight_decay=cfg['training'].get('weight_decay', 1e-4))

    # TODO: load checkpoint if resume is set
    # TODO: add LR scheduler
    # TODO: swap in TrajectoryHead + NLL loss when cfg['task'] == 'trajectory'

    for epoch in range(cfg['training']['epochs']):
        model.train()
        for bev, crop, box, label in train_loader:
            bev, crop, box, label = (x.to(device) for x in (bev, crop, box, label))
            pred = model(bev, crop, box)
            loss = velocity_mse_loss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ade = 0.0
        with torch.no_grad():
            for bev, crop, box, label in val_loader:
                bev, crop, box, label = (x.to(device) for x in (bev, crop, box, label))
                pred = model(bev, crop, box)
                val_ade += ade(pred, label).item()
        val_ade /= len(val_loader)
        print(f"epoch {epoch:03d}  val_ADE={val_ade:.4f}")

        save_checkpoint(model, optimizer, epoch,
                        {'val_ade': val_ade},
                        f"{cfg['training']['output_dir']}/ckpt_{epoch:03d}.pt")


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume)
