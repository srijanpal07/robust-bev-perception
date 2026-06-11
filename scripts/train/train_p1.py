"""Phase 1 training script — single-agent vehicle.car trajectory forecasting.

Usage:
    conda run -n bevrobust python scripts/train/train_p1.py --config configs/train_p1.yaml
    conda run -n bevrobust python scripts/train/train_p1.py --config configs/train_p1.yaml --resume outputs/phase1/checkpoints/latest.pt

Outputs (all under cfg.output.ckpt_dir):
    best.pt      — checkpoint with lowest val NLL
    latest.pt    — checkpoint after each epoch (for resuming)
    train_log.csv — per-epoch metrics including per-beam NLL/ADE/FDE
"""

import argparse
import csv
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from nuscenes.nuscenes import NuScenes
from src.data.nuscenes_dataset import NuScenesDataset
from src.data.beam_degradation import BEAM_LEVELS
from src.models.phase1_model import Phase1Model
from src.training.losses import combined_trajectory_loss
from src.training.metrics import ade, fde, mean_nll
from src.training.calibration import interval_coverage
from src.training.checkpointing import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(run_cfg: dict) -> Phase1Model:
    m = run_cfg['model']
    return Phase1Model(
        bev_channels=m['bev_channels'],
        box_dim=m['box_dim'],
        box_proj_dim=m['box_proj_dim'],
        lidar_feat_dim=m['lidar_feat_dim'],
        t_future=m['t_future'],
        dropout=m['dropout'],
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             nll_w: float, ade_w: float) -> dict:
    model.eval()
    tot_loss = tot_nll = tot_ade = tot_fde = tot_cov = 0.0
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            mu, log_sigma = model(
                batch['bev'], batch['crop'],
                batch['box_params'], batch['q_score'],
            )
        gt = batch['future_waypoints']
        tot_loss += combined_trajectory_loss(mu, log_sigma, gt, nll_w, ade_w).item()
        tot_nll  += mean_nll(mu, log_sigma, gt).item()
        tot_ade  += ade(mu, gt).item()
        tot_fde  += fde(mu, gt).item()
        tot_cov  += interval_coverage(mu, log_sigma, gt)
        n += 1
    return {
        'loss': tot_loss / n,
        'nll':  tot_nll  / n,
        'ade':  tot_ade  / n,
        'fde':  tot_fde  / n,
        'cov90': tot_cov / n,
    }


@torch.no_grad()
def degradation_eval(model: nn.Module, val_ds: NuScenesDataset,
                     device: torch.device, n_samples: int) -> dict[int, dict]:
    """Evaluate fixed beam levels on a capped subset of val_ds.

    Temporarily overrides val_ds.beam_level for each level (single-process only).
    """
    model.eval()
    indices = list(range(min(n_samples, len(val_ds))))
    orig_beam = val_ds.beam_level

    results: dict[int, dict] = {}
    for bl in BEAM_LEVELS:
        val_ds.beam_level = bl
        sub_loader = DataLoader(
            Subset(val_ds, indices),
            batch_size=32, num_workers=0, shuffle=False,
            pin_memory=(device.type == 'cuda'),
        )
        tot_nll = tot_ade = tot_fde = tot_cov = 0.0
        n = 0
        for batch in sub_loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=device.type == 'cuda'):
                mu, log_sigma = model(
                    batch['bev'], batch['crop'],
                    batch['box_params'], batch['q_score'],
                )
            gt = batch['future_waypoints']
            tot_nll  += mean_nll(mu, log_sigma, gt).item()
            tot_ade  += ade(mu, gt).item()
            tot_fde  += fde(mu, gt).item()
            tot_cov  += interval_coverage(mu, log_sigma, gt)
            n += 1
        results[bl] = {
            'nll': tot_nll / n, 'ade': tot_ade / n,
            'fde': tot_fde / n, 'cov90': tot_cov / n,
        }

    val_ds.beam_level = orig_beam
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(run_cfg: dict, resume: str | None = None):
    t_cfg  = run_cfg['training']
    d_cfg  = run_cfg['data']
    out    = run_cfg['output']

    set_seed(t_cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Directories
    ckpt_dir = Path(out['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(out['log_csv'])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Datasets — workers lazy-load NuScenes once per process via _get_nusc().
    # If index caches exist we skip loading NuScenes in the main process entirely
    # so workers fork with a small footprint (del nusc doesn't release RSS on Linux).
    cache_dir = Path(out.get('cache_dir', 'outputs/phase1/index_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = str(cache_dir / 'train.pkl')
    val_cache   = str(cache_dir / 'val.pkl')

    caches_exist = Path(train_cache).exists() and Path(val_cache).exists()
    if caches_exist:
        nusc = None
    else:
        print(f'Loading nuScenes {d_cfg["version"]}...')
        t0 = time.time()
        nusc = NuScenes(version=d_cfg['version'], dataroot=d_cfg['dataroot'], verbose=False)
        print(f'  loaded in {time.time()-t0:.1f}s')

    print('Building train index...')
    t0 = time.time()
    train_ds = NuScenesDataset(
        dataroot=d_cfg['dataroot'],
        split='train',
        version=d_cfg['version'],
        t_future=d_cfg['t_future'],
        beam_level=None,
        rng_seed=t_cfg['seed'],
        nusc=nusc,
        cache_path=train_cache,
    )
    print(f'  train: {len(train_ds)} samples  ({time.time()-t0:.1f}s)')

    print('Building val index...')
    t0 = time.time()
    val_ds = NuScenesDataset(
        dataroot=d_cfg['dataroot'],
        split='val',
        version=d_cfg['version'],
        t_future=d_cfg['t_future'],
        beam_level=None,
        rng_seed=0,
        nusc=nusc,
        cache_path=val_cache,
    )
    print(f'  val:   {len(val_ds)} samples  ({time.time()-t0:.1f}s)')

    if nusc is not None:
        del nusc
        gc.collect()

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg['batch_size'],
        shuffle=True,
        num_workers=t_cfg['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(t_cfg['num_workers'] > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # Model
    model = build_model(run_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {n_params/1e6:.1f}M trainable parameters')

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg['lr'],
        weight_decay=t_cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg['epochs'], eta_min=t_cfg['lr'] * 0.01,
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    accum_steps = t_cfg.get('accum_steps', 1)
    nll_w = t_cfg.get('nll_weight', 1.0)
    ade_w = t_cfg.get('ade_weight', 0.5)
    grad_clip = t_cfg.get('grad_clip', 1.0)

    # Resume
    start_epoch = 0
    best_val_nll = float('inf')
    if resume:
        meta = load_checkpoint(model, resume, optimizer=optimizer, device=str(device))
        start_epoch = meta.get('epoch', 0) + 1
        best_val_nll = meta.get('metrics', {}).get('val_nll', float('inf'))
        # Rewind scheduler to correct state
        for _ in range(start_epoch):
            scheduler.step()
        print(f'Resumed from epoch {start_epoch-1}, best val NLL={best_val_nll:.4f}')

    # CSV log header
    csv_fields = (
        ['epoch', 'lr', 'train_loss', 'val_nll', 'val_ade', 'val_fde', 'val_cov90']
        + [f'beam{b}_nll' for b in BEAM_LEVELS]
        + [f'beam{b}_ade' for b in BEAM_LEVELS]
        + [f'beam{b}_fde' for b in BEAM_LEVELS]
    )
    write_header = not log_path.exists()
    log_file = open(log_path, 'a', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(log_file, fieldnames=csv_fields, extrasaction='ignore')
    if write_header:
        csv_writer.writeheader()

    # Training
    for epoch in range(start_epoch, t_cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}', leave=False,
                    dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            batch = move_batch(batch, device)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                mu, log_sigma = model(
                    batch['bev'], batch['crop'],
                    batch['box_params'], batch['q_score'],
                )
                loss = combined_trajectory_loss(
                    mu, log_sigma, batch['future_waypoints'], nll_w, ade_w,
                ) / accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f'{loss.item()*accum_steps:.4f}')

        train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # Validation
        val_metrics = validate(model, val_loader, device, nll_w, ade_w)
        model.train()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f'Epoch {epoch:03d}  lr={lr_now:.2e}  '
            f'train_loss={train_loss:.4f}  '
            f'val_nll={val_metrics["nll"]:.4f}  '
            f'val_ade={val_metrics["ade"]:.4f}  '
            f'val_fde={val_metrics["fde"]:.4f}  '
            f'cov90={val_metrics["cov90"]:.3f}'
        )

        # Degradation curve eval
        deg_results: dict[int, dict] = {}
        if (epoch + 1) % t_cfg.get('eval_every', 5) == 0:
            print('  Running degradation eval...')
            deg_results = degradation_eval(
                model, val_ds, device, t_cfg.get('eval_n_samples', 2000),
            )
            for bl, m in deg_results.items():
                print(f'    beam={bl:2d}  nll={m["nll"]:.4f}  '
                      f'ade={m["ade"]:.4f}  fde={m["fde"]:.4f}  cov90={m["cov90"]:.3f}')

        # CSV log row
        row = {
            'epoch': epoch, 'lr': f'{lr_now:.6f}',
            'train_loss': f'{train_loss:.6f}',
            'val_nll': f'{val_metrics["nll"]:.6f}',
            'val_ade': f'{val_metrics["ade"]:.6f}',
            'val_fde': f'{val_metrics["fde"]:.6f}',
            'val_cov90': f'{val_metrics["cov90"]:.6f}',
        }
        for bl in BEAM_LEVELS:
            if bl in deg_results:
                row[f'beam{bl}_nll'] = f'{deg_results[bl]["nll"]:.6f}'
                row[f'beam{bl}_ade'] = f'{deg_results[bl]["ade"]:.6f}'
                row[f'beam{bl}_fde'] = f'{deg_results[bl]["fde"]:.6f}'
        csv_writer.writerow(row)
        log_file.flush()

        # Checkpoints
        metrics_to_save = {'val_nll': val_metrics['nll'], 'val_ade': val_metrics['ade']}
        save_checkpoint(model, optimizer, epoch, metrics_to_save,
                        ckpt_dir / 'latest.pt')
        if val_metrics['nll'] < best_val_nll:
            best_val_nll = val_metrics['nll']
            save_checkpoint(model, optimizer, epoch, metrics_to_save,
                            ckpt_dir / 'best.pt')
            print(f'  ✓ New best val NLL: {best_val_nll:.4f}')

    log_file.close()
    print('Training complete. Best val NLL:', best_val_nll)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume)
