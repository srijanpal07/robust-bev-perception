import argparse
import json
import os, sys
import time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import yaml
from src.dataset import BEVVelocityDataset
from src.model   import TemporalVelocityPredictor
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/train.yaml')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

DATA_DIR         = cfg['data']['data_dir']
META_PATH        = cfg['data']['meta_path']
CKPT_PATH        = os.path.join(cfg['output']['ckpt_dir'], 'best_model.pt')
RESULTS_DIR      = os.path.join(cfg['output'].get('results_dir', 'outputs/results'))
BOX_NOISE        = cfg['data'].get('box_noise', False)
BOX_NOISE_PARAMS = cfg['data'].get('box_noise_params', {})

# Load checkpoint
ckpt           = torch.load(CKPT_PATH, map_location='cpu')
T_MODEL        = ckpt['T']                                      # BEV temporal steps (model)
USE_SUBFRAMES  = ckpt.get('use_subframes', False)
T              = ckpt.get('T_kf', T_MODEL // 2 if USE_SUBFRAMES else T_MODEL)  # keyframe count (dataset)
VAL_LAST_N     = ckpt.get('val_last_n',    5)
SPLIT_MODE     = ckpt.get('split_mode',    'temporal')
VAL_SCENES     = ckpt.get('val_scenes',    None)
DELTA_BEV      = ckpt.get('delta_bev',     False)
ADD_KINEMATICS = ckpt.get('add_kinematics', False)
USE_KALMAN     = ckpt.get('use_kalman',     True)
TEMPORAL_MODEL  = ckpt.get('temporal_model', 'gru')
NHEAD           = ckpt.get('transformer_nhead',           4)
NUM_LAYERS      = ckpt.get('transformer_num_layers',      2)
DIM_FEEDFORWARD = ckpt.get('transformer_dim_feedforward', 512)
HIDDEN_SIZE    = ckpt['hidden_size']
BOX_DIM        = ckpt['box_dim']
DROPOUT        = ckpt.get('dropout',       0.1)
BEV_ENCODER    = ckpt.get('bev_encoder',   'lightweight')
CROP_ENCODER   = ckpt.get('crop_encoder',  'lightweight')
BEV_CHANNELS   = ckpt.get('bev_channels',  3)
CROP_CHANNELS  = ckpt.get('crop_channels', 3)
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TemporalVelocityPredictor(T=T_MODEL, box_dim=BOX_DIM,
                                  hidden_size=HIDDEN_SIZE,
                                  dropout=DROPOUT,
                                  bev_encoder=BEV_ENCODER,
                                  crop_encoder=CROP_ENCODER,
                                  bev_channels=BEV_CHANNELS,
                                  crop_channels=CROP_CHANNELS,
                                  temporal_model=TEMPORAL_MODEL,
                                  nhead=NHEAD,
                                  num_layers=NUM_LAYERS,
                                  dim_feedforward=DIM_FEEDFORWARD).to(device)
_state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in _state):
    _state = {k.removeprefix('_orig_mod.'): v for k, v in _state.items()}
model.load_state_dict(_state)
model.eval()

label_mean     = ckpt['label_mean'].to(device)
label_std      = ckpt['label_std'].to(device)
residual_mean  = ckpt.get('residual_mean', label_mean).to(device)
residual_std   = ckpt.get('residual_std',  label_std).to(device)
RESIDUAL_VEL   = ckpt.get('residual_velocity', False)
box_mean       = ckpt['box_mean'].to(device)
box_std        = ckpt['box_std'].to(device)

dataset = BEVVelocityDataset(META_PATH, DATA_DIR, T=T, split='val',
                             val_last_n=VAL_LAST_N, split_scenes=VAL_SCENES,
                             use_subframes=USE_SUBFRAMES, delta_bev=DELTA_BEV,
                             add_kinematics=ADD_KINEMATICS,
                             use_kalman=USE_KALMAN,
                             box_noise=BOX_NOISE,
                             box_noise_params=BOX_NOISE_PARAMS,
                             rng_seed=ckpt.get('seed', 42))
loader  = DataLoader(dataset, batch_size=1, shuffle=False)

results   = []
latencies = []

with torch.no_grad():
    for i, (bev, crop, box, label) in enumerate(loader):
        bev, crop, box = bev.to(device), crop.to(device), box.to(device)

        box_norm = (box[:, :, :BOX_DIM] - box_mean) / box_std

        start = time.perf_counter()
        raw_pred = model(bev, crop, box_norm)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

        if RESIDUAL_VEL:
            residual   = raw_pred.cpu() * residual_std.cpu() + residual_mean.cpu()
            noisy_vhat = box.cpu()[:, -1, 8:10]
            pred = (residual + noisy_vhat).numpy()[0]
        else:
            pred = (raw_pred.cpu() * label_std.cpu() + label_mean.cpu()).numpy()[0]
        gt = label.numpy()[0]

        # Distance of target vehicle from ego (last keyframe, LiDAR frame x/y)
        dist_m     = float(torch.norm(box[0, -1, :2].cpu()))

        speed_pred = float(np.linalg.norm(pred))
        speed_gt   = float(np.linalg.norm(gt))
        error      = float(np.linalg.norm(pred - gt))

        kf_last  = box.cpu()[0, -1, 8:10].numpy() if ADD_KINEMATICS else None
        kf_error = float(np.linalg.norm(kf_last - gt)) if kf_last is not None else None
        fd_last  = box.cpu()[0, -1, 10:12].numpy() if ADD_KINEMATICS else None
        fd_error = float(np.linalg.norm(fd_last - gt)) if fd_last is not None else None

        results.append({
            'sample_idx': i,
            'dist_m':  dist_m,
            'pred_vx': float(pred[0]), 'pred_vy': float(pred[1]),
            'gt_vx':   float(gt[0]),   'gt_vy':   float(gt[1]),
            'speed_pred': speed_pred,
            'speed_gt':   speed_gt,
            'speed_error': abs(speed_pred - speed_gt),
            'vector_error': error,
            'kf_error': kf_error,
            'fd_error': fd_error,
            'latency_ms': latencies[-1],
        })
        print(f"[{i:03d}] dist={dist_m:5.1f}m  err={error:.3f} m/s  t={latencies[-1]:.1f}ms")
        print(f"  {'':6s}  {'vx':>8s}  {'vy':>8s}  {'spd':>7s}")
        print(f"  {'gt':<6s}  {gt[0]:>+8.2f}  {gt[1]:>+8.2f}  {speed_gt:>6.2f} m/s")
        print(f"  {'pred':<6s}  {pred[0]:>+8.2f}  {pred[1]:>+8.2f}  {speed_pred:>6.2f} m/s")
        if ADD_KINEMATICS:
            box_cpu = box.cpu()[0]          # (T_kf, box_dim+2)
            T_kf    = box_cpu.shape[0]
            for t in range(T_kf):
                kf     = box_cpu[t, 8:10]
                fd     = box_cpu[t, 10:12]  # noisy FD from reference-frame positions
                kf_spd = float(np.linalg.norm(kf.numpy()))
                fd_spd = float(np.linalg.norm(fd.numpy()))
                marker = " ←" if t == T_kf - 1 else "  "
                print(f"  t={t}  kf=({float(kf[0]):+6.2f},{float(kf[1]):+6.2f}) {kf_spd:5.2f}"
                      f"   fd=({float(fd[0]):+6.2f},{float(fd[1]):+6.2f}) {fd_spd:5.2f} m/s{marker}")

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, 'predictions.json'), 'w') as f:
    json.dump(results, f, indent=2)

mae = np.mean([r['vector_error'] for r in results])

summary_lines = []
summary_lines.append(f"Mean vector error:  {mae:.4f} m/s")
summary_lines.append(f"Mean latency:       {np.mean(latencies):.2f} ms/sample")
summary_lines.append(f"Median latency:     {np.median(latencies):.2f} ms/sample")

vhat_label = "KF" if USE_KALMAN else "FD"
kf_errors = [r['kf_error'] for r in results if r['kf_error'] is not None]
if kf_errors:
    mean_kf_err = np.mean(kf_errors)
    delta = mean_kf_err - mae
    pct   = 100.0 * delta / mean_kf_err
    direction = "better" if delta >= 0 else "worse"
    summary_lines.append(f"\nMean {vhat_label} error:      {mean_kf_err:.4f} m/s")
    summary_lines.append(f"Model vs {vhat_label}:        {delta:+.4f} m/s  ({abs(pct):.1f}% {direction} than {vhat_label} alone)")

fd_errors = [r['fd_error'] for r in results if r['fd_error'] is not None]
if fd_errors and USE_KALMAN:
    mean_fd_err = np.mean(fd_errors)
    delta = mean_fd_err - mae
    pct   = 100.0 * delta / mean_fd_err
    direction = "better" if delta >= 0 else "worse"
    summary_lines.append(f"\nMean FD error:      {mean_fd_err:.4f} m/s")
    summary_lines.append(f"Model vs FD:        {delta:+.4f} m/s  ({abs(pct):.1f}% {direction} than FD alone)")

errs  = np.array([r['vector_error'] for r in results])
dists = np.array([r['dist_m']       for r in results])
summary_lines.append(f"\nMedian vector error: {np.median(errs):.4f} m/s")
summary_lines.append(f"P90 vector error:    {np.percentile(errs, 90):.4f} m/s")
summary_lines.append(f"P95 vector error:    {np.percentile(errs, 95):.4f} m/s")
summary_lines.append(f"Max vector error:    {np.max(errs):.4f} m/s")

summary_lines.append("\nError by distance:")
for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 100)]:
    mask = (dists >= lo) & (dists < hi)
    if mask.sum() == 0:
        continue
    summary_lines.append(
        f"  {lo:3d}-{hi:3d}m  n={mask.sum():4d}"
        f"  mean={errs[mask].mean():.3f}"
        f"  median={np.median(errs[mask]):.3f}"
        f"  p90={np.percentile(errs[mask], 90):.3f} m/s"
    )

summary = '\n'.join(summary_lines)
print(f"\n{summary}")

summary_path = os.path.join(RESULTS_DIR, f'infer_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
with open(summary_path, 'w') as f:
    f.write(summary + '\n')
print(f"\nSummary saved to {summary_path}")
