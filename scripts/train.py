import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset   import BEVVelocityDataset
from src.model     import TemporalVelocityPredictor
from src.kalman    import kalman_velocity
from src.box_noise import add_camera_detector_noise

parser = argparse.ArgumentParser()
parser.add_argument('--config',      default='configs/train.yaml')
parser.add_argument('--resume',      action='store_true',
                    help='Resume training from outputs/checkpoints/latest_ckpt.pt')
parser.add_argument('--resume-from', default=None, metavar='CKPT',
                    help='Resume from a specific checkpoint path')
parser.add_argument('--amp',         action='store_true',
                    help='Enable automatic mixed precision (bfloat16) — ~40%% faster on Ada/Ampere GPUs')
parser.add_argument('--compile',     action='store_true',
                    help='Apply torch.compile to the model — additional ~20%% speedup after warmup')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

DATA_DIR          = cfg['data']['data_dir']
META_PATH         = cfg['data']['meta_path']
NUSCENES_VERSION  = cfg['data'].get('nuscenes_version', 'full')
USE_SUBFRAMES    = cfg['data'].get('use_subframes',    False)
DELTA_BEV        = cfg['data'].get('delta_bev',        False)
ADD_KINEMATICS   = cfg['data'].get('add_kinematics',   False)
BOX_NOISE        = cfg['data'].get('box_noise',        False)
BOX_NOISE_PARAMS = cfg['data'].get('box_noise_params', {})
RESIDUAL_VEL     = cfg['data'].get('residual_velocity',    False) and ADD_KINEMATICS
USE_KALMAN       = cfg['data'].get('use_kalman_velocity', True)  and ADD_KINEMATICS
CKPT_DIR     = cfg['output']['ckpt_dir']
PLOTS_DIR    = cfg['output']['plots_dir']
SEED         = cfg['training']['seed']
T            = cfg['training']['T']
BATCH_SIZE   = cfg['training']['batch_size']
EPOCHS       = cfg['training']['epochs']
LR           = cfg['training']['lr']
WD_ENCODER  = float(cfg['training'].get('weight_decay_encoder',  1.0e-4))
WD_TEMPORAL = float(cfg['training'].get('weight_decay_temporal', 1.0e-3))
WD_OTHER    = float(cfg['training'].get('weight_decay_other',    0.0))
SPLIT_MODE    = cfg['training'].get('split_mode', 'scene')
VAL_LAST_N    = cfg['training'].get('val_last_n', 5)
SCHEDULER     = cfg['training'].get('scheduler', 'cosine')
GRAD_CLIP     = float(cfg['training'].get('grad_clip', 1.0))
HUBER_DELTA   = float(cfg['training'].get('huber_delta', 1.0))
ACCUM_STEPS   = int(cfg['training'].get('accum_steps', 1))
HIDDEN_SIZE   = cfg['model']['hidden_size']
_BASE_BOX_DIM = cfg['model']['box_dim']
BOX_DIM       = _BASE_BOX_DIM + (3 if ADD_KINEMATICS else 0)  # +[dt, vx̂, vŷ] (#21/#22)
DROPOUT       = float(cfg['model'].get('dropout', 0.1))
BEV_ENCODER   = cfg['model'].get('bev_encoder',    'lightweight')
CROP_ENCODER  = cfg['model'].get('crop_encoder',   'lightweight')
TEMPORAL_MODEL  = cfg['model'].get('temporal_model', 'gru')
NHEAD           = int(cfg['model'].get('transformer_nhead',           4))
NUM_LAYERS      = int(cfg['model'].get('transformer_num_layers',      2))
DIM_FEEDFORWARD = int(cfg['model'].get('transformer_dim_feedforward', 512))
BEV_CHANNELS  = cfg['model'].get('bev_channels',  3)
CROP_CHANNELS = cfg['model'].get('crop_channels', 3)
T_MODEL      = T * 2 if USE_SUBFRAMES else T
BEV_CH_MODEL = BEV_CHANNELS * 2 if DELTA_BEV else BEV_CHANNELS

# --- Reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

RESULTS_DIR = cfg['output'].get('results_dir', 'outputs/results')
os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_TIMESTAMP  = datetime.now().strftime('%Y%m%d_%H%M%S')
TRAIN_LOG_PATH = os.path.join(RESULTS_DIR, f'train_log_{RUN_TIMESTAMP}.csv')

print(f"=== Config: {args.config} ===")
print(f"  SEED       = {SEED}")
print(f"  T          = {T}")
print(f"  BATCH_SIZE = {BATCH_SIZE}  (effective {BATCH_SIZE * ACCUM_STEPS} with accum_steps={ACCUM_STEPS})")
print(f"  EPOCHS     = {EPOCHS}")
print(f"  LR         = {LR}")
print(f"  WD_ENCODER   = {WD_ENCODER}  WD_TEMPORAL = {WD_TEMPORAL}  WD_OTHER = {WD_OTHER}")
print(f"  SPLIT_MODE = {SPLIT_MODE}")
print(f"  VAL_LAST_N = {VAL_LAST_N}  (temporal mode only)")
print(f"  SCHEDULER  = {SCHEDULER}")
print(f"  GRAD_CLIP  = {GRAD_CLIP}  HUBER_DELTA = {HUBER_DELTA}")
print(f"  HIDDEN_SIZE  = {HIDDEN_SIZE}")
print(f"  BOX_DIM        = {BOX_DIM}" + (f"  (base {_BASE_BOX_DIM} + 3 kinematics)" if ADD_KINEMATICS else ""))
print(f"  DROPOUT        = {DROPOUT}")
print(f"  BEV_ENCODER    = {BEV_ENCODER}")
print(f"  CROP_ENCODER   = {CROP_ENCODER}")
print(f"  TEMPORAL_MODEL = {TEMPORAL_MODEL}")
if TEMPORAL_MODEL == 'transformer':
    print(f"  TRANSFORMER    = {NUM_LAYERS}L x {NHEAD}H x FFN{DIM_FEEDFORWARD}")
print(f"  CKPT_DIR       = {CKPT_DIR}")
print(f"  USE_SUBFRAMES  = {USE_SUBFRAMES}")
print(f"  DELTA_BEV      = {DELTA_BEV}")
print(f"  ADD_KINEMATICS = {ADD_KINEMATICS}  BOX_NOISE = {BOX_NOISE}  RESIDUAL_VEL = {RESIDUAL_VEL}  USE_KALMAN = {USE_KALMAN}")
print(f"  T_MODEL        = {T_MODEL}  (= T*2 if subframes)")
print(f"  BEV_CH_MODEL   = {BEV_CH_MODEL}  (= bev_channels*2 if delta_bev)")
print(f"  AMP            = {args.amp}  COMPILE = {args.compile}")

# --- Pre-load resume checkpoint (skips expensive stat recomputation) ---
_resume_ckpt = None
_resume_path = args.resume_from or (os.path.join(CKPT_DIR, 'latest_ckpt.pt') if args.resume else None)
if _resume_path:
    if not os.path.exists(_resume_path):
        sys.exit(f"ERROR: resume checkpoint not found: {_resume_path}")
    _resume_ckpt = torch.load(_resume_path, map_location='cpu')
    print(f"\nResuming from: {_resume_path}")
    print(f"  Checkpoint epoch: {_resume_ckpt['epoch'] + 1}  val_loss: {_resume_ckpt['val_loss']:.4f}")

# --- Data ---
_train_scenes = _val_scenes = None
if SPLIT_MODE == 'scene':
    from nuscenes.utils.splits import create_splits_scenes
    _all_splits = create_splits_scenes()
    if NUSCENES_VERSION == 'mini':
        _train_scenes = _all_splits['mini_train']
        _val_scenes   = _all_splits['mini_val']
    else:
        _train_scenes = _all_splits['train']
        _val_scenes   = _all_splits['val']
    print(f"  Scene split : {len(_train_scenes)} train scenes / {len(_val_scenes)} val scenes")

train_dataset = BEVVelocityDataset(META_PATH, DATA_DIR, T=T,
                                    split='train', val_last_n=VAL_LAST_N,
                                    split_scenes=_train_scenes,
                                    use_subframes=USE_SUBFRAMES,
                                    delta_bev=DELTA_BEV,
                                    add_kinematics=ADD_KINEMATICS,
                                    box_noise=BOX_NOISE,
                                    box_noise_params=BOX_NOISE_PARAMS,
                                    use_kalman=USE_KALMAN,
                                    rng_seed=SEED)
val_dataset   = BEVVelocityDataset(META_PATH, DATA_DIR, T=T,
                                    split='val',   val_last_n=VAL_LAST_N,
                                    split_scenes=_val_scenes,
                                    use_subframes=USE_SUBFRAMES,
                                    delta_bev=DELTA_BEV,
                                    add_kinematics=ADD_KINEMATICS,
                                    box_noise=BOX_NOISE,
                                    box_noise_params=BOX_NOISE_PARAMS,
                                    use_kalman=USE_KALMAN,
                                    rng_seed=SEED)

_g = torch.Generator()
_g.manual_seed(SEED)
_num_workers_train = min(8, os.cpu_count() or 1)
_num_workers_val   = min(4, os.cpu_count() or 1)
_pin_memory  = torch.cuda.is_available()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          generator=_g, num_workers=_num_workers_train,
                          pin_memory=_pin_memory, persistent_workers=True,
                          prefetch_factor=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=_num_workers_val, pin_memory=_pin_memory,
                          persistent_workers=True, prefetch_factor=2)
print(f"DataLoader workers: train={_num_workers_train}  val={_num_workers_val}  pin_memory: {_pin_memory}")

print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

# --- Label / residual / box normalization stats ---
# When resuming, load from checkpoint to avoid recomputing over the full dataset.
if _resume_ckpt is not None:
    label_mean    = _resume_ckpt['label_mean']
    label_std     = _resume_ckpt['label_std']
    residual_mean = _resume_ckpt['residual_mean']
    residual_std  = _resume_ckpt['residual_std']
    box_mean      = _resume_ckpt['box_mean']
    box_std       = _resume_ckpt['box_std']
    print("Normalization stats loaded from checkpoint.")
else:
    train_labels = [window[-1]['velocity_gt'] for window in train_dataset.samples]
    all_labels   = torch.tensor(np.array(train_labels), dtype=torch.float32)
    label_mean   = all_labels.mean(0)
    label_std    = all_labels.std(0).clamp(min=1e-6)
    print(f"Label mean (vx, vy): {label_mean.tolist()}")
    print(f"Label std  (vx, vy): {label_std.tolist()}")

    # Compute normalization stats from the actual noisy training distribution.
    # Reproduces per-sample noise with the same seed+idx as the dataset, then uses
    # the correct velocity estimator (KF or FD per USE_KALMAN), fixing two prior bugs:
    #   1. residual stats always called kalman_velocity even when USE_KALMAN=False
    #   2. stats were computed from clean positions instead of noisy positions
    _kf_scale = BOX_NOISE_PARAMS.get('noise_scale', 1.0) if BOX_NOISE else 0.05
    _rng_seed  = train_dataset.rng_seed
    _residuals = []
    _box_rows  = []
    for idx, window in enumerate(train_dataset.samples):
        base_list = [f['box_3d']['center_lidar'] + f['box_3d']['dimensions'] + [f['box_3d']['yaw']]
                     for f in window]
        base_arr = np.array(base_list, dtype=np.float32)
        if BOX_NOISE:
            rng      = (np.random.default_rng([_rng_seed, idx])
                        if _rng_seed is not None else np.random.default_rng())
            base_arr = add_camera_detector_noise(base_arr, rng=rng, **BOX_NOISE_PARAMS)

        if ADD_KINEMATICS:
            noisy_refs = np.array([
                np.array(f['box_3d']['center_ref'][:2])
                + (base_arr[t, :2] - np.array(f['box_3d']['center_lidar'][:2]))
                for t, f in enumerate(window)
            ], dtype=np.float64)
            tss = np.array([f['timestamp'] for f in window])
            if USE_KALMAN:
                vhat_seq = kalman_velocity(noisy_refs, tss, noise_scale=_kf_scale)
            else:
                vhat_seq = np.zeros((len(window), 2))
                for t in range(len(window)):
                    if t == 0:
                        if len(window) > 1:
                            dt = max((tss[1] - tss[0]) / 1e6, 0.1)
                            vhat_seq[0] = (noisy_refs[1] - noisy_refs[0]) / dt
                    else:
                        dt = max((tss[t] - tss[t - 1]) / 1e6, 0.1)
                        vhat_seq[t] = (noisy_refs[t] - noisy_refs[t - 1]) / dt

        for t, frame in enumerate(window):
            row = base_arr[t].tolist()
            if ADD_KINEMATICS:
                if t == 0 and len(window) > 1:
                    dt = max((window[1]['timestamp'] - frame['timestamp']) / 1e6, 0.1)
                elif t > 0:
                    dt = max((frame['timestamp'] - window[t - 1]['timestamp']) / 1e6, 0.1)
                else:
                    dt = 0.5
                row += [dt, float(vhat_seq[t, 0]), float(vhat_seq[t, 1])]
            _box_rows.append(row)

        if RESIDUAL_VEL:
            gt = np.array(window[-1]['velocity_gt'])
            _residuals.append(gt - (vhat_seq[-1] if ADD_KINEMATICS else np.zeros(2)))

    all_box_params = torch.tensor(np.array(_box_rows), dtype=torch.float32)
    box_mean = all_box_params.mean(0)
    box_std  = all_box_params.std(0).clamp(min=1e-6)
    _names = ['x', 'y', 'z', 'l', 'w', 'h', 'yaw'] + (['dt', 'vx_hat', 'vy_hat'] if ADD_KINEMATICS else [])
    print(f"Box  mean {dict(zip(_names, [f'{v:.3f}' for v in box_mean.tolist()]))}")
    print(f"Box  std  {dict(zip(_names, [f'{v:.3f}' for v in box_std.tolist()]))}")

    if RESIDUAL_VEL:
        all_residuals = torch.tensor(np.array(_residuals), dtype=torch.float32)
        residual_mean = all_residuals.mean(0)
        residual_std  = all_residuals.std(0).clamp(min=1e-6)
        print(f"Residual mean (vx, vy): {residual_mean.tolist()}")
        print(f"Residual std  (vx, vy): {residual_std.tolist()}")
    else:
        residual_mean = label_mean
        residual_std  = label_std

# --- Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nCUDA available: {torch.cuda.is_available()} — training on {device}")
label_mean    = label_mean.to(device)
label_std     = label_std.to(device)
residual_mean = residual_mean.to(device)
residual_std  = residual_std.to(device)
box_mean      = box_mean.to(device)
box_std       = box_std.to(device)
model   = TemporalVelocityPredictor(T=T_MODEL, box_dim=BOX_DIM,
                                    hidden_size=HIDDEN_SIZE, dropout=DROPOUT,
                                    bev_encoder=BEV_ENCODER,
                                    crop_encoder=CROP_ENCODER,
                                    bev_channels=BEV_CH_MODEL,
                                    crop_channels=CROP_CHANNELS,
                                    temporal_model=TEMPORAL_MODEL,
                                    nhead=NHEAD,
                                    num_layers=NUM_LAYERS,
                                    dim_feedforward=DIM_FEEDFORWARD).to(device)

if args.compile:
    print("Compiling model with torch.compile …")
    model = torch.compile(model)

# Per-component weight decay: encoder / temporal / other (#17/#18 best practice)
# Biases and 1-D normalization params (BN γ/β, LN γ/β) never get weight decay.
_ENCODER_MODS  = {'bev_encoder', 'crop_encoder'}
_TEMPORAL_MODS = {'gru', 'transformer', 'pos_enc', 'temporal_proj'}
_enc_decay, _enc_nodecay, _tmp_decay, _tmp_nodecay, _other = [], [], [], [], []
for name, param in model.named_parameters():
    prefix = name.split('.')[0]
    is_matrix = param.dim() >= 2   # weight matrices; 1-D = bias / BN / LN param
    if prefix in _ENCODER_MODS:
        ((_enc_decay if is_matrix else _enc_nodecay)).append(param)
    elif prefix in _TEMPORAL_MODS:
        ((_tmp_decay if is_matrix else _tmp_nodecay)).append(param)
    else:
        _other.append(param)
optim = torch.optim.Adam([
    {'params': _enc_decay,   'weight_decay': WD_ENCODER,  'lr': LR},
    {'params': _enc_nodecay, 'weight_decay': 0.0,          'lr': LR},
    {'params': _tmp_decay,   'weight_decay': WD_TEMPORAL, 'lr': LR},
    {'params': _tmp_nodecay, 'weight_decay': 0.0,          'lr': LR},
    {'params': _other,       'weight_decay': WD_OTHER,     'lr': LR},
])
loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

if SCHEDULER == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', patience=5, factor=0.5, min_lr=1e-7)
else:  # cosine (default)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EPOCHS, eta_min=1e-7)

# --- Restore checkpoint state if resuming ---
start_epoch   = 0
best_val_loss = float('inf')
train_losses  = []
val_losses    = []

if _resume_ckpt is not None:
    _state = _resume_ckpt['model_state']
    # Normalise to plain keys first (strip _orig_mod. if present)
    if any(k.startswith('_orig_mod.') for k in _state):
        _state = {k.removeprefix('_orig_mod.'): v for k, v in _state.items()}
    # Re-add prefix when loading into a torch.compile'd (OptimizedModule) model
    if hasattr(model, '_orig_mod'):
        _state = {'_orig_mod.' + k: v for k, v in _state.items()}
    model.load_state_dict(_state)
    optim.load_state_dict(_resume_ckpt['optim_state'])
    scheduler.load_state_dict(_resume_ckpt['sched_state'])
    start_epoch   = _resume_ckpt['epoch'] + 1
    best_val_loss = _resume_ckpt.get('best_val_loss', _resume_ckpt['val_loss'])
    train_losses  = list(_resume_ckpt.get('train_losses', []))
    val_losses    = list(_resume_ckpt.get('val_losses', []))
    print(f"Resumed. Starting from epoch {start_epoch + 1}/{EPOCHS}  best_val_loss={best_val_loss:.4f}")
    if start_epoch >= EPOCHS:
        sys.exit(f"Already trained {start_epoch} epochs. Increase 'epochs' in config to continue.")

# --- Shared checkpoint dict builder ---
def _make_ckpt(epoch, val_loss):
    return {
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'val_loss':    val_loss,
        'best_val_loss': best_val_loss,
        'train_losses':  train_losses,
        'val_losses':    val_losses,
        'T':           T_MODEL,
        'T_kf':        T,
        'hidden_size': HIDDEN_SIZE,
        'box_dim':     BOX_DIM,
        'val_last_n':  VAL_LAST_N,
        'seed':        SEED,
        'label_mean':     label_mean.cpu(),
        'label_std':      label_std.cpu(),
        'residual_mean':  residual_mean.cpu(),
        'residual_std':   residual_std.cpu(),
        'residual_velocity': RESIDUAL_VEL,
        'box_mean':       box_mean.cpu(),
        'box_std':        box_std.cpu(),
        'dropout':       DROPOUT,
        'bev_encoder':   BEV_ENCODER,
        'crop_encoder':  CROP_ENCODER,
        'bev_channels':  BEV_CH_MODEL,
        'crop_channels': CROP_CHANNELS,
        'sched_state':  scheduler.state_dict(),
        'scheduler':    SCHEDULER,
        'grad_clip':    GRAD_CLIP,
        'split_mode':   SPLIT_MODE,
        'val_scenes':   list(_val_scenes) if _val_scenes is not None else None,
        'use_subframes':  USE_SUBFRAMES,
        'delta_bev':      DELTA_BEV,
        'add_kinematics': ADD_KINEMATICS,
        'use_kalman':     USE_KALMAN,
        'temporal_model': TEMPORAL_MODEL,
        'transformer_nhead':           NHEAD,
        'transformer_num_layers':      NUM_LAYERS,
        'transformer_dim_feedforward': DIM_FEEDFORWARD,
    }

# --- Training loop ---
for epoch in range(start_epoch, EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [train]",
                unit="batch", leave=False)
    optim.zero_grad()
    for step, (bev, crop, box, label) in enumerate(pbar):
        bev, crop, box, label = bev.to(device), crop.to(device), box.to(device), label.to(device)
        if RESIDUAL_VEL:
            target = (label - box[:, -1, 8:10] - residual_mean) / residual_std
        else:
            target = (label - label_mean) / label_std
        box_norm = (box[:, :, :BOX_DIM] - box_mean) / box_std
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp):
            pred = model(bev, crop, box_norm)
            loss = loss_fn(pred, target) / ACCUM_STEPS
        loss.backward()
        train_loss += loss.item() * ACCUM_STEPS  # log unscaled loss
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            optim.zero_grad()

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bev, crop, box, label in tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [val]",
                                          unit="batch", leave=False):
            bev, crop, box, label = bev.to(device), crop.to(device), box.to(device), label.to(device)
            if RESIDUAL_VEL:
                target = (label - box[:, -1, 8:10] - residual_mean) / residual_std
            else:
                target = (label - label_mean) / label_std
            box_norm = (box[:, :, :BOX_DIM] - box_mean) / box_std
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp):
                pred = model(bev, crop, box_norm)
                val_loss += loss_fn(pred, target).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)

    # Step scheduler (#19)
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()
    current_lr = optim.param_groups[0]['lr']

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}")

    with open(TRAIN_LOG_PATH, 'a') as _f:
        if epoch == start_epoch:
            _f.write(f"# run:            {RUN_TIMESTAMP}\n")
            _f.write(f"# seed:           {SEED}\n")
            _f.write(f"# T:              {T}\n")
            _f.write(f"# batch_size:     {BATCH_SIZE}  (effective {BATCH_SIZE * ACCUM_STEPS})\n")
            _f.write(f"# epochs:         {EPOCHS}\n")
            _f.write(f"# lr:             {LR}\n")
            _f.write(f"# scheduler:      {SCHEDULER}\n")
            _f.write(f"# grad_clip:      {GRAD_CLIP}\n")
            _f.write(f"# huber_delta:    {HUBER_DELTA}\n")
            _f.write(f"# wd_encoder:     {WD_ENCODER}  wd_temporal: {WD_TEMPORAL}  wd_other: {WD_OTHER}\n")
            _f.write(f"# hidden_size:    {HIDDEN_SIZE}\n")
            _f.write(f"# dropout:        {DROPOUT}\n")
            _f.write(f"# temporal_model: {TEMPORAL_MODEL}\n")
            if TEMPORAL_MODEL == 'transformer':
                _f.write(f"# transformer:    {NUM_LAYERS}L x {NHEAD}H x FFN{DIM_FEEDFORWARD}\n")
            _f.write(f"# bev_encoder:    {BEV_ENCODER}  crop_encoder: {CROP_ENCODER}\n")
            _f.write(f"# bev_channels:   {BEV_CHANNELS}\n")
            _f.write(f"# add_kinematics: {ADD_KINEMATICS}  box_noise: {BOX_NOISE}  residual_vel: {RESIDUAL_VEL}\n")
            _f.write(f"# split_mode:     {SPLIT_MODE}\n")
            if start_epoch > 0:
                _f.write(f"# resumed_from:   epoch {start_epoch}  best_val_loss: {best_val_loss:.6f}\n")
            _f.write('epoch,train_loss,val_loss,lr\n')
        _f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.2e}\n")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(_make_ckpt(epoch, val_loss), os.path.join(CKPT_DIR, 'best_model.pt'))
        print(f"  → Saved best model (val_loss={val_loss:.4f})")

    # Save latest checkpoint every epoch — enables --resume after any interruption
    torch.save(_make_ckpt(epoch, val_loss), os.path.join(CKPT_DIR, 'latest_ckpt.pt'))

# --- Save last epoch checkpoint ---
torch.save(_make_ckpt(epoch, val_loss), os.path.join(CKPT_DIR, 'last_model.pt'))
print(f"Last epoch model saved → {os.path.join(CKPT_DIR, 'last_model.pt')}")

# --- Plot training curves ---
_n_epochs  = len(train_losses)
epochs_plot = range(1, _n_epochs + 1)
best_epoch  = val_losses.index(min(val_losses)) + 1

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs_plot, train_losses, label='Train loss')
ax.plot(epochs_plot, val_losses,   label='Val loss')
ax.axvline(best_epoch, color='grey', linestyle='--', linewidth=0.8,
           label=f'Best epoch ({best_epoch})')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE loss (normalized)')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot_path = os.path.join(PLOTS_DIR, 'loss_curve.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nLoss curve saved to {plot_path}")
