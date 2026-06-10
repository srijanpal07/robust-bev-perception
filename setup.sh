#!/usr/bin/env bash
# Environment setup for robust-bev-perception
# Usage: bash setup.sh
#
# Creates conda env 'bevrobust' with Python 3.10 + PyTorch 2.7.0 (CUDA 12.4 wheel).
# Driver on this machine supports CUDA 12.8; cu126 is the stable prebuilt pairing for torch 2.7.0
# (cu124 index stops at torch 2.6.0).
# Isaac Sim work stays in the separate 'env_isaacsim' environment — do not merge.

set -euo pipefail

ENV_NAME="bevrobust"
PYTHON_VERSION="3.10"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
CUDA_WHEEL="cu126"

# ── 0. guard ──────────────────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[bevrobust] Conda env '${ENV_NAME}' already exists."
  echo "  To rebuild from scratch: conda env remove -n ${ENV_NAME} && bash setup.sh"
  exit 0
fi

# ── 1. create env ─────────────────────────────────────────────────────────────
echo "[bevrobust] Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate — works whether the user sourced conda init or not
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[bevrobust] Python: $(python --version)"

# ── 2. PyTorch with CUDA wheel ────────────────────────────────────────────────
echo "[bevrobust] Installing PyTorch ${TORCH_VERSION} (${CUDA_WHEEL})..."
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

# ── 3. project dependencies ───────────────────────────────────────────────────
echo "[bevrobust] Installing project dependencies..."
pip install -r requirements.txt

# ── 4. install project as editable package ────────────────────────────────────
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
  echo "[bevrobust] Installing project in editable mode..."
  pip install -e .
else
  # No setup.py yet — add repo root to path so 'from src.models...' works
  SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "${REPO_ROOT}" > "${SITE_PACKAGES}/bevrobust.pth"
  echo "[bevrobust] Added repo root to Python path via .pth file: ${REPO_ROOT}"
fi

# ── 5. register Jupyter kernel ────────────────────────────────────────────────
echo "[bevrobust] Registering Jupyter kernel..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "bevrobust (3.10)"

# ── 6. smoke test ─────────────────────────────────────────────────────────────
echo "[bevrobust] Running smoke test..."
python - <<'EOF'
import torch
import numpy as np
from nuscenes.nuscenes import NuScenes  # import only — don't load data

print(f"  torch      {torch.__version__}")
print(f"  CUDA avail {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  numpy      {np.__version__}")
print("  nuscenes-devkit  OK")
EOF

# ── 7. done ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  bevrobust env ready."
echo ""
echo "  Activate:   conda activate bevrobust"
echo "  Train:      python scripts/train/train.py --config configs/train_baseline.yaml"
echo "  Tensorboard: tensorboard --logdir outputs/"
echo ""
echo "  When starting the CLIP ablation (C5), add:"
echo "    pip install transformers peft"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
