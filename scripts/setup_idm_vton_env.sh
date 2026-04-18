#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/home/m25csa007/Mlops_project/external/IDM-VTON}"
VENV_DIR="${2:-$REPO_DIR/.venv_idm}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "IDM-VTON repo not found at: $REPO_DIR"
  echo "Clone it first: git clone https://github.com/yisol/IDM-VTON.git $REPO_DIR"
  exit 1
fi

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Torch stack (matches this workstation CUDA runtime better than cu118 pinning in upstream conda file).
pip install --upgrade --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# IDM-VTON core dependency set (close to official).
pip install --upgrade --force-reinstall \
  accelerate==0.25.0 \
  transformers==4.36.2 \
  diffusers==0.25.0 \
  torchmetrics==1.2.1 \
  tqdm==4.66.1 \
  einops==0.7.0 \
  scipy==1.11.1 \
  opencv-python \
  gradio==4.24.0 \
  fvcore \
  cloudpickle \
  omegaconf \
  pycocotools \
  basicsr \
  av \
  onnxruntime==1.16.2

# Keep wrappers predictable.
cat <<EOF

IDM-VTON environment setup complete.
Repo: $REPO_DIR
Python: $VENV_DIR/bin/python

Recommended exports:
  export IDM_VTON_REPO_PATH="$REPO_DIR"
  export IDM_VTON_PYTHON_BIN="$VENV_DIR/bin/python"
  export IDM_VTON_USE_ACCELERATE=false

EOF
