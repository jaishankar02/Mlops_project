#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/home/m25csa007/Mlops_project/backend/hr_viton}"
VENV_DIR="${2:-$REPO_DIR/.venv_hr}"
HR_VITON_REPO_URL="${HR_VITON_REPO_URL:-https://github.com/sangyun884/HR-VITON.git}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_HR_VENV="${INSTALL_HR_VENV:-false}"

TOCG_ID="1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ"
TOCG_DIS_ID="1T4V3cyRlY5sHVK7Quh_EJY5dovb5FxGX"
GEN_ID="1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy"
ALEXNET_ID="1FF3BBSDIA3uavmAiuMH6YFCv09Lt8jUr"

mkdir -p "$(dirname "$REPO_DIR")"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Cloning official HR-VITON repo to $REPO_DIR"
  git clone "$HR_VITON_REPO_URL" "$REPO_DIR"
else
  echo "HR-VITON repo already present: $REPO_DIR"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Required interpreter not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN to a valid Python 3.10+ executable and rerun."
  exit 1
fi

if [[ "$INSTALL_HR_VENV" == "true" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  python -m pip install --upgrade pip setuptools wheel

  # Matches current project CUDA stack and keeps compatibility with this host.
  if ! pip install --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1; then
    # Fallback for environments where CUDA wheel index resolution is restricted.
    pip install --upgrade --force-reinstall \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
  fi

  pip install --upgrade --force-reinstall \
    opencv-python \
    torchgeometry \
    Pillow \
    tqdm \
    tensorboardX \
    scikit-image \
    scipy \
    gdown
else
  # Fast path: only download official checkpoints for backend integration.
  if ! command -v gdown >/dev/null 2>&1; then
    "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install gdown
  fi
fi

WEIGHTS_DIR="$REPO_DIR/eval_models/weights/v0.1"
mkdir -p "$WEIGHTS_DIR"

if [[ ! -f "$WEIGHTS_DIR/mtviton.pth" ]]; then
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=$TOCG_ID" -O "$WEIGHTS_DIR/mtviton.pth"
fi

if [[ ! -f "$WEIGHTS_DIR/mtviton_discriminator.pth" ]]; then
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=$TOCG_DIS_ID" -O "$WEIGHTS_DIR/mtviton_discriminator.pth"
fi

if [[ ! -f "$WEIGHTS_DIR/gen.pth" ]]; then
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=$GEN_ID" -O "$WEIGHTS_DIR/gen.pth"
fi

if [[ ! -f "$WEIGHTS_DIR/alexnet.pth" ]]; then
  "$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=$ALEXNET_ID" -O "$WEIGHTS_DIR/alexnet.pth"
fi

cat <<EOF

HR-VITON setup complete.
Repo: $REPO_DIR
Python: ${VENV_DIR}/bin/python (if INSTALL_HR_VENV=true)
Weights: $WEIGHTS_DIR

Use these exports in your backend shell:
  export HR_VITON_LOCAL_REPO_PATH="$REPO_DIR"
  export HR_VITON_TOCG_CHECKPOINT="$WEIGHTS_DIR/mtviton.pth"
  export HR_VITON_GEN_CHECKPOINT="$WEIGHTS_DIR/gen.pth"
  export HR_VITON_ENABLED=true
  export HR_VITON_PYTHON_BIN="/home/m25csa007/Mlops_project/.venv/bin/python"

EOF
