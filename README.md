# Mlops_project

Full-stack fashion MLOps project with:
- visual garment recommendation (CLIP plus FAISS)
- virtual try-on (IDM-VTON integration)
- experiment tracking (MLflow and Weights and Biases)
- FastAPI backend plus Streamlit frontend

## Current Status

- Backend and frontend are active and tested on:
  - backend: http://127.0.0.1:8001
  - frontend: http://127.0.0.1:8501
- IDM-VTON is now vendored inside this repository at `backend/idm_vton`.
- `external/` is no longer required at runtime.

## Project Layout

```text
Mlops_project/
├── backend/
│   ├── main.py
│   ├── routes/
│   │   ├── recommender.py
│   │   └── tryon.py
│   ├── schemas.py
│   └── idm_vton/                 # vendored IDM-VTON runtime/checkpoints
├── frontend/
│   └── app.py
├── ml_models/
│   ├── recommender/
│   └── tryon/
│       └── idm_vton_wrapper.py
├── config/
│   ├── settings.py
│   ├── mlflow_config.py
│   └── wandb_config.py
├── scripts/
│   ├── build_faiss_from_vitonhd.py
│   └── setup_idm_vton_env.sh
├── tests/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Requirements

- Linux with Python 3.9+
- virtual environment support
- GPU optional for recommender, recommended for try-on
- Git LFS for large model/checkpoint files

## Setup

```bash
cd /home/m25csa007/Mlops_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Run Services (Local)

Terminal 1 (backend):

```bash
cd /home/m25csa007/Mlops_project
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

Terminal 2 (frontend):

```bash
cd /home/m25csa007/Mlops_project
source .venv/bin/activate
API_URL=http://127.0.0.1:8001 streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

Optional Terminal 3 (MLflow file backend):

```bash
cd /home/m25csa007/Mlops_project
source .venv/bin/activate
python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns --default-artifact-root ./mlruns
```

## API Endpoints

- health: `GET /health`
- recommender:
  - `POST /api/recommender/upload-garment`
  - `POST /api/recommender/search`
  - `GET /api/recommender/stats`
- try-on:
  - `POST /api/tryon/generate`
  - `POST /api/tryon/generate-streaming`
  - `GET /api/tryon/health`

OpenAPI docs are available at: http://127.0.0.1:8001/docs

## Recommender Index Build

Build FAISS index from a VITON-HD style dataset:

```bash
cd /home/m25csa007/Mlops_project
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/build_faiss_from_vitonhd.py --extract-dir /data/m25csa007/datasets/high_resolution_viton_zalando --limit 5000
```

Useful dataset env vars:

```bash
export RECOMMENDER_DATASET_DIR=/path/to/test/cloth
export TRYON_TRAIN_DATASET_DIR=/path/to/train/cloth
```

## Try-On Integration Notes

- Runtime repository path is local: `backend/idm_vton`.
- Wrapper entry point: `ml_models/tryon/idm_vton_wrapper.py`.
- If needed, override path with:

```bash
export IDM_VTON_LOCAL_REPO_PATH=/absolute/path/to/backend/idm_vton
```

## MLflow and WandB

- MLflow tracker is implemented in `config/mlflow_config.py`.
- WandB tracker is implemented in `config/wandb_config.py`.
- Try-on and recommender events are logged through these wrappers.

## Git LFS for Large Models

This repository uses Git LFS for large artifacts such as:
- `.onnx`, `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.bin`, `.pkl`, `.zip`

If cloning fresh:

```bash
git lfs install
git lfs pull
```

## Tests

```bash
cd /home/m25csa007/Mlops_project
source .venv/bin/activate
pytest tests/ -v
```

## Notes

- Docker is optional and depends on host availability.
- On constrained hosts, local venv workflow is the most reliable path.
- For long-running services, use background sessions (`tmux` or similar).
