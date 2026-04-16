# StyleSync: AI-Powered Visual Fashion Recommender & Virtual Try-On (GCP-Optimized)

A full-stack MLOps project for visual fashion recommendations and realistic virtual try-ons, optimized for GCP T4 GPUs.

## Project Architecture

### Components
1. **Recommender Branch**: Garment image upload → Feature extraction (CLIP/ResNet) → Vector database (FAISS) → Similar items suggestion
2. **Try-On Branch**: User photo + Selected garment → Preprocessing → Efficient GAN model → Synthesized try-on image
3. **Fallback Mechanism**: SimpleTransform-based try-on when GAN models are resource-constrained
4. **Experiment Tracking**: MLflow for tracking model performance metrics
5. **CI/CD**: GitHub Actions for automated testing and deployment

### Technology Stack
- **Backend**: FastAPI with Python 3.9+
- **Frontend**: Streamlit for UI
- **Models**: 
  - Recommender: CLIP/ResNet50 (pretrained)
  - Try-On: Lightweight GAN (HR-VITON / PF-AFN) + SimpleTransform Fallback
- **Database**: FAISS for vector search + PostgreSQL for metadata
- **Experiment Tracking**: MLflow
- **Deployment**: Docker + GCP Compute Engine (T4 GPUs)
- **API**: FastAPI with async support

## Project Structure

```
mlops_project_new/
├── backend/                   # FastAPI application
│   ├── __init__.py
│   ├── main.py               # Main FastAPI app
│   ├── routes/               # API endpoints
│   │   ├── recommender.py
│   │   └── tryon.py
│   └── schemas.py            # Pydantic models
├── ml_models/                # ML models & inference
│   ├── recommender/          # Feature extraction & recommendations
│   │   ├── __init__.py
│   │   ├── clipmodel.py
│   │   ├── feature_extractor.py
│   │   └── recommendation_engine.py
│   ├── tryon/                # Try-on models
│   │   ├── __init__.py
│   │   ├── gan_model.py      # Efficient GAN model
│   │   ├── simple_transform.py # Lightweight fallback
│   │   └── model_loader.py
│   └── utils.py
├── frontend/                 # Streamlit UI
│   └── app.py
├── database/                 # Database setup
│   ├── vector_db.py          # FAISS setup
│   └── metadata_db.py        # PostgreSQL setup
├── config/                   # Configuration files
│   ├── settings.py           # Settings & environment
│   ├── mlflow_config.py      # MLflow configuration
│   └── model_config.yaml     # Model configurations
├── utils/                    # Utility functions
│   ├── image_processing.py
│   ├── preprocessing.py
│   └── logging.py
├── .github/workflows/        # CI/CD pipelines
│   ├── test.yml
│   └── deploy.yml
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Backend Docker image
├── requirements.txt          # Python dependencies
├── mlflow_tracking.py        # MLflow experiment setup
└── tests/                    # Unit tests
    ├── test_recommender.py
    └── test_tryon.py
```

## Phase 1: Recommender System (Current)

### ✅ What's Implemented
1. **Visual Feature Extraction**
   - CLIP (ViT-B/32) for advanced semantic understanding
   - ResNet50 alternative for fallback
   - Automatic model downloading on first use

2. **Fast Vector Search**
   - FAISS for efficient similarity search
   - Support for large-scale datasets (1M+ items)
   - Persistent index storage (JSON + binary)

3. **MLflow Experiment Tracking** ✅ (Addressing Professor Feedback)
   - Logs upload events and metrics
   - Tracks search query latency
   - Records model performance
   - Experiment versioning and artifact storage

4. **FastAPI Backend**
   - Async request handling
   - Multiple endpoints for upload/search/stats
   - Comprehensive error handling
   - Automatic API documentation (Swagger UI)

5. **Streamlit Web UI**
   - Upload single/multiple garments
   - Real-time similarity search
   - Index statistics dashboard
   - Interactive results display

6. **Production-Ready Architecture**
   - Docker containerization
   - PostgreSQL for metadata
   - Redis for caching (prepared)
   - Health checks & monitoring

7. **CI/CD Pipeline** ✅ (Addressing Professor Feedback)
   - GitHub Actions for automated testing
   - Code quality checks (flake8, black)
   - Security scanning (bandit)
   - Docker image building
   - GCP deployment workflow

8. **Comprehensive Testing**
   - Unit tests for core components
   - Image validation tests
   - Feature extraction tests
   - Pytest configuration

## Phase 2: Virtual Try-On (Deferred - Addresses Ambition Feedback)

### 🔴 Why Deferred to Phase 2?
- **GANs are resource-intensive**: Scoped down as requested
- **Simpler Phase 1 ensures quality**: Focus on robust recommender
- **Fallback mechanism ready**: SimpleTransform implementation prepared
- **Extensible architecture**: Easy to integrate Phase 2 models

### 📋 Phase 2 Features (Ready to Implement)
- **Efficient GAN Models**
  - HR-VITON/PF-AFN architectures
  - FP16 precision optimization
  - T4 GPU support

- **Fallback Mechanism**
  - Automatic GAN→SimpleTransform switching
  - No service degradation
  - Graceful performance trade-off

- **Advanced Try-On**
  - Pose detection for alignment
  - Size fitting logic
  - Real-time preview
  - WandB integration for model comparison

## Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Navigate to project directory
cd /home/jaishankar/Documents/mlops_project_new

# 2. Start all services
docker-compose up -d

# 3. Wait for services to initialize (30-40 seconds)
sleep 40

# 4. Access applications
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Frontend: http://localhost:8501
# - MLflow: http://localhost:5000
# - Database: postgres://localhost:5432
```

### Local Development Setup

```bash
# 1. Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup project structure
python setup.py

# 4. Start database & MLflow (in separate terminal)
docker-compose up -d db mlflow

# 5. Run backend (Terminal 1)
export PYTHONPATH=$PWD:$PYTHONPATH
python -m backend.main

# 6. Run frontend (Terminal 2)
streamlit run frontend/app.py

# 7. Access Streamlit at http://localhost:8501
```

## Phase 1 Features & Usage

### Feature Extraction
```python
from ml_models.recommender.feature_extractor import get_feature_extractor
from PIL import Image

# Load extractors
clip_extractor = get_feature_extractor("clip")

# Extract image features
image = Image.open("garment.jpg")
features = clip_extractor.extract_image_features(image)

# Or extract text features (queries)
text_features = clip_extractor.extract_text_features("red silk dress")
```

### Recommendation Engine
```python
from ml_models.recommender.recommendation_engine import RecommendationEngine
from PIL import Image

engine = RecommendationEngine()
engine.initialize_feature_extractor("clip")

# Add garments to index
image = Image.open("garment.jpg")
engine.add_garment("g123", image, metadata={"category": "dress"})

# Search for similar items
results = engine.search_similar(image, k=5)
# Returns: [(garment_id, similarity_score), ...]

# Save/load index
engine.save_index("data/faiss_index.bin")
engine.load_index("data/faiss_index.bin")
```

### Web Interface Usage

**Via Streamlit (http://localhost:8501)**
1. **🔍 Search Tab**: Upload garment → Find similar items
2. **📤 Upload Tab**: Upload single garment with metadata
3. **📦 Bulk Upload**: Upload multiple garments at once
4. **📈 Statistics**: View index statistics and model info

**Via API (http://localhost:8000)**

```bash
# Single upload
curl -F "file=@garment.jpg" \
  -F "category=shirt" \
  -F "color=blue" \
  http://localhost:8000/api/recommender/upload-garment

# Search similar
curl -F "file=@query.jpg" \
  "http://localhost:8000/api/recommender/search?k=5"

# Get index stats
curl http://localhost:8000/api/recommender/stats

# API docs
curl http://localhost:8000/docs
```

## Experiment Tracking with MLflow

### Access MLflow Dashboard
```bash
# Already running at http://localhost:5000
# Or start manually:
mlflow server --backend-store-uri sqlite:///mlflow.db
```

### Logged Events

**Garment Upload Events**
- Records each upload with garment ID
- Tracks total items in index
- Category and color metadata

**Search Query Events**
- Logs search latency (ms)
- Records number of results
- Tracks similarity scores
- Model version used

**Bulk Upload Events**
- Total uploaded, successful, failed counts
- Batch size and processing time
- Error tracking and categorization

### Example Queries
```python
import mlflow
from config.mlflow_config import get_mlflow_tracker

tracker = get_mlflow_tracker()
tracker.start_run("recommender-experiment")
tracker.log_metrics_batch({
    "search_latency_ms": 45.2,
    "results_count": 5,
    "top_similarity": 0.92
}, step=1)
tracker.end_run()
```

## Testing

```bash
pytest tests/ -v
```

## Deployment to GCP

```bash
# Build and push Docker image
docker build -t gcr.io/PROJECT_ID/stylesync:latest .
docker push gcr.io/PROJECT_ID/stylesync:latest

# Deploy on Compute Engine with T4 GPU
gcloud compute instances create stylesync-server \
  --image=ubuntu-2204-lts \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --container-image=gcr.io/PROJECT_ID/stylesync:latest
```

## References
- CLIP: https://github.com/openai/CLIP
- ResNet: https://pytorch.org/vision/main/models/resnet.html
- HR-VITON: https://github.com/sangyun884/HR-VITON
- PF-AFN: https://github.com/geyuying/PF-AFN
- FAISS: https://github.com/facebookresearch/faiss
- MLflow: https://mlflow.org/

## Authors
- Bhatt Vasisth (M25CSA007)
- Jai Shanakar (M25CSA014)

## Date
10-3-2026
# Mlops_project
