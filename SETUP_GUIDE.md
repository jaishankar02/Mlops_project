# StyleSync: Phase 1 Setup Guide

## Overview
This guide walks you through setting up the **Phase 1: Recommender Only** implementation of StyleSync.

### What's in Phase 1?
✅ Visual fashion recommendations using CLIP
✅ FAISS vector search for similarity matching
✅ MLflow experiment tracking
✅ FastAPI backend with async support
✅ Streamlit web interface
✅ Docker containerization
✅ CI/CD with GitHub Actions
✅ Unit tests and code quality checks

⏸️ **Deferred to Phase 2**: GAN-based virtual try-on

---

## Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose**
- **Git**
- **At least 8GB RAM** (for models)
- **CUDA 11.8+** (optional, for GPU acceleration)

---

## Quick Start (Recommended)

### 1. Clone & Navigate
```bash
cd /home/jaishankar/Documents/mlops_project_new
```

### 2. Run Quick Start Script
```bash
chmod +x quick_start.sh
./quick_start.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Set up project structure
- Initialize MLflow

### 3. Start Services (Terminal 1)
```bash
# Activate venv
source venv/bin/activate

# Start all services (backend, frontend, MLflow, DB)
docker-compose up -d
```

### 4. Verify Services (Terminal 1)
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### 5. Access Applications
- **API Documentation**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **MLflow Tracking**: http://localhost:5000

---

## Manual Setup (If Needed)

### Step 1: Create Virtual Environment
```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Create Project Structure
```bash
python setup.py
```

### Step 4: Set Environment Variables
```bash
cp .env.example .env
# Edit .env if needed (defaults should work for local dev)
```

### Step 5: Start Database & MLflow
```bash
docker-compose up -d db mlflow
```

### Step 6: Run Backend
```bash
python -m backend.main
```

### Step 7: Run Frontend (New Terminal)
```bash
source venv/bin/activate
streamlit run frontend/app.py
```

---

## Project Structure Explained

```
mlops_project_new/
├── backend/                      # FastAPI application
│   ├── main.py                  # Entry point
│   ├── routes/
│   │   └── recommender.py       # API endpoints
│   └── schemas.py               # Data models
│
├── ml_models/recommender/        # Feature extraction & recommendations
│   ├── feature_extractor.py     # CLIP/ResNet models
│   └── recommendation_engine.py # FAISS search
│
├── frontend/
│   └── app.py                   # Streamlit UI
│
├── config/
│   ├── settings.py              # Configuration
│   └── mlflow_config.py         # Experiment tracking
│
├── utils/
│   ├── image_processing.py      # Image validation/optimization
│   └── preprocessing.py         # Feature preprocessing
│
├── tests/
│   ├── test_recommender.py      # Unit tests
│   └── conftest.py              # Pytest configuration
│
├── .github/workflows/           # CI/CD pipelines
│   ├── test.yml                 # Test on push
│   └── deploy.yml               # Deploy to GCP
│
├── docker-compose.yml           # Service orchestration
├── Dockerfile                   # Backend container
├── Dockerfile.streamlit         # Frontend container
└── requirements.txt             # Python dependencies
```

---

## API Endpoints Overview

### Upload Garment
```bash
POST /api/recommender/upload-garment
- file: Image file (JPEG/PNG)
- category: Optional category
- color: Optional color
- size: Optional size (XS-XXL)
- price: Optional price

Response: {garment_id: "...", status: "success"}
```

### Bulk Upload
```bash
POST /api/recommender/bulk-upload
- files: Multiple image files (max 50)

Response: {total_uploaded: N, successful: N, failed: N}
```

### Search Similar
```bash
POST /api/recommender/search
- file: Query image
- k: Number of results (1-20, default 5)

Response: {recommendations: [{garment_id, similarity_score, metadata}], ...}
```

### Get Statistics
```bash
GET /api/recommender/stats

Response: {total_items: N, feature_dimension: 512, model_type: "CLIP ViT-B/32", ...}
```

### Save Index
```bash
POST /api/recommender/save-index

Response: {status: "success"}
```

---

## MLflow - Experiment Tracking

Access MLflow UI to monitor:
- Garment uploads
- Search queries
- Processing times
- Model performance

```bash
# Access at http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Logged Events
- `garment_upload`: Individual garment uploads
- `bulk_upload`: Bulk operations
- `search_query`: Search operations with latency metrics
- `index_reset`: Index reset operations

---

## Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_recommender.py::TestRecommendationEngine::test_engine_initialization -v
```

### Check Code Quality
```bash
flake8 .
black --check .
```

---

## Docker Commands

### View Logs
```bash
docker-compose logs -f backend    # Backend logs
docker-compose logs -f frontend   # Frontend logs
docker-compose logs -f mlflow     # MLflow logs
```

### Stop Services
```bash
docker-compose down
```

### Rebuild Services
```bash
docker-compose up -d --build
```

### Access Container Shell
```bash
docker exec -it stylesync-backend bash
```

---

## Troubleshooting

### "Cannot connect to API server"
- Check if backend is running: `docker-compose ps`
- Check logs: `docker-compose logs backend`
- Ensure port 8000 is not in use

### "CLIP model download fails"
- The model (2GB) is downloaded on first use
- Ensure stable internet connection
- Check /tmp for space

### "Database connection error"
- Ensure PostgreSQL is running: `docker-compose up -d db`
- Check credentials in .env match docker-compose.yml

### "Out of memory"
- Reduce batch size for bulk uploads
- Use FAISS on CPU if GPU memory insufficient

---

## Performance Tuning

### For Large Datasets
1. Use PostgreSQL for metadata (already configured)
2. Implement pagination in search results
3. Add Redis caching (commented in docker-compose)

### For Faster Searches
1. Use GPU: Ensure CUDA drivers installed
2. Optimize FAISS index (GPU-accelerated)
3. Cache frequent queries

---

## Phase 2: Virtual Try-On Roadmap

When ready for Phase 2, we'll add:
- GAN models with fallback mechanism
- Pose detection for better alignment
- Size fitting logic
- Real-time preview
- WandB for comparison with MLflow

---

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest tests/`
3. Push to GitHub
4. CI/CD will run tests automatically

---

## Support & References

### Key Technologies
- **CLIP**: https://github.com/openai/CLIP
- **FAISS**: https://github.com/facebookresearch/faiss
- **FastAPI**: https://fastapi.tiangolo.com/
- **Streamlit**: https://streamlit.io/
- **MLflow**: https://mlflow.org/

### Professor Feedback Addressed
1. ✅ Scoped down: Recommender only (GAN deferred)
2. ✅ Experiment tracking: MLflow integrated
3. ✅ CI/CD: GitHub Actions workflows
4. ✅ Fallback mechanism: Ready for Phase 2 implementation

---

## Project Team
- **Bhatt Vasisth** (M25CSA007)
- **Jai Shanakar** (M25CSA014)

**Date**: 16 April 2026
