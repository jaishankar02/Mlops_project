# StyleSync Phase 1: Developer Quick Reference

## Command Cheat Sheet

### Setup & Installation
```bash
# Quick start (automated)
./quick_start.sh

# Manual setup
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py
```

### Running Services

#### Using Docker (All in One)
```bash
docker-compose up -d              # Start all services
docker-compose down               # Stop all services
docker-compose logs -f backend    # Stream backend logs
docker-compose ps                 # Check status
```

#### Manual (Development)
```bash
# Terminal 1: Start database & MLflow
docker-compose up -d db mlflow

# Terminal 2: Run backend
python -m backend.main

# Terminal 3: Run frontend
streamlit run frontend/app.py
```

### Testing & QA
```bash
pytest tests/ -v                    # Run tests
flake8 .                           # Code linting
black --check .                    # Format check
black .                            # Auto-format
bandit -r . -ll                    # Security check
```

### Accessing Services
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Streamlit UI**: http://localhost:8501
- **MLflow Dashboard**: http://localhost:5000
- **PostgreSQL**: localhost:5432

### Common Debugging

```bash
# View service logs
docker logs stylesync-backend
docker logs stylesync-frontend
docker logs stylesync-mlflow

# Access container shell
docker exec -it stylesync-backend bash

# Check API health
curl http://localhost:8000/health

# Rebuild specific service
docker-compose up -d --build backend
```

---

## Project Directory Guide

### Key Files to Edit

**API Endpoints**
- `backend/routes/recommender.py` - Add new endpoints here

**Data Models**
- `backend/schemas.py` - Add new request/response models

**ML Models**
- `ml_models/recommender/feature_extractor.py` - Change feature extraction
- `ml_models/recommender/recommendation_engine.py` - Modify search logic

**Frontend Pages**
- `frontend/app.py` - Add UI features/pages

**Configuration**
- `.env` - Environment variables
- `config/settings.py` - App settings
- `config/mlflow_config.py` - Tracking settings

**Tests**
- `tests/test_recommender.py` - Add unit tests

---

## Common Tasks

### Add New API Endpoint
```python
# In backend/routes/recommender.py
@router.get("/my-endpoint")
async def my_endpoint():
    """Description."""
    return {"result": "data"}
```

### Add New Streamlit Page
```python
# In frontend/app.py
if page == "My Page":
    st.subheader("My Page")
    # Add UI here
```

### Log Metrics to MLflow
```python
from config.mlflow_config import log_recommendation_event

log_recommendation_event("search_query", {
    "k_results": 5,
    "processing_time_ms": 45.2,
    "results_count": 5
})
```

### Extract Image Features
```python
from ml_models.recommender.feature_extractor import get_feature_extractor
from PIL import Image

extractor = get_feature_extractor("clip")
image = Image.open("garment.jpg")
features = extractor.extract_image_features(image)
```

---

## Deployment Checklist

### Pre-Deployment
```bash
[ ] Run full test suite: pytest tests/ -v
[ ] Check code quality: flake8 .
[ ] Format code: black .
[ ] Security scan: bandit -r .
[ ] Build Docker: docker-compose build
[ ] Test in Docker: docker-compose up -d
```

### Deployment to GCP
```bash
[ ] Set GCP project ID
[ ] Configure credentials
[ ] Update .env with production values
[ ] Build Docker: docker build -t gcr.io/project/stylesync .
[ ] Push to GCR: docker push gcr.io/project/stylesync
[ ] Deploy to Compute Engine
[ ] Verify health checks
```

### Post-Deployment
```bash
[ ] Check API at /health
[ ] Access MLflow dashboard
[ ] Verify data persistence
[ ] Monitor logs
[ ] Load test with examples.py
```

---

## Troubleshooting Matrix

| Problem | Solution |
|---------|----------|
| Port 8000 already in use | Kill process: `lsof -i :8000` |
| CLIP download fails | Check internet, try again |
| DB connection error | Ensure postgres running: `docker-compose up -d db` |
| Out of memory | Stop services: `docker-compose down` |
| API returns 500 | Check logs: `docker-compose logs backend` |
| Streamlit won't load | Ensure API is running on :8000 |

---

## Environment Variables Reference

```
# API
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Models
FEATURE_EXTRACTOR=clip  # or 'resnet'
USE_GPU=True

# Database
DATABASE_URL=postgresql://user:pass@host/db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=recommender-phase-1

# Storage
FAISS_INDEX_PATH=data/faiss_index.bin

# Image Processing
MAX_IMAGE_SIZE_MB=10
IMAGE_RESIZE_HEIGHT=224
```

---

## API Quick Reference

### Upload Garment
```
POST /api/recommender/upload-garment
Content-Type: multipart/form-data

Parameters:
- file: Image (required)
- category: string (optional)
- color: string (optional)
- size: string (optional)
- price: float (optional)

Response: {garment_id, message, status, timestamp}
```

### Search Similar
```
POST /api/recommender/search
Content-Type: multipart/form-data

Parameters:
- file: Image (required)
- k: int (1-20, default 5)

Response: {recommendations, total_count, processing_time_ms, model_used}
```

### Get Stats
```
GET /api/recommender/stats

Response: {total_items, feature_dimension, model_type, index_size_mb}
```

---

## Performance Monitoring

### Via MLflow
```
http://localhost:5000
- View all experiments
- Check run metrics
- Download artifacts
```

### Via API
```
curl http://localhost:8000/metrics
curl http://localhost:8000/health
```

### Via Docker
```
docker stats stylesync-backend
docker logs --tail 100 stylesync-backend
```

---

## Phase 2 Integration

### When Ready for Try-On:

1. **Uncomment in backend/main.py**
```python
from backend.routes import tryon
app.include_router(tryon.router, prefix="/api/tryon")
```

2. **Add to frontend/app.py**
```python
elif page == "👕 Try-On":
    tryon_ui()
```

3. **Download Models**
```bash
# Pre-trained weights
# Store in models/
```

4. **Update CI/CD**
```yaml
# .github/workflows/test.yml
- Run try-on tests
```

---

## Documentation Structure

- **README.md** - Overview, features, quick start
- **SETUP_GUIDE.md** - Detailed setup instructions
- **IMPLEMENTATION_SUMMARY.md** - What was implemented, architecture
- **IMPLEMENTATION_CHECKLIST.md** - Complete feature checklist
- **QUICK_REFERENCE.md** - This file
- **examples.py** - Code examples
- **API Docs** - http://localhost:8000/docs

---

## Getting Help

### Check Documentation
1. README.md - General info
2. SETUP_GUIDE.md - Setup issues
3. examples.py - Usage examples

### Debug Steps
1. Check Docker: `docker-compose ps`
2. Check logs: `docker-compose logs backend`
3. Hit health endpoint: `curl localhost:8000/health`
4. Run tests: `pytest tests/ -v`

### Common Issues
- **API won't start**: Check .env, check port 8000
- **No results**: Need to upload garments first
- **Slow search**: GPU not detected, check CUDA

---

## Team Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes, test locally
pytest tests/ -v
flake8 .
black .

# 3. Commit & push
git add .
git commit -m "Add feature description"
git push origin feature/your-feature

# 4. Create PR on GitHub
# CI/CD will run automatically

# 5. Merge after approval
# Deployment workflow triggers on main
```

---

## Additional Resources

- **CLIP**: https://github.com/openai/CLIP
- **FAISS**: https://github.com/facebookresearch/faiss
- **FastAPI**: https://fastapi.tiangolo.com/
- **Streamlit**: https://streamlit.io/
- **MLflow**: https://mlflow.org/
- **Docker**: https://docs.docker.com/

---

*Last updated: 16 April 2026*
