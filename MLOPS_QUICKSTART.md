# MLOps Quick Start Guide

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify Python version
python --version  # Must be 3.9 or higher
```

## Quick Start (5 minutes)

### 1. Download Pretrained Models (1 minute)

```bash
cd /home/jaishankar/Documents/mlops_project_new
python download_models.py
```

**Output**:
```
Downloading CLIP ViT-B/32...
[████████████████████████████] 352MB/352MB
Registered: clip-vit-b-32 (STAGING)

Downloading ResNet50...
[████████████████████████████] 102MB/102MB
Registered: resnet50 (STAGING)

Available Models:
- clip-vit-b-32: 512-dim embeddings
- clip-vit-l-14: 768-dim embeddings (not downloaded)
- resnet50: 2048-dim embeddings
- efficientnet-b0: 1280-dim embeddings (not downloaded)
```

### 2. Run Training Pipeline (2 minutes)

```bash
python train_models.py
```

**Output**:
```
Training MLOps Demo Pipeline
=============================

Creating synthetic dataset (2000 samples)...
Splitting: Train=1400, Val=300, Test=300

Starting experiment: synthetic-training-demo
Epoch 1/5: Loss=0.456
Epoch 2/5: Loss=0.342
Epoch 3/5: Loss=0.278
Epoch 4/5: Loss=0.201
Epoch 5/5: Loss=0.156

Evaluation Results:
├─ Recall@5: 0.847 ± 0.032
├─ Recall@10: 0.923 ± 0.021
├─ NDCG@5: 0.812 ± 0.041
└─ NDCG@10: 0.891 ± 0.028

Model Card: experiments/model_cards/synthetic-training-demo.md
Results: experiments/results_synthetic-training-demo.json
```

### 3. Start Docker Services (2 minutes)

```bash
docker-compose up -d
```

**Check Status**:
```bash
docker-compose ps
```

### 4. Access Services

| Service | URL |
|---------|-----|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## Common Workflows

### Workflow 1: Add New Dataset & Train

```python
# 1. Register dataset
from ml_models.data_versioning import get_data_version_manager

manager = get_data_version_manager()
manager.register_dataset(
    dataset_id="fashion-catalog-v2",
    description="Updated catalog",
    data_paths={"train": "data/train", "val": "data/val", "test": "data/test"}
)

# 2. Create splits
from pathlib import Path
splits = manager.create_dataset_splits(
    source_dir=Path("data/raw"),
    output_dir=Path("data/splits")
)

# 3. Train and evaluate
from ml_models.evaluation import TrainingPipeline
pipeline = TrainingPipeline()
# ... training code
```

### Workflow 2: Compare Model Experiments

```python
from ml_models.batch_prediction import ExperimentTracker

tracker = ExperimentTracker()

# List all experiments
experiments = tracker.list_experiments()

# Compare two experiments
comparison = tracker.compare_experiments(
    experiment_ids=[exp1, exp2],
    metric_keys=["recall@5", "inference_time_ms"]
)

# Get best by metric
best = tracker.get_best_experiment("recall@10", mode="max")
print(f"Best experiment: {best['experiment_name']} "
      f"(Recall@10={best['metrics']['recall@10']:.4f})")
```

### Workflow 3: Production Deployment

```python
from ml_models.model_registry import get_model_registry, ModelStatus

registry = get_model_registry()

# Get model from staging
model_info = registry.get_model("clip-vit-b-32")

# Promote to production
registry.promote_model("clip-vit-b-32", ModelStatus.PRODUCTION)

# Verify
prod_models = registry.list_models(status=ModelStatus.PRODUCTION)
print(f"Models in production: {len(prod_models)}")
```

### Workflow 4: Monitor Performance

```python
from ml_models.evaluation import ModelMonitor

monitor = ModelMonitor()

# Log inference metrics
for inference in inferences:
    monitor.log_inference(
        query_id=inference['id'],
        inference_time_ms=inference['latency'],
        results_count=len(inference['results']),
        top_similarity=inference['score']
    )

# Get stats
stats = monitor.get_performance_stats(window_size=1000)
print(f"P95 Latency: {stats['p95_inference_time_ms']:.2f}ms")
print(f"Avg Similarity: {stats['avg_top_similarity']:.4f}")

# Check for drift
drift = monitor.detect_performance_drift(
    baseline_stats={"avg_inference_time_ms": 40.0},
    threshold=0.2
)
if drift:
    print("⚠️ Performance drift detected!")
```

### Workflow 5: Feature Store Management

```python
from ml_models.feature_store import get_feature_store, FeatureEngineer

store = get_feature_store()

# Cache features
store.write_features(
    feature_name="clip-v1-normalized",
    features=embeddings,
    metadata={"model": "clip-vit-b-32", "normalized": True}
)

# Reuse features
cached = store.read_features("clip-v1-normalized")

# Engineer features
reduced = FeatureEngineer.dimensionality_reduction(
    cached, 
    n_components=256,
    method="pca"
)
```

---

## Monitoring & Debugging

### Check MLflow Experiments

```bash
# View MLflow UI
open http://localhost:5000

# Or access via Python
from ml_models.batch_prediction import ExperimentTracker

tracker = ExperimentTracker()
experiments = tracker.list_experiments()
for exp in experiments:
    print(f"{exp['experiment_name']}: {exp['status']}")
```

### View Model Registry

```python
from ml_models.model_registry import get_model_registry

registry = get_model_registry()
models = registry.list_models()
for model in models:
    print(f"Model: {model['model_id']}")
    print(f"  Status: {model['status']}")
    print(f"  Created: {model['created_at']}")
```

### Check Data Quality

```python
from ml_models.data_versioning import DataQualityMetrics
from pathlib import Path

stats = DataQualityMetrics.calculate_image_stats(Path("data/train"))
print(f"Total images: {stats['total_images']}")
print(f"Corrupted: {stats['corrupted_count']}")
print(f"Average size: {stats['avg_width']}x{stats['avg_height']}")
```

### Monitor System Health

```bash
# Check Docker services
docker-compose ps

# View service logs
docker-compose logs backend
docker-compose logs frontend

# Check API health
curl http://localhost:8000/health
```

---

## Troubleshooting

### Issue: Model Download Fails

**Error**: `Connection refused` or `Download timeout`

**Solution**:
```bash
# Check internet connection
ping google.com

# Try with timeout increase
python -c "
from ml_models.model_registry import ModelRegistry
registry = ModelRegistry(timeout=300)  # 5 minutes
registry.download_pretrained('clip-vit-b-32')
"
```

### Issue: Out of Memory (OOM) During Training

**Error**: `CUDA out of memory`

**Solution**:
```python
# Reduce batch size in train_models.py
BATCH_SIZE = 16  # was 32

# Or use CPU
import torch
device = "cpu"  # Force CPU usage
```

### Issue: Docker Services Won't Start

**Error**: `docker-compose up` fails

**Solution**:
```bash
# Clean up old containers
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Try again
docker-compose up -d
```

### Issue: MLflow Experiments Not Showing

**Error**: Experiments empty in MLflow UI

**Solution**:
```bash
# Check if tracking URI is correct
python -c "
import mlflow
current_uri = mlflow.get_tracking_uri()
print(f'Current tracking URI: {current_uri}')
"

# Set to correct location if needed
export MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Performance Tips

### Optimize Inference Speed

```python
# Use lighter model for speed
model_name = "efficientnet-b0"  # Smaller, faster

# Enable batch processing
from ml_models.batch_prediction import BatchPredictionPipeline
pipeline = BatchPredictionPipeline(batch_size=64)  # Increase for throughput
```

### Optimize Memory Usage

```python
# Use feature dimensionality reduction
from ml_models.feature_store import FeatureEngineer
reduced = FeatureEngineer.dimensionality_reduction(
    features, 
    n_components=128,  # Reduce from 512
    method="pca"
)

# Reduces storage from 512 to 128 dims per embedding
```

### Optimize GPU Usage

```python
# Ensure GPU is available
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Use half precision for faster computation
pipeline = BatchPredictionPipeline(dtype=torch.float16)
```

---

## Next Steps

1. **Explore the Streamlit App**: Navigate to http://localhost:8501
2. **Check MLflow Experiments**: View at http://localhost:5000
3. **Upload Custom Data**: Add your fashion images to `data/` directory
4. **Run Custom Model**: Modify `train_models.py` with your data
5. **Deploy to Production**: Use Docker image from CI/CD pipeline

---

## Useful Commands

```bash
# View model registry
python -c "from ml_models.model_registry import get_model_registry; print(get_model_registry().list_models())"

# View data versions
python -c "from ml_models.data_versioning import get_data_version_manager; print(get_data_version_manager().get_dataset_info('fashion-catalog-v1'))"

# View experiments
python -c "from ml_models.batch_prediction import ExperimentTracker; print(ExperimentTracker().list_experiments())"

# Check feature store
python -c "from ml_models.feature_store import get_feature_store; print(get_feature_store().list_features())"

# View performance stats
python -c "from ml_models.evaluation import ModelMonitor; print(ModelMonitor().get_performance_stats())"

# Generate model card
python -c """
from ml_models.batch_prediction import ModelCardGenerator
ModelCardGenerator.generate_model_card(
    model_name='clip-v1', 
    model_type='Feature Extraction',
    description='Production CLIP model',
    performance_metrics={'recall@5': 0.87}
)
"""
```

---

## Support

For issues or questions:
1. Check [MLOPS_PIPELINE.md](MLOPS_PIPELINE.md) for detailed documentation
2. Review [README.md](README.md) for project overview
3. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for installation issues
4. File issues in GitHub repository

Happy ML Ops! 🚀
