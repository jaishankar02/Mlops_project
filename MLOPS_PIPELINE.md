# StyleSync MLOps Pipeline - Comprehensive Guide

## Overview

This document describes the complete MLOps infrastructure for the StyleSync recommender system, addressing production ML requirements including model management, data versioning, experiment tracking, and deployment monitoring.

---

## 1. Model Management & Registry

### Model Registry (`ml_models/model_registry.py`)

**Purpose**: Centralized model versioning, tracking, and lifecycle management.

**Features**:
- Pretrained model downloads (CLIP, ResNet50, EfficientNet)
- Model status tracking (PRODUCTION, STAGING, ARCHIVED, EXPERIMENTAL)
- File integrity checking (SHA256 hashing)
- MLflow integration for experiment tracking

**Pretrained Models Available**:

| Model | Task | Size | Feature Dim |
|-------|------|------|------------|
| CLIP ViT-B/32 | Feature Extraction | 352MB | 512 |
| CLIP ViT-L/14 | Feature Extraction | 903MB | 768 |
| ResNet50 | Feature Extraction | 102MB | 2048 |
| EfficientNet-B0 | Feature Extraction | 20MB | 1280 |

**Usage**:

```python
from ml_models.model_registry import get_model_registry

registry = get_model_registry()

# Download pretrained model
model_path = registry.download_pretrained("clip-vit-b-32")

# Register custom model
registry.register_model(
    model_id="custom-model-v1",
    model_path="/path/to/model.pt",
    task="feature_extraction",
    framework="pytorch",
    metrics={"accuracy": 0.92},
    status=ModelStatus.STAGING
)

# Promote to production
registry.promote_model("custom-model-v1", ModelStatus.PRODUCTION)

# Get model info
info = registry.get_model("custom-model-v1")
```

---

## 2. Data Versioning

### Data Version Manager (`ml_models/data_versioning.py`)

**Purpose**: Track data versions, splits, and quality metrics.

**Features**:
- Dataset versioning with SHA256 hashing
- Train/val/test split management
- Data quality metrics calculation
- Dataset manifest tracking

**Usage**:

```python
from ml_models.data_versioning import get_data_version_manager

manager = get_data_version_manager()

# Register dataset version
manager.register_dataset(
    dataset_id="fashion-catalog-v1",
    description="Fashion items catalog with 10K images",
    data_paths={
        "train": "data/train",
        "val": "data/val",
        "test": "data/test",
    },
    splits={"train": 0.7, "val": 0.15, "test": 0.15},
    schema={"image": "image/jpeg", "id": "string"},
    statistics={"total_items": 10000, "avg_filesize_mb": 1.5}
)

# Get dataset info
dataset_info = manager.get_dataset_info("fashion-catalog-v1")

# Create dataset splits
splits = manager.create_dataset_splits(
    source_dir=Path("data/raw"),
    output_dir=Path("data/splits"),
    splits={"train": 0.7, "val": 0.15, "test": 0.15}
)

# Calculate data quality metrics
stats = DataQualityMetrics.calculate_image_stats(Path("data/train"))
print(f"Total images: {stats['total_images']}")
print(f"Avg size: {stats['avg_width']}x{stats['avg_height']}")
```

---

## 3. Experiment Tracking

### Experiment Tracker (`ml_models/batch_prediction.py`)

**Purpose**: Manage and compare multiple ML experiments.

**Features**:
- Experiment lifecycle management
- Metric logging and comparison
- Artifact tracking
- Best experiment selection

**Usage**:

```python
from ml_models.batch_prediction import ExperimentTracker

tracker = ExperimentTracker()

# Start experiment
exp_id = tracker.start_experiment(
    experiment_name="clip-recommender-v1",
    description="CLIP-based feature extractor",
    parameters={"learning_rate": 0.001, "batch_size": 32}
)

# Log metrics
tracker.log_experiment_metrics(exp_id, {
    "recall@5": 0.87,
    "recall@10": 0.92,
    "inference_time_ms": 45.2
})

# Log artifacts
tracker.log_artifact(exp_id, "model.pt")

# End experiment
tracker.end_experiment(exp_id, status="completed")

# Compare experiments
comparison = tracker.compare_experiments(
    experiment_ids=[exp_id1, exp_id2],
    metric_keys=["recall@5", "inference_time_ms"]
)

# Get best experiment
best_exp = tracker.get_best_experiment("recall@5", mode="max")
```

---

## 4. Model Evaluation

### Model Evaluator (`ml_models/evaluation.py`)

**Purpose**: Comprehensive model evaluation on retrieval tasks.

**Metrics**:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR@K)
- NDCG@K (Normalized Discounted Cumulative Gain)

**Usage**:

```python
from ml_models.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

metrics = evaluator.evaluate_retrieval(
    embeddings_db=db_features,  # (N, D)
    query_embeddings=query_features,  # (Q, D)
    ground_truth={0: [1, 5, 8]},  # query_idx -> relevant_items
    k_values=[1, 5, 10]
)

print(f"Recall@5: {metrics['recall@5']:.4f}")
print(f"NDCG@5: {metrics['ndcg@5']:.4f}")
```

---

## 5. Batch Prediction & Inference

### Batch Prediction Pipeline (`ml_models/batch_prediction.py`)

**Purpose**: Efficient large-scale inference on batches.

**Usage**:

```python
from ml_models.batch_prediction import BatchPredictionPipeline

pipeline = BatchPredictionPipeline(batch_size=32)

# Predict on batch
predictions = pipeline.predict_batch(
    model=feature_extractor,
    data=image_list,
    model_type="feature_extraction",
    device="cuda"
)

# Predict from image files
results = pipeline.predict_from_files(
    model=feature_extractor,
    file_paths=glob("data/images/*.jpg"),
    preprocessor=preprocess_image,
    device="cuda"
)

# Save predictions
pipeline.save_predictions("results/predictions.pkl")
```

---

## 6. Feature Store

### Feature Store (`ml_models/feature_store.py`)

**Purpose**: Cache and manage features across experiments.

**Features**:
- Feature persistence with numpy format
- Metadata tracking
- Feature normalization
- Dimensionality reduction (PCA, UMAP)
- Feature augmentation

**Usage**:

```python
from ml_models.feature_store import get_feature_store, FeatureEngineer

store = get_feature_store()

# Write features to store
store.write_features(
    feature_name="clip-features-v1",
    features=embeddings,  # (N, 512)
    metadata={"model": "clip-vit-b-32", "dataset": "fashion-catalog"}
)

# Read features from store
features = store.read_features("clip-features-v1")

# Feature engineering
normalized = FeatureEngineer.normalize_features(features, method="l2")
reduced = FeatureEngineer.dimensionality_reduction(features, n_components=256, method="pca")
augmented = FeatureEngineer.augment_features(features, augmentation_type="mixup")
```

---

## 7. Model Cards & Documentation

### Model Card Generator (`ml_models/batch_prediction.py`)

**Purpose**: Automatically generate model documentation.

**Usage**:

```python
from ml_models.batch_prediction import ModelCardGenerator

ModelCardGenerator.generate_model_card(
    model_name="clip-recommender-v1",
    model_type="Feature Extraction",
    description="CLIP-based visual feature extractor",
    performance_metrics={"recall@5": 0.87, "precision@5": 0.92},
    training_data={"dataset": "Fashion Catalog", "num_samples": 10000},
    limitations=["Optimized for fashion images"],
    bias_risks=["Potential bias in model training"],
    output_path="model_cards/clip-recommender-v1.md"
)
```

---

## 8. Training Workflow

### Training Pipeline (`ml_models/evaluation.py`)

**Purpose**: End-to-end model training with monitoring.

**Usage**:

```python
from ml_models.evaluation import TrainingPipeline

pipeline = TrainingPipeline()

# Train for epochs
for epoch in range(num_epochs):
    train_loss = pipeline.train_epoch(model, train_loader, optimizer)
    val_loss = pipeline.evaluate(model, val_loader)
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

---

## 9. Production Monitoring

### Model Monitor (`ml_models/evaluation.py`)

**Purpose**: Monitor model performance in production.

**Metrics Tracked**:
- Inference latency (mean, p95, p99)
- Similarity scores distribution
- Performance drift detection

**Usage**:

```python
from ml_models.evaluation import ModelMonitor

monitor = ModelMonitor()

# Log inference metrics
monitor.log_inference(
    query_id="query_123",
    inference_time_ms=45.2,
    results_count=5,
    top_similarity=0.92
)

# Get performance stats
stats = monitor.get_performance_stats(window_size=100)
print(f"P95 Latency: {stats['p95_inference_time_ms']:.2f}ms")

# Detect performance drift
drift_detected = monitor.detect_performance_drift(
    baseline_stats={"avg_inference_time_ms": 40.0},
    threshold=0.2  # 20% drift threshold
)
```

---

## 10. Scripts & Entry Points

### Model Download (`download_models.py`)

```bash
python download_models.py
```

Downloads and registers pretrained models:
- CLIP ViT-B/32
- ResNet50
- Registers with MLflow

### Training Script (`train_models.py`)

```bash
python train_models.py
```

Complete training workflow:
1. Create synthetic data
2. Start MLflow experiment
3. Train model
4. Evaluate on test set
5. Generate model card
6. Save results

---

## 11. Integration with API

### Updated Backend Routes

```python
# Download models on startup
@app.on_event("startup")
async def startup():
    registry = get_model_registry()
    # Models are loaded automatically on first use
```

### Monitoring Endpoint

```python
@app.get("/api/recommender/monitoring")
async def get_monitoring_data():
    monitor = ModelMonitor()
    stats = monitor.get_performance_stats()
    return stats
```

---

## 12. MLOps Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline Workflow                      │
└─────────────────────────────────────────────────────────────────┘

1. Data Management
   └─ DataVersionManager: Version datasets, track splits
       └─ DataQualityMetrics: Calculate statistics

2. Model Development
   └─ ModelEvaluator: Evaluate on test sets
       └─ ExperimentTracker: Track experiments
           └─ BatchPredictionPipeline: Large-scale inference

3. Feature Engineering
   └─ FeatureStore: Cache features
       └─ FeatureEngineer: Transform features

4. Model Registry
   └─ ModelRegistry: Version models
       └─ Promote to production

5. Production Monitoring
   └─ ModelMonitor: Track latency & accuracy
       └─ Detect drift
           └─ Alert if needed

6. Documentation
   └─ ModelCardGenerator: Auto-generate cards
```

---

## 13. Running Complete Pipeline

### Step 1: Download Models
```bash
python download_models.py
```

### Step 2: Train & Evaluate
```bash
python train_models.py
```

### Step 3: Start Services
```bash
docker-compose up -d
```

### Step 4: Access Frontend
- Streamlit: http://localhost:8501
- API: http://localhost:8000
- MLflow: http://localhost:5000

### Step 5: Monitor Performance
```python
# Monitor inferences
from ml_models.evaluation import ModelMonitor

monitor = ModelMonitor()
stats = monitor.get_performance_stats()
print(stats)
```

---

## 14. Best Practices

### Model Management
- ✅ Always version models with registry
- ✅ Use semantic versioning
- ✅ Document model limitations
- ✅ Track training hyperparameters

### Data Management
- ✅ Version all datasets
- ✅ Document data schema
- ✅ Calculate quality metrics
- ✅ Track splits reproducibly

### Experimentation
- ✅ Use experiment tracker for all runs
- ✅ Log all hyperparameters
- ✅ Compare experiments systematically
- ✅ Archive failed experiments

### Production Monitoring
- ✅ Monitor inference latency
- ✅ Track prediction confidence
- ✅ Detect performance drift
- ✅ Set alerting thresholds

### Documentation
- ✅ Generate model cards automatically
- ✅ Document data provenance
- ✅ Create experiment reports
- ✅ Maintain runbooks

---

## 15. Troubleshooting

### Model Download Fails
```bash
# Check internet connection
# Try manually downloading model
wget https://openaipublic.blob.core.windows.net/clip/models/.../ViT-B-32.pt

# Verify download
sha256sum ViT-B-32.pt
```

### Performance Issues
```python
# Check inference latency
stats = monitor.get_performance_stats()
if stats['p95_inference_time_ms'] > 100:
    logger.warning("Latency degradation detected")
```

### Drift Detection
```python
# Check if performance drifted
drift = monitor.detect_performance_drift(baseline_stats)
if drift:
    logger.critical("Model performance drift detected!")
    # Trigger retraining
```

---

## Summary

The StyleSync MLOps pipeline provides:

✅ **Model Management**: Registry, versioning, status tracking
✅ **Data Management**: Versioning, quality metrics, splits
✅ **Experiment Tracking**: Full lifecycle management, comparison
✅ **Model Evaluation**: Comprehensive metrics (Recall, NDCG, MRR)
✅ **Batch Processing**: Efficient large-scale inference
✅ **Feature Engineering**: Store, normalize, augment features
✅ **Production Monitoring**: Latency tracking, drift detection
✅ **Documentation**: Auto-generated model cards

This production-ready infrastructure ensures reproducibility, traceability, and scalability for the recommendation system.
