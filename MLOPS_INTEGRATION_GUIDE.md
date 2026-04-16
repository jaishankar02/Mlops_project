# MLOps Components Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         StyleSync MLOps Architecture                 │
└─────────────────────────────────────────────────────────────────────┘

                          FastAPI Backend
                       (main.py + routes)
                               │
                               ├─────────────────────┐
                               │                     │
                    ┌──────────────────┐   ┌──────────────────┐
                    │  Recommender      │   │  MLOps Pipeline  │
                    │  (Phase 1)        │   │  (Phase 2)       │
                    └──────────────────┘   └──────────────────┘
                               │                     │
        ┌──────────────────────┼─────────────────────┼──────────────────┐
        │                      │                     │                  │
        ▼                      ▼                     ▼                  ▼
   Streamlit             FAISS Index            Model Registry    Feature
   Frontend              (Similarity)           & Versioning      Store
   (app.py)              Search                 (model_registry.py)(feature_store.py)
        │                   │                        │                 │
        └───────────────────┼────────────────────────┼─────────────────┘
                            │                        │
                    ┌───────┴────────┬───────────────┴──────┐
                    │                │                      │
                    ▼                ▼                      ▼
              Feature           Data Versioning      Model Evaluation
              Extraction         & Quality             & Monitoring
              (feature_extractor.py)  (data_versioning.py) (evaluation.py)
                    │                │                      │
                    └────────────────┼──────────────────────┘
                                     │
                          ┌──────────┴───────────┐
                          │                      │
                          ▼                      ▼
                    Experiment Tracker    Batch Prediction
                    (batch_prediction.py) (batch_prediction.py)
                          │                      │
                          └──────────┬───────────┘
                                     │
                                     ▼
                              MLflow Tracking
                           (config/mlflow_config.py)
                                     │
                          ┌──────────┴───────────┐
                          │                      │
                          ▼                      ▼
                    TrainingPipeline    Model Cards
                    (evaluation.py)    (batch_prediction.py)
```

---

## 1. Component Definitions

### A. Feature Extraction Layer

**Module**: `ml_models/recommender/feature_extractor.py`

**Responsibility**: Extract embeddings from images/text

**Components**:
```
┌─────────────────────────────────────────┐
│    Feature Extractor                    │
├─────────────────────────────────────────┤
│ • CLIPFeatureExtractor                  │
│   - extract_image_features()            │
│   - extract_text_features()             │
│                                         │
│ • ResNetFeatureExtractor                │
│   - extract_image_features()            │
└─────────────────────────────────────────┘
         │ Output: 512/2048-dim vectors
```

**Integration Points**:
- Input: Images from Streamlit or API
- Output: Embeddings → FAISS Index, Feature Store

### B. Vector Search Layer

**Module**: `ml_models/recommender/recommendation_engine.py`

**Responsibility**: Find similar items using FAISS

**Components**:
```
┌─────────────────────────────────────────┐
│    Recommendation Engine                │
├─────────────────────────────────────────┤
│ • FAISS Index                           │
│   - add_garment()                       │
│   - search_similar()                    │
│                                         │
│ • Index Persistence                     │
│   - save_index()                        │
│   - load_index()                        │
└─────────────────────────────────────────┘
```

**Integration Points**:
- Input: Query embeddings from Feature Extraction
- Output: Top-K similar items
- Storage: Index saved to `ml_models/indexes/`

### C. Model Registry Layer

**Module**: `ml_models/model_registry.py`

**Responsibility**: Manage model versioning and lifecycle

**Components**:
```
┌─────────────────────────────────────────┐
│    Model Registry                       │
├─────────────────────────────────────────┤
│ • Model Status Tracking                 │
│   - PRODUCTION                          │
│   - STAGING                             │
│   - EXPERIMENTAL                        │
│   - ARCHIVED                            │
│                                         │
│ • Pretrained Models                     │
│   - clip-vit-b-32: 512-dim              │
│   - clip-vit-l-14: 768-dim              │
│   - resnet50: 2048-dim                  │
│   - efficientnet-b0: 1280-dim           │
│                                         │
│ • File Integrity                        │
│   - SHA256 hashing                      │
│   - Metadata tracking                   │
└─────────────────────────────────────────┘
```

**Integration Points**:
- Downloads pretrained models on first use
- Registers with MLflow for tracking
- Used by: Feature Extractor, Evaluation

### D. Data Versioning Layer

**Module**: `ml_models/data_versioning.py`

**Responsibility**: Track datasets and maintain quality

**Components**:
```
┌─────────────────────────────────────────┐
│    Data Versioning                      │
├─────────────────────────────────────────┤
│ • Dataset Registration                  │
│   - Version tracking (SHA256)           │
│   - Metadata storage                    │
│                                         │
│ • Split Management                      │
│   - Train/Val/Test splits               │
│   - Reproducible (random_seed)          │
│                                         │
│ • Quality Metrics                       │
│   - Image statistics                    │
│   - Corruption detection                │
│   - Format validation                   │
└─────────────────────────────────────────┘
```

**Integration Points**:
- Input: Raw data from `data/` directory
- Output: Versioned splits for training
- Used by: Training Pipeline, Evaluation

### E. Evaluation & Monitoring Layer

**Module**: `ml_models/evaluation.py`

**Responsibility**: Model evaluation and performance monitoring

**Components**:
```
┌──────────────────────────────────────────┐
│    Evaluation & Monitoring               │
├──────────────────────────────────────────┤
│ • Model Evaluator                        │
│   - Recall@K, Precision@K                │
│   - MRR@K, NDCG@K                       │
│   - Aggregate metrics                    │
│                                          │
│ • Model Monitor                          │
│   - Latency tracking (p95, p99)          │
│   - Similarity distribution              │
│   - Drift detection                      │
│                                          │
│ • Training Pipeline                      │
│   - train_epoch()                        │
│   - evaluate()                           │
│   - Loss tracking                        │
└──────────────────────────────────────────┘
```

**Integration Points**:
- Input: Predictions from Batch Pipeline
- Output: Metrics → MLflow, Dashboard
- Monitoring: Real-time inference statistics

### F. Batch Prediction Layer

**Module**: `ml_models/batch_prediction.py`

**Responsibility**: Large-scale inference and experiment management

**Components**:
```
┌──────────────────────────────────────────┐
│    Batch Prediction                      │
├──────────────────────────────────────────┤
│ • Batch Pipeline                         │
│   - predict_batch()                      │
│   - predict_from_files()                 │
│   - Device management (GPU/CPU)          │
│                                          │
│ • Experiment Tracker                     │
│   - start_experiment()                   │
│   - log_metrics()                        │
│   - compare_experiments()                │
│   - get_best_experiment()                │
│                                          │
│ • Model Card Generator                   │
│   - generate_model_card()                │
│   - Markdown documentation               │
└──────────────────────────────────────────┘
```

**Integration Points**:
- Input: Embeddings from Feature Store
- Output: Predictions & Experiment metadata
- Logging: MLflow experiment tracking

### G. Feature Store Layer

**Module**: `ml_models/feature_store.py`

**Responsibility**: Cache and transform features

**Components**:
```
┌──────────────────────────────────────────┐
│    Feature Store                         │
├──────────────────────────────────────────┤
│ • Feature Store                          │
│   - write_features()                     │
│   - read_features()                      │
│   - list_features()                      │
│                                          │
│ • Feature Engineer                       │
│   - normalize_features()                 │
│   - dimensionality_reduction()           │
│   - augment_features()                   │
│                                          │
│ • Storage                                │
│   - NumPy format (.npy)                  │
│   - JSON metadata                        │
└──────────────────────────────────────────┘
```

**Integration Points**:
- Input: Raw embeddings from Feature Extraction
- Output: Normalized/reduced features
- Used by: Batch Prediction, Evaluation

---

## 2. Data Flow Diagrams

### Flow 1: User Query → Recommendation

```
User Upload Image (Streamlit)
        │
        ▼
API: POST /api/recommender/search
        │
        ▼
Feature Extractor.extract_image_features()
        │ Produces: 512-dim embedding
        ▼
Recommendation Engine.search_similar()
        │ FAISS search
        ▼
Top-K similar items returned
        │
        ▼
ModelMonitor.log_inference()
        │ Logs: latency, top_similarity
        ▼
Response to Streamlit UI
        │
        ▼
User sees results
```

### Flow 2: Model Training Pipeline

```
Raw Data (data/train, data/val, data/test)
        │
        ▼
DataVersionManager.register_dataset()
        │ Creates versioned split
        ▼
Feature Extraction.extract_image_features()
        │ Produces: training embeddings
        ▼
Feature Store.write_features()
        │ Caches for reuse
        ▼
ExperimentTracker.start_experiment()
        │
        ⤵
TrainingPipeline.train_epoch() (multiple epochs)
        │ Logs loss to tracking
        ├─ Forward pass
        ├─ Loss calculation
        ├─ Backward pass
        └─ Optimizer step
        │
        ▼
ModelEvaluator.evaluate_retrieval()
        │ Calculates Recall@K, NDCG@K
        ▼
ExperimentTracker.log_metrics()
        │ Logs to MLflow
        ▼
ModelCardGenerator.generate_model_card()
        │ Auto-generates documentation
        ▼
ExperimentTracker.end_experiment()
        │
        ▼
Results saved to experiments/
```

### Flow 3: Model Deployment & Monitoring

```
ExperimentTracker.get_best_experiment()
        │
        ▼
ModelRegistry.register_model()
        │ Register with status STAGING
        ▼
ModelRegistry.promote_model()
        │ Promote to PRODUCTION
        ▼
Feature Extractor uses production model
        │
        ├─ Every inference logs metrics
        ▼
ModelMonitor.log_inference()
        │ Tracks latency, similarity
        ├─ Appends to inference history
        └─ Calculates statistics
        │
        ▼
ModelMonitor.get_performance_stats()
        │ Returns p95, p99 latency
        ├─ Avg/max similarity
        └─ Performance over time
        │
    ┌───┴───┐
    │       │
    ▼       ▼
Threshold Drift? 
Met         Exceeded
 │            │
 │            ▼
 │     ModelMonitor.detect_performance_drift()
 │            │
 │            ├─ Drift detected = True
 │            │
 │            ▼
 │     Alert/Log warning
 │
 └─────→ Continue serving
```

---

## 3. Integration Points Between Components

### A. Feature Extractor ↔ Recommendation Engine

```python
# Feature Extractor produces embeddings
embeddings = feature_extractor.extract_image_features(images)
# Shape: (batch_size, embedding_dim)

# Recommendation Engine consumes embeddings
similar_items = recommendation_engine.search_similar(
    query_embedding=embeddings[0],  # Single query
    k=5
)
```

### B. Feature Extractor ↔ Feature Store

```python
# Extract features
embeddings = feature_extractor.extract_image_features(images)

# Store features
feature_store.write_features(
    feature_name="clip-v1-raw",
    features=embeddings,
    metadata={"model": "clip-vit-b-32"}
)

# Retrieve and transform
raw_features = feature_store.read_features("clip-v1-raw")
normalized = FeatureEngineer.normalize_features(raw_features, method="l2")
reduced = FeatureEngineer.dimensionality_reduction(normalized, n_components=256)
```

### C. Data Versioning ↔ Training Pipeline

```python
# Version data
data_manager.register_dataset(
    dataset_id="fashion-v1",
    data_paths={"train": "data/train", ...}
)

# Create splits
splits = data_manager.create_dataset_splits(
    source_dir=Path("data/raw"),
    output_dir=Path("data/splits")
)

# Use splits in training
train_loader = DataLoader(
    dataset=ImageDataset(root=Path("data/splits/train")),
    batch_size=32
)
```

### D. Batch Prediction ↔ Experiment Tracking ↔ MLflow

```python
from ml_models.batch_prediction import ExperimentTracker

tracker = ExperimentTracker()

# Start experiment (logs to MLflow)
exp_id = tracker.start_experiment(
    experiment_name="clip-v1-eval",
    description="Evaluate CLIP on test set"
)

# Make predictions
predictions = batch_pipeline.predict_batch(
    model=model,
    data=test_data
)

# Evaluate
metrics = evaluator.evaluate_retrieval(embeddings_db, query_embeddings)

# Log metrics (goes to MLflow)
tracker.log_experiment_metrics(exp_id, metrics)

# Access in MLflow UI
# http://localhost:5000/
```

### E. Model Registry ↔ Feature Extractor

```python
from ml_models.model_registry import get_model_registry

# Registry downloads model on first call
registry = get_model_registry()
model_path = registry.download_pretrained("clip-vit-b-32")

# Feature Extractor uses model
feature_extractor = CLIPFeatureExtractor(model_path)
embeddings = feature_extractor.extract_image_features(images)
```

### F. Evaluation ↔ Monitoring

```python
# Monitoring logs individual inferences
for inference in inferences:
    monitor.log_inference(
        query_id=inference['id'],
        inference_time_ms=inference['latency'],
        results_count=5,
        top_similarity=inference['score']
    )

# Evaluation aggregates statistics
stats = monitor.get_performance_stats()
# {"p95_inference_time_ms": 45.2, "avg_top_similarity": 0.87, ...}

# Check for drift
drift = monitor.detect_performance_drift(
    baseline_stats={"avg_inference_time_ms": 40.0},
    threshold=0.2
)
```

---

## 4. Configuration & State Management

### Singleton Instances

```python
# ml_models/model_registry.py
def get_model_registry() -> ModelRegistry:
    """Get or create singleton instance"""
    return ModelRegistry.get_instance()

# ml_models/data_versioning.py
def get_data_version_manager() -> DataVersionManager:
    """Get or create singleton instance"""
    return DataVersionManager.get_instance()

# ml_models/feature_store.py
def get_feature_store() -> FeatureStore:
    """Get or create singleton instance"""
    return FeatureStore.get_instance()
```

**Purpose**: Prevent resource duplication, ensure consistent state

### Configuration

**Location**: `config/settings.py`

```python
# Model paths
MODEL_CACHE_DIR = "ml_models/pretrained"
INDEX_DIR = "ml_models/indexes"
FEATURES_DIR = "ml_models/features"

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "StyleSync"

# Paths
DATA_DIR = "data"
EXPERIMENTS_DIR = "experiments"
```

---

## 5. Life Cycle of a Prediction Request

### Detailed Sequence

```
Sequence: User uploads image and searches for similar items

1. USER INTERACTION
   ├─ User opens Streamlit app (http://localhost:8501)
   ├─ User uploads image via drag-and-drop
   └─ User clicks "Search Similar"

2. API REQUEST
   ├─ Streamlit sends: POST /api/recommender/search
   │  Payload: {"image": <binary>, "top_k": 5}
   └─ FastAPI endpoint receives request

3. IMAGE PROCESSING
   ├─ Image validation (size, format)
   ├─ Image preprocessing (resize, normalize)
   └─ Store temporary image

4. FEATURE EXTRACTION
   ├─ Call: feature_extractor.extract_image_features()
   ├─ Model type: CLIP ViT-B/32 (from model_registry)
   ├─ Device: GPU if available, CPU otherwise
   ├─ Output: 512-dimensional embedding
   └─ Embedding stored temporarily

5. SIMILARITY SEARCH
   ├─ Call: recommendation_engine.search_similar()
   ├─ Input: Query embedding (512-dim)
   ├─ FAISS index search: K-NN search for top_k=5
   ├─ Output: List of (item_id, similarity_score)
   └─ Retrieve item metadata (name, description, image)

6. INFERENCE MONITORING
   ├─ Start timer
   ├─ Monitor logs inference metrics:
   │  ├─ query_id: unique request ID
   │  ├─ inference_time_ms: elapsed time
   │  ├─ results_count: number of results
   │  └─ top_similarity: max similarity score
   └─ End timer

7. RESPONSE PREPARATION
   ├─ Format results as JSON
   ├─ Include metadata (search_time, model_used)
   └─ Send to Streamlit

8. FRONTEND DISPLAY
   ├─ Streamlit receives JSON response
   ├─ Display query image
   ├─ Display results as image grid
   ├─ Show similarity scores
   └─ Show search time

9. BACKGROUND: MONITORING
   ├─ Monitor aggregates statistics
   ├─ Calculates percentiles (p95, p99)
   ├─ Detects performance drift
   └─ Logs to monitoring system

10. OPTIONAL: ANALYSIS
    ├─ Access performance stats: monitor.get_performance_stats()
    ├─ Check inference histories
    ├─ Analyze drift trends
    └─ Generate reports
```

---

## 6. Workflow: Adding New Model

```
Step 1: Train External Model
   └─ Custom training script
      └─ Output: model.pt

Step 2: Register with Registry
   ├─ registry.register_model(
   │  model_id="custom-v1",
   │  model_path="path/to/model.pt",
   │  status=ModelStatus.EXPERIMENTAL
   │)
   └─ Model metadata stored in registry

Step 3: Evaluate Model
   ├─ batch_pipeline.predict_batch(model)
   ├─ evaluator.evaluate_retrieval()
   ├─ tracker.start_experiment()
   └─ tracker.log_metrics()

Step 4: Compare with Baseline
   ├─ tracker.compare_experiments(
   │  experiment_ids=[exp1, exp2],
   │  metric_keys=["recall@5", "ndcg@5"]
   │)
   └─ Metrics compared side-by-side

Step 5: Promote to Staging
   ├─ registry.promote_model(
   │  "custom-v1",
   │  ModelStatus.STAGING
   │)
   └─ Model ready for testing

Step 6: Promote to Production
   ├─ registry.promote_model(
   │  "custom-v1",
   │  ModelStatus.PRODUCTION
   │)
   └─ Feature Extractor uses new model

Step 7: Monitor Performance
   ├─ monitor.get_performance_stats()
   ├─ Compare with baseline metrics
   └─ detect_performance_drift()
```

---

## 7. Key Integration Patterns

### Pattern 1: Singleton Lazy Loading

```python
# Module initialization
_instance = None

def get_instance() -> 'ModelRegistry':
    global _instance
    if _instance is None:
        _instance = ModelRegistry()
    return _instance
```

**Benefit**: Load expensive resources only when needed

### Pattern 2: Metadata Tracking

```python
# Every component stores metadata
metadata = {
    "created_at": datetime.utcnow().isoformat(),
    "version": "1.0.0",
    "status": ModelStatus.STAGING,
    "file_hash": sha256_hash,
    "framework": "pytorch"
}
```

**Benefit**: Full auditability and reproducibility

### Pattern 3: Metric Aggregation

```python
# Individual operations log metrics
monitor.log_inference(...)
monitor.log_inference(...)
monitor.log_inference(...)

# Periodic aggregation
stats = monitor.get_performance_stats(window_size=100)
# Calculates: mean, p95, p99, max, min
```

**Benefit**: Real-time performance tracking with statistical rigor

### Pattern 4: Feature Caching

```python
# Check cache first
if store.feature_exists("clip-v1-normalized"):
    features = store.read_features("clip-v1-normalized")
else:
    # Compute and cache
    raw = extract_features()
    features = normalize(raw)
    store.write_features("clip-v1-normalized", features)
```

**Benefit**: Avoid recomputation, speed up iterations

---

## 8. Extension Points

### Add New Feature Extractor
```python
# ml_models/recommender/feature_extractor.py
class CustomExtractor:
    def __init__(self, model_path):
        self.model = load_custom_model(model_path)
    
    def extract_image_features(self, images):
        return self.model(images)
```

### Add New Evaluation Metric
```python
# ml_models/evaluation.py
def evaluate_custom_metric(predictions, ground_truth):
    # Custom logic here
    return metric_value
```

### Add New Monitoring Check
```python
# ml_models/evaluation.py
def check_custom_condition(stats):
    if stats['custom_metric'] > threshold:
        logger.warning("Custom alert triggered!")
```

---

## Summary

The MLOps architecture provides:

✅ **Modular Design**: Each component has clear responsibility
✅ **Integration Points**: Well-defined interfaces between components
✅ **Data Flow**: Clear paths from input to output
✅ **State Management**: Singletons prevent resource duplication
✅ **Monitoring**: Built-in performance tracking
✅ **Extensibility**: Easy to add new components
✅ **Production Ready**: Monitoring, versioning, and rollback support

All components work together seamlessly while remaining independently testable and deployable.
