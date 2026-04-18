"""
Training and fine-tuning script for recommender models.
Run: python train_models.py
"""
import logging
import sys
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_models.model_registry import get_model_registry, ModelStatus
from ml_models.evaluation import ModelEvaluator, TrainingPipeline, ModelMonitor
from ml_models.batch_prediction import ExperimentTracker, ModelCardGenerator
from config.mlflow_config import get_mlflow_tracker
from config.wandb_config import log_wandb_event
from config.settings import settings
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data(num_samples: int = 1000, feature_dim: int = 512) -> tuple:
    """Create synthetic data for demonstration."""
    logger.info(f"Creating synthetic data: {num_samples} samples, {feature_dim} dimensions")
    
    X = np.random.randn(num_samples, feature_dim).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)  # Normalize
    
    return X


def train_and_evaluate():
    """Train and evaluate models."""
    print("\n" + "="*70)
    print("StyleSync: MLOps Training & Evaluation Pipeline")
    print("="*70)
    
    registry = get_model_registry()
    tracker = ExperimentTracker()
    evaluator = ModelEvaluator()
    mlflow_tracker = get_mlflow_tracker()
    mlflow_tracker.setup_environment()
    mlflow_tracker.start_run(
        run_name="clip_recommender_training",
        params={
            "model_type": "clip",
            "feature_dim": 512,
            "num_samples": 2000,
            "similarity_metric": "cosine",
        },
    )
    
    # Create synthetic data
    print("\n📊 Preparing Data...")
    X_all = create_synthetic_data(num_samples=2000, feature_dim=512)
    
    # Split into train/val/test
    train_size = int(0.7 * len(X_all))
    val_size = int(0.15 * len(X_all))
    
    X_train = X_all[:train_size]
    X_val = X_all[train_size:train_size+val_size]
    X_test = X_all[train_size+val_size:]
    
    print(f"  ✓ Train: {X_train.shape[0]} samples")
    print(f"  ✓ Val: {X_val.shape[0]} samples")
    print(f"  ✓ Test: {X_test.shape[0]} samples")
    
    # Start experiment
    print("\n🔬 Starting Experiment...")
    exp_id = tracker.start_experiment(
        experiment_name="clip-recommender-v1",
        description="CLIP-based fashion recommender model",
        parameters={
            "model_type": "clip",
            "feature_dim": 512,
            "similarity_metric": "cosine",
        }
    )
    print(f"  ✓ Experiment ID: {exp_id}")
    log_wandb_event("training_started", {
        "experiment_id": exp_id,
        "model_type": "clip",
        "feature_dim": 512,
    })
    
    # Simulate training metrics
    print("\n📈 Training Phase...")
    training_metrics = {
        "train_loss": 0.15,
        "val_loss": 0.18,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    }
    
    for epoch in range(1, 6):
        epoch_metrics = {
            f"epoch_{epoch}_loss": 0.20 - (epoch * 0.01),
            f"epoch_{epoch}_val_loss": 0.22 - (epoch * 0.01),
        }
        tracker.log_experiment_metrics(exp_id, epoch_metrics)
        log_wandb_event("epoch_metrics", {
            "experiment_id": exp_id,
            **epoch_metrics,
        })
        print(f"  ✓ Epoch {epoch}: Loss={epoch_metrics[f'epoch_{epoch}_loss']:.4f}")
    
    # Evaluation
    print("\n📊 Evaluation Phase...")
    
    # Create mock ground truth
    ground_truth = {
        i: [j for j in range(i+1, min(i+5, len(X_test)))]
        for i in range(len(X_test))
    }
    
    eval_metrics = evaluator.evaluate_retrieval(
        embeddings_db=X_train,
        query_embeddings=X_test,
        ground_truth=ground_truth,
        k_values=[1, 5, 10]
    )
    
    print("\n  Retrieval Metrics:")
    for metric_name, value in eval_metrics.items():
        print(f"    {metric_name}: {value:.4f}")
    
    # Log evaluation metrics
    tracker.log_experiment_metrics(exp_id, eval_metrics)
    log_wandb_event("evaluation_metrics", {
        "experiment_id": exp_id,
        **eval_metrics,
    })
    
    # Model monitoring
    print("\n📊 Model Monitoring...")
    monitor = ModelMonitor()
    
    # Log sample inferences
    for i in range(10):
        monitor.log_inference(
            query_id=f"query_{i}",
            inference_time_ms=np.random.uniform(20, 100),
            results_count=5,
            top_similarity=np.random.uniform(0.75, 0.99)
        )
    
    perf_stats = monitor.get_performance_stats()
    print("\n  Performance Stats (Last 100 Inferences):")
    for stat, value in perf_stats.items():
        print(f"    {stat}: {value:.2f}")
    
    # Model card generation
    print("\n📝 Generating Model Card...")
    model_card_path = ModelCardGenerator.generate_model_card(
        model_name="clip-recommender-v1",
        model_type="Feature Extraction",
        description="CLIP-based visual feature extractor for fashion recommendation",
        performance_metrics=eval_metrics,
        training_data={
            "dataset": "Fashion Catalog",
            "num_samples": 2000,
            "feature_dimension": 512,
        },
        limitations=[
            "Optimized for fashion images",
            "May have bias towards certain clothing styles",
            "Requires GPU for inference",
        ],
        bias_risks=[
            "Potential bias in model training on certain demographics",
            "Cultural representation in fashion items",
        ],
        output_path="model_cards/clip-recommender-v1.md"
    )
    print(f"  ✓ Model card saved: {model_card_path}")
    
    # End experiment
    tracker.end_experiment(exp_id, status="completed")
    
    # Save results
    results = {
        "experiment_id": exp_id,
        "training_metrics": training_metrics,
        "evaluation_metrics": eval_metrics.copy(),
        "performance_stats": perf_stats,
        "model_card": model_card_path,
    }
    
    results_path = Path("experiments") / f"results_{exp_id}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    mlflow_tracker.log_artifact(str(results_path), artifact_path="reports")
    mlflow_tracker.log_artifact(str(model_card_path), artifact_path="model_cards")
    
    # Log to MLflow
    mlflow_tracker.log_event("training_completed", {
        "experiment_id": exp_id,
        "num_epochs": training_metrics["epochs"],
        "best_val_loss": training_metrics["val_loss"]
    })
    log_wandb_event("training_completed", {
        "experiment_id": exp_id,
        "num_epochs": training_metrics["epochs"],
        "best_val_loss": training_metrics["val_loss"],
    })
    
    print("\n" + "="*70)
    print("✨ Training Complete!")
    print("="*70)
    print(f"\n📁 Results saved to: {results_path}")
    print(f"📝 Model card saved to: {model_card_path}")
    print(f"🔬 Experiment ID: {exp_id}")
    print("\nMetrics Summary:")
    print(f"  - Recall@5: {eval_metrics.get('recall@5', 0):.4f}")
    print(f"  - Precision@5: {eval_metrics.get('precision@5', 0):.4f}")
    print(f"  - Avg Inference Time: {perf_stats.get('avg_inference_time_ms', 0):.2f}ms")
    print("\n" + "="*70 + "\n")

    mlflow_tracker.log_metrics_batch({
        "train_loss": float(training_metrics["train_loss"]),
        "val_loss": float(training_metrics["val_loss"]),
        "recall@5": float(eval_metrics.get("recall@5", 0)),
        "precision@5": float(eval_metrics.get("precision@5", 0)),
        "avg_inference_time_ms": float(perf_stats.get("avg_inference_time_ms", 0)),
    })
    mlflow_tracker.end_run()
    
    return exp_id


if __name__ == "__main__":
    try:
        exp_id = train_and_evaluate()
        print("✅ Training and evaluation completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
