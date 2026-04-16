"""
MLflow configuration and experiment tracking.
Addresses professor feedback on experiment tracking.
"""
import mlflow
import logging
from config.settings import settings
from datetime import datetime
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracker for recommender system."""
    
    def __init__(self):
        """Initialize MLflow tracker."""
        self.setup_mlflow()
        self.metrics_buffer = {}
    
    def setup_mlflow(self):
        """Setup MLflow with configured backend."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            logger.info(f"MLflow configured with backend: {settings.MLFLOW_TRACKING_URI}")
            logger.info(f"Experiment: {settings.MLFLOW_EXPERIMENT_NAME}")
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
    
    def start_run(self, run_name: str, params: Optional[Dict[str, Any]] = None):
        """Start a new MLflow run."""
        try:
            mlflow.start_run(run_name=run_name)
            
            if params:
                mlflow.log_params(params)
            
            logger.info(f"MLflow run started: {run_name}")
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log a metric to MLflow."""
        try:
            mlflow.log_metric(metric_name, value, step=step)
        except Exception as e:
            logger.error(f"Error logging metric {metric_name}: {e}")
    
    def log_metrics_batch(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics batch: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact to MLflow."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.error(f"Error logging artifact {local_path}: {e}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log a model to MLflow."""
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log a custom event."""
        try:
            event_json = json.dumps({
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": event_data
            })
            mlflow.log_param(f"event_{event_type}", event_json[:500])  # MLflow has param size limit
            logger.debug(f"Event logged: {event_type}")
        except Exception as e:
            logger.error(f"Error logging event: {e}")


# Global MLflow tracker instance
mlflow_tracker = None


def get_mlflow_tracker() -> MLflowTracker:
    """Get or create MLflow tracker instance."""
    global mlflow_tracker
    if mlflow_tracker is None:
        mlflow_tracker = MLflowTracker()
    return mlflow_tracker


def log_recommendation_event(event_type: str, event_data: Dict[str, Any]):
    """
    Log recommendation system events to MLflow.
    
    Event types:
    - garment_upload: Upload a single garment
    - bulk_upload: Bulk upload garments
    - search_query: Search/recommendation query
    - index_reset: Index reset operation
    """
    try:
        tracker = get_mlflow_tracker()
        
        # Log as tags/metrics for better MLflow UI experience
        if event_type == "search_query":
            tracker.log_metrics_batch({
                "k_results": float(event_data.get("k_results", 0)),
                "results_count": float(event_data.get("results_count", 0)),
                "processing_time_ms": event_data.get("processing_time_ms", 0),
                "top_score": event_data.get("top_score", 0),
            })
        elif event_type in ["garment_upload", "bulk_upload"]:
            tracker.log_metrics_batch({
                "total_items": float(event_data.get("total_items", 0)),
                "successful": float(event_data.get("successful", 0)),
                "failed": float(event_data.get("failed", 0)),
            })
        
        logger.debug(f"Event {event_type} logged to MLflow")
    except Exception as e:
        logger.error(f"Error logging recommendation event: {e}")


def get_metrics() -> Dict[str, Any]:
    """Get current metrics for monitoring."""
    try:
        tracker = get_mlflow_tracker()
        return {
            "status": "active",
            "experiment": settings.MLFLOW_EXPERIMENT_NAME,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"status": "error", "message": str(e)}


# Initialize MLflow on module import
try:
    mlflow_tracker = MLflowTracker()
except Exception as e:
    logger.warning(f"MLflow initialization failed, using fallback: {e}")
