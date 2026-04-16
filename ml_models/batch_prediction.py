"""
Batch prediction and inference pipeline.
Handles large-scale predictions and experiment management.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class BatchPredictionPipeline:
    """
    Batch prediction pipeline for processing large datasets efficiently.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch prediction pipeline.
        
        Args:
            batch_size: Number of samples per batch
        """
        self.batch_size = batch_size
        self.predictions = []
        self.metadata = {}
    
    def predict_batch(
        self,
        model,
        data: List,
        model_type: str = "feature_extraction",
        device: str = "cuda"
    ) -> np.ndarray:
        """
        Run predictions on batch of data.
        
        Args:
            model: Model for prediction
            data: List of input data
            model_type: Type of model (feature_extraction, classification)
            device: Device to run on
            
        Returns:
            Predictions array
        """
        try:
            import torch
            from tqdm import tqdm
            
            all_predictions = []
            
            # Process in batches
            num_batches = (len(data) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in tqdm(range(num_batches), desc="Predicting batches"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(data))
                batch_data = data[start_idx:end_idx]
                
                # Convert to tensor
                batch_tensor = torch.stack([torch.from_numpy(np.array(x)) for x in batch_data]).to(device)
                
                # Predict
                with torch.no_grad():
                    predictions = model(batch_tensor)
                    all_predictions.append(predictions.cpu().numpy())
            
            # Concatenate all predictions
            result = np.concatenate(all_predictions, axis=0)
            logger.info(f"Generated predictions for {len(data)} samples")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise
    
    def predict_from_files(
        self,
        model,
        file_paths: List[str],
        preprocessor=None,
        device: str = "cuda"
    ) -> Dict[str, np.ndarray]:
        """
        Run predictions on files (e.g., images).
        
        Args:
            model: Model for prediction
            file_paths: List of file paths
            preprocessor: Preprocessing function
            device: Device to run on
            
        Returns:
            Dictionary mapping file path to predictions
        """
        try:
            from PIL import Image
            from tqdm import tqdm
            
            results = {}
            
            for file_path in tqdm(file_paths, desc="Processing files"):
                try:
                    # Load and preprocess
                    image = Image.open(file_path)
                    if preprocessor:
                        image = preprocessor(image)
                    
                    # Predict
                    pred = self.predict_batch(model, [image], device=device)
                    results[file_path] = pred[0]
                    
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
            
            logger.info(f"Processed {len(results)} files successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in file prediction: {e}")
            return {}
    
    def save_predictions(self, output_path: str):
        """Save predictions to disk."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump({
                    "predictions": self.predictions,
                    "metadata": self.metadata,
                }, f)
            
            logger.info(f"Predictions saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")


class ExperimentTracker:
    """
    Track and compare multiple experiments.
    Integrates with MLflow for reproducibility.
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """Initialize experiment tracker."""
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
        self.experiments = {}
        self._load_experiments()
    
    def _load_experiments(self):
        """Load existing experiments."""
        experiment_files = self.experiments_dir.glob("experiment_*.json")
        for exp_file in experiment_files:
            with open(exp_file, 'r') as f:
                exp_data = json.load(f)
                self.experiments[exp_data["experiment_id"]] = exp_data
    
    def start_experiment(
        self,
        experiment_name: str,
        description: str = "",
        parameters: Dict[str, Any] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            experiment_name: Name of experiment
            description: Experiment description
            parameters: Experiment parameters
            
        Returns:
            Experiment ID
        """
        import uuid
        
        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
        
        experiment = {
            "experiment_id": experiment_id,
            "name": experiment_name,
            "description": description,
            "parameters": parameters or {},
            "start_time": datetime.utcnow().isoformat(),
            "metrics": {},
            "artifacts": [],
            "status": "running",
        }
        
        self.experiments[experiment_id] = experiment
        self._save_experiment(experiment_id)
        
        logger.info(f"Experiment started: {experiment_id}")
        return experiment_id
    
    def log_experiment_metrics(self, experiment_id: str, metrics: Dict[str, float]):
        """Log metrics for an experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return
        
        self.experiments[experiment_id]["metrics"].update(metrics)
        self._save_experiment(experiment_id)
        
        logger.info(f"Logged metrics for {experiment_id}")
    
    def log_artifact(self, experiment_id: str, artifact_path: str):
        """Log an artifact for an experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return
        
        self.experiments[experiment_id]["artifacts"].append({
            "path": artifact_path,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._save_experiment(experiment_id)
    
    def end_experiment(self, experiment_id: str, status: str = "completed"):
        """End an experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return
        
        self.experiments[experiment_id]["status"] = status
        self.experiments[experiment_id]["end_time"] = datetime.utcnow().isoformat()
        self._save_experiment(experiment_id)
        
        logger.info(f"Experiment ended: {experiment_id} - {status}")
    
    def _save_experiment(self, experiment_id: str):
        """Save experiment to disk."""
        exp_file = self.experiments_dir / f"experiment_{experiment_id}.json"
        with open(exp_file, 'w') as f:
            json.dump(self.experiments[experiment_id], f, indent=2)
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_keys: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_keys: Specific metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                
                comparison[exp_id] = {
                    "name": exp["name"],
                    "parameters": exp["parameters"],
                    "metrics": exp["metrics"],
                }
                
                # Filter metrics if specified
                if metric_keys:
                    comparison[exp_id]["metrics"] = {
                        k: v for k, v in exp["metrics"].items()
                        if k in metric_keys
                    }
        
        logger.info(f"Comparison created for {len(comparison)} experiments")
        return comparison
    
    def get_best_experiment(self, metric_name: str, mode: str = "max") -> Optional[str]:
        """
        Get best experiment based on metric.
        
        Args:
            metric_name: Metric to compare on
            mode: "max" or "min"
            
        Returns:
            Best experiment ID
        """
        best_exp_id = None
        best_value = -np.inf if mode == "max" else np.inf
        
        for exp_id, exp in self.experiments.items():
            if metric_name in exp["metrics"]:
                value = exp["metrics"][metric_name]
                
                if (mode == "max" and value > best_value) or \
                   (mode == "min" and value < best_value):
                    best_value = value
                    best_exp_id = exp_id
        
        return best_exp_id


class ModelCardGenerator:
    """Generate model cards for documentation."""
    
    @staticmethod
    def generate_model_card(
        model_name: str,
        model_type: str,
        description: str,
        performance_metrics: Dict[str, float] = None,
        training_data: Dict[str, Any] = None,
        limitations: List[str] = None,
        bias_risks: List[str] = None,
        output_path: str = "model_cards/model_card.md"
    ) -> str:
        """
        Generate a model card in Markdown format.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            description: Model description
            performance_metrics: Performance metrics
            training_data: Training data information
            limitations: Limitations of the model
            bias_risks: Potential bias and risks
            output_path: Output file path
            
        Returns:
            Path to generated model card
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            card = f"""# Model Card: {model_name}

## Overview
**Model Type**: {model_type}
**Date Created**: {datetime.utcnow().strftime('%Y-%m-%d')}
**Version**: 1.0

## Description
{description}

## Performance Metrics
"""
            
            if performance_metrics:
                card += "| Metric | Value |\n|--------|-------|\n"
                for metric, value in performance_metrics.items():
                    card += f"| {metric} | {value:.4f} |\n"
            else:
                card += "Not evaluated yet\n"
            
            card += "\n## Training Data\n"
            if training_data:
                for key, value in training_data.items():
                    card += f"- **{key}**: {value}\n"
            else:
                card += "No training data information\n"
            
            card += "\n## Limitations\n"
            if limitations:
                for limitation in limitations:
                    card += f"- {limitation}\n"
            else:
                card += "- Not specified\n"
            
            card += "\n## Bias and Risks\n"
            if bias_risks:
                for risk in bias_risks:
                    card += f"- {risk}\n"
            else:
                card += "- Not specified\n"
            
            card += "\n## Usage\n"
            card += f"""```python
from ml_models.model_registry import get_model_registry

registry = get_model_registry()
model_info = registry.get_model("{model_name}")
```

---
Generated automatically. Please update with additional information.
"""
            
            with open(output_path, 'w') as f:
                f.write(card)
            
            logger.info(f"Model card generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating model card: {e}")
            return ""
