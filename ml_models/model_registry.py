"""
Model management and pretrained model downloading.
Handles model versioning, registry, and lifecycle management.
"""
import os
import logging
from pathlib import Path
from enum import Enum
import json
from typing import Dict, Any, Optional
import hashlib
import urllib.request

logger = logging.getLogger(__name__)

# Model registry with pretrained models
PRETRAINED_MODELS = {
    "clip-vit-b-32": {
        "name": "CLIP ViT-B/32",
        "description": "Vision Transformer model trained with contrastive learning",
        "url": "https://openaipublic.blob.core.windows.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58aaad624a/ViT-B-32.pt",
        "task": "feature_extraction",
        "framework": "pytorch",
        "input_size": 224,
        "feature_dim": 512,
        "size_mb": 352,
        "version": "1.0",
    },
    "clip-vit-l-14": {
        "name": "CLIP ViT-L/14",
        "description": "Larger Vision Transformer model for better accuracy",
        "url": "https://openaipublic.blob.core.windows.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d5d35de16210aac993749e2f2b92e45dd6a/ViT-L-14.pt",
        "task": "feature_extraction",
        "framework": "pytorch",
        "input_size": 224,
        "feature_dim": 768,
        "size_mb": 903,
        "version": "1.0",
    },
    "resnet50": {
        "name": "ResNet50",
        "description": "Residual Network with 50 layers for image classification",
        "url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "task": "feature_extraction",
        "framework": "pytorch",
        "input_size": 224,
        "feature_dim": 2048,
        "size_mb": 102,
        "version": "1.0",
    },
    "efficientnet-b0": {
        "name": "EfficientNet-B0",
        "description": "Efficient neural network for mobile and edge devices",
        "url": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
        "task": "feature_extraction",
        "framework": "pytorch",
        "input_size": 224,
        "feature_dim": 1280,
        "size_mb": 20,
        "version": "1.0",
    }
}


class ModelStatus(str, Enum):
    """Model status enumeration."""
    PRODUCTION = "production"
    STAGING = "staging"
    ARCHIVED = "archived"
    EXPERIMENTAL = "experimental"


class ModelRegistry:
    """
    MLOps Model Registry for tracking and managing models.
    Integrates with MLflow for versioning and deployment.
    """
    
    def __init__(self, registry_dir: str = "models", mlflow_tracking_uri: str = None):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store model metadata
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.registry_dir / "model_registry.json"
        self.models_dir = self.registry_dir / "pretrained"
        self.models_dir.mkdir(exist_ok=True)
        
        self.registry = self._load_registry()
        
        # MLflow integration
        self.mlflow_uri = mlflow_tracking_uri
        if mlflow_tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        logger.info(f"Model Registry initialized at {self.registry_dir}")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load existing registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
        logger.debug("Registry saved to disk")
    
    def register_model(
        self,
        model_id: str,
        model_path: str,
        task: str,
        framework: str,
        metrics: Dict[str, float] = None,
        status: ModelStatus = ModelStatus.EXPERIMENTAL,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Register a model in the registry.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to model file
            task: Task type (e.g., feature_extraction, classification)
            framework: Framework used (e.g., pytorch, tensorflow)
            metrics: Performance metrics
            status: Model status
            metadata: Additional metadata
            
        Returns:
            Success flag
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(model_path)
            
            self.registry[model_id] = {
                "model_id": model_id,
                "path": str(model_path),
                "task": task,
                "framework": framework,
                "status": status.value,
                "metrics": metrics or {},
                "file_hash": file_hash,
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "metadata": metadata or {},
                "registered_at": str(Path(self.metadata_file).stat().st_mtime),
            }
            
            self._save_registry()
            logger.info(f"Model {model_id} registered successfully")
            
            # Log to MLflow if available
            if self.mlflow_uri:
                self._log_to_mlflow(model_id, self.registry[model_id])
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata from registry."""
        return self.registry.get(model_id)
    
    def list_models(self, task: str = None, status: str = None) -> list:
        """List all registered models, optionally filtered."""
        models = list(self.registry.values())
        
        if task:
            models = [m for m in models if m.get("task") == task]
        if status:
            models = [m for m in models if m.get("status") == status]
        
        return models
    
    def promote_model(self, model_id: str, target_status: ModelStatus) -> bool:
        """Promote model to different status (e.g., STAGING -> PRODUCTION)."""
        if model_id not in self.registry:
            logger.error(f"Model {model_id} not found")
            return False
        
        old_status = self.registry[model_id]["status"]
        self.registry[model_id]["status"] = target_status.value
        self._save_registry()
        
        logger.info(f"Model {model_id} promoted: {old_status} -> {target_status.value}")
        return True
    
    def download_pretrained(self, model_name: str, force: bool = False) -> Optional[Path]:
        """
        Download pretrained model.
        
        Args:
            model_name: Name of pretrained model (e.g., clip-vit-b-32)
            force: Force download even if exists
            
        Returns:
            Path to downloaded model
        """
        if model_name not in PRETRAINED_MODELS:
            logger.error(f"Unknown pretrained model: {model_name}")
            return None
        
        model_info = PRETRAINED_MODELS[model_name]
        model_path = self.models_dir / f"{model_name}.pt"
        
        # Check if already downloaded
        if model_path.exists() and not force:
            logger.info(f"Model {model_name} already exists at {model_path}")
            return model_path
        
        # Download
        try:
            logger.info(f"Downloading {model_info['name']} ({model_info['size_mb']}MB)...")
            urllib.request.urlretrieve(model_info['url'], model_path)
            logger.info(f"Downloaded to {model_path}")
            
            # Register in registry
            self.register_model(
                model_id=model_name,
                model_path=str(model_path),
                task=model_info["task"],
                framework=model_info["framework"],
                metadata=model_info,
                status=ModelStatus.PRODUCTION
            )
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file hash for integrity checking."""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _log_to_mlflow(self, model_id: str, model_info: Dict[str, Any]):
        """Log model to MLflow registry."""
        try:
            import mlflow
            mlflow.start_run()
            mlflow.log_params({"model_id": model_id, "framework": model_info["framework"]})
            mlflow.log_metrics(model_info.get("metrics", {}))
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")


# Global registry instance
_registry = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def download_all_pretrained_models() -> Dict[str, Optional[Path]]:
    """Download all available pretrained models."""
    registry = get_model_registry()
    results = {}
    
    for model_name in PRETRAINED_MODELS.keys():
        logger.info(f"Downloading {model_name}...")
        path = registry.download_pretrained(model_name)
        results[model_name] = path
    
    return results
