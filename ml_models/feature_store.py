"""
Feature store for caching and serving features.
Enables efficient feature reuse across experiments.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature store for caching and managing features across experiments.
    """
    
    def __init__(self, store_dir: str = "data/feature_store"):
        """Initialize feature store."""
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.store_dir / "feature_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load feature metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save feature metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def write_features(
        self,
        feature_name: str,
        features: np.ndarray,
        metadata: Dict = None,
        overwrite: bool = False
    ) -> bool:
        """
        Write features to store.
        
        Args:
            feature_name: Name of feature set
            features: Feature array (N, D)
            metadata: Feature metadata
            overwrite: Overwrite existing features
            
        Returns:
            Success flag
        """
        try:
            feature_path = self.store_dir / f"{feature_name}.npy"
            
            if feature_path.exists() and not overwrite:
                logger.warning(f"Features {feature_name} already exist. Use overwrite=True to replace.")
                return False
            
            # Save features
            np.save(feature_path, features)
            
            # Update metadata
            self.metadata[feature_name] = {
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "path": str(feature_path),
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }
            
            self._save_metadata()
            logger.info(f"Features saved: {feature_name} {features.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing features: {e}")
            return False
    
    def read_features(self, feature_name: str) -> Optional[np.ndarray]:
        """Read features from store."""
        try:
            if feature_name not in self.metadata:
                logger.warning(f"Features {feature_name} not found in store")
                return None
            
            feature_path = self.metadata[feature_name]["path"]
            features = np.load(feature_path)
            
            logger.info(f"Features loaded: {feature_name} {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error reading features: {e}")
            return None
    
    def list_features(self) -> List[str]:
        """List all available features."""
        return list(self.metadata.keys())
    
    def get_feature_info(self, feature_name: str) -> Optional[Dict]:
        """Get metadata for a feature."""
        return self.metadata.get(feature_name)


class FeatureEngineer:
    """Feature engineering utilities."""
    
    @staticmethod
    def normalize_features(features: np.ndarray, method: str = "l2") -> np.ndarray:
        """
        Normalize features.
        
        Args:
            features: Input features (N, D)
            method: "l2" (default) or "minmax" or "standard"
            
        Returns:
            Normalized features
        """
        try:
            if method == "l2":
                norm = np.linalg.norm(features, axis=1, keepdims=True)
                return features / (norm + 1e-8)
            
            elif method == "minmax":
                min_val = np.min(features, axis=0)
                max_val = np.max(features, axis=0)
                return (features - min_val) / (max_val - min_val + 1e-8)
            
            elif method == "standard":
                mean = np.mean(features, axis=0)
                std = np.std(features, axis=0)
                return (features - mean) / (std + 1e-8)
            
            else:
                logger.warning(f"Unknown normalization method: {method}")
                return features
                
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features
    
    @staticmethod
    def dimensionality_reduction(
        features: np.ndarray,
        n_components: int = 512,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Reduce feature dimensionality.
        
        Args:
            features: Input features (N, D)
            n_components: Target number of components
            method: "pca" or "umap"
            
        Returns:
            Reduced features (N, n_components)
        """
        try:
            if method == "pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components)
                reduced = pca.fit_transform(features)
                logger.info(f"PCA: {features.shape} -> {reduced.shape}")
                return reduced
            
            elif method == "umap":
                import umap
                reducer = umap.UMAP(n_components=n_components)
                reduced = reducer.fit_transform(features)
                logger.info(f"UMAP: {features.shape} -> {reduced.shape}")
                return reduced
            
            else:
                logger.warning(f"Unknown reduction method: {method}")
                return features
                
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            return features
    
    @staticmethod
    def augment_features(features: np.ndarray, augmentation_type: str = "noise") -> np.ndarray:
        """
        Augment features for data augmentation.
        
        Args:
            features: Input features (N, D)
            augmentation_type: "noise", "dropout", or "mixup"
            
        Returns:
            Augmented features
        """
        try:
            if augmentation_type == "noise":
                noise = np.random.normal(0, 0.01, features.shape)
                return features + noise
            
            elif augmentation_type == "dropout":
                mask = np.random.binomial(1, 0.8, features.shape)
                return features * mask
            
            elif augmentation_type == "mixup":
                indices = np.random.permutation(len(features))
                alpha = 0.2
                mixed = alpha * features + (1 - alpha) * features[indices]
                return mixed
            
            else:
                logger.warning(f"Unknown augmentation type: {augmentation_type}")
                return features
                
        except Exception as e:
            logger.error(f"Error in feature augmentation: {e}")
            return features


# Global feature store instance
_feature_store = None


def get_feature_store() -> FeatureStore:
    """Get or create global feature store."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store
