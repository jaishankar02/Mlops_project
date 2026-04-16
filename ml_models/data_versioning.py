"""
Data versioning and management for MLOps.
Tracks data versioning, dataset splits, and data quality metrics.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetSplit(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DataVersionManager:
    """
    Data versioning for MLOps pipeline.
    Tracks dataset versions, schemas, and statistics.
    """
    
    def __init__(self, data_dir: str = "data", version_dir: str = "data/versions"):
        """
        Initialize data version manager.
        
        Args:
            data_dir: Main data directory
            version_dir: Directory to store version metadata
        """
        self.data_dir = Path(data_dir)
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.version_dir / "dataset_manifest.json"
        self.manifest = self._load_manifest()
        
        logger.info(f"DataVersionManager initialized at {self.version_dir}")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load existing manifest."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def register_dataset(
        self,
        dataset_id: str,
        description: str,
        data_paths: Dict[str, str],
        splits: Dict[str, float] = None,
        schema: Dict[str, str] = None,
        statistics: Dict[str, Any] = None
    ) -> bool:
        """
        Register a dataset version.
        
        Args:
            dataset_id: Unique dataset identifier
            description: Dataset description
            data_paths: Paths to data files (e.g., {"train": "path/to/train", "val": "path/to/val"})
            splits: Dataset splits (e.g., {"train": 0.7, "val": 0.15, "test": 0.15})
            schema: Data schema
            statistics: Dataset statistics
            
        Returns:
            Success flag
        """
        try:
            # Calculate data hashes
            data_hashes = {}
            for split, path in data_paths.items():
                if Path(path).exists():
                    data_hashes[split] = self._calculate_dir_hash(Path(path))
            
            version_hash = self._calculate_version_hash(data_hashes)
            
            self.manifest[dataset_id] = {
                "dataset_id": dataset_id,
                "description": description,
                "paths": data_paths,
                "splits": splits or {"train": 0.7, "val": 0.15, "test": 0.15},
                "schema": schema or {},
                "statistics": statistics or {},
                "data_hashes": data_hashes,
                "version_hash": version_hash,
                "registered_at": datetime.utcnow().isoformat(),
            }
            
            self._save_manifest()
            logger.info(f"Dataset {dataset_id} registered with version {version_hash[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error registering dataset: {e}")
            return False
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata."""
        return self.manifest.get(dataset_id)
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all registered datasets."""
        return list(self.manifest.values())
    
    def _calculate_dir_hash(self, directory: Path) -> str:
        """Calculate hash of directory contents."""
        file_hashes = []
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                file_hash = self._calculate_file_hash(file_path)
                file_hashes.append(file_hash)
        
        combined = "".join(file_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash."""
        hash_obj = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _calculate_version_hash(self, data_hashes: Dict[str, str]) -> str:
        """Calculate overall version hash."""
        combined = "".join(sorted(data_hashes.values()))
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def create_dataset_splits(
        self,
        source_dir: Path,
        output_dir: Path,
        splits: Dict[str, float] = None,
        random_seed: int = 42
    ) -> Dict[str, Path]:
        """
        Create train/val/test splits from dataset.
        
        Args:
            source_dir: Source data directory
            output_dir: Output directory for splits
            splits: Split ratios (e.g., {"train": 0.7, "val": 0.15, "test": 0.15})
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split paths
        """
        try:
            import os
            import random
            
            if splits is None:
                splits = {"train": 0.7, "val": 0.15, "test": 0.15}
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create split directories
            split_dirs = {}
            for split in splits.keys():
                split_dir = output_dir / split
                split_dir.mkdir(exist_ok=True)
                split_dirs[split] = split_dir
            
            # Get all files
            all_files = list(source_dir.glob("**/*"))
            all_files = [f for f in all_files if f.is_file()]
            
            # Shuffle with seed
            random.seed(random_seed)
            random.shuffle(all_files)
            
            # Calculate split indices
            total = len(all_files)
            train_idx = int(total * splits.get("train", 0.7))
            val_idx = train_idx + int(total * splits.get("val", 0.15))
            
            # Assign files to splits
            split_assignments = {
                "train": all_files[:train_idx],
                "val": all_files[train_idx:val_idx],
                "test": all_files[val_idx:],
            }
            
            # Create symlinks
            for split, files in split_assignments.items():
                for file in files:
                    dest = split_dirs[split] / file.name
                    if not dest.exists():
                        try:
                            os.symlink(file, dest)
                        except (OSError, NotImplementedError):
                            # Fallback to copy if symlink not supported
                            import shutil
                            shutil.copy2(file, dest)
            
            logger.info(f"Created dataset splits in {output_dir}")
            logger.info(f"  Train: {len(split_assignments['train'])} files")
            logger.info(f"  Val: {len(split_assignments['val'])} files")
            logger.info(f"  Test: {len(split_assignments['test'])} files")
            
            return split_dirs
            
        except Exception as e:
            logger.error(f"Error creating splits: {e}")
            return {}


class DataQualityMetrics:
    """Calculate data quality metrics."""
    
    @staticmethod
    def calculate_image_stats(image_dir: Path) -> Dict[str, Any]:
        """Calculate statistics for image dataset."""
        try:
            from PIL import Image
            import numpy as np
            
            stats = {
                "total_images": 0,
                "avg_width": 0,
                "avg_height": 0,
                "avg_file_size_mb": 0,
                "formats": {},
                "corrupted": [],
            }
            
            widths = []
            heights = []
            file_sizes = []
            
            for img_path in image_dir.glob("**/*"):
                if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif"]:
                    try:
                        img = Image.open(img_path)
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        
                        file_size = img_path.stat().st_size / (1024 * 1024)
                        file_sizes.append(file_size)
                        
                        fmt = img.format or "unknown"
                        stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1
                        
                        stats["total_images"] += 1
                        img.close()
                    except Exception as e:
                        stats["corrupted"].append(str(img_path))
                        logger.warning(f"Corrupted image: {img_path}")
            
            if widths:
                stats["avg_width"] = float(np.mean(widths))
                stats["avg_height"] = float(np.mean(heights))
                stats["avg_file_size_mb"] = float(np.mean(file_sizes))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating image stats: {e}")
            return {}


# Global data version manager instance
_data_manager = None


def get_data_version_manager() -> DataVersionManager:
    """Get or create global data version manager."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataVersionManager()
    return _data_manager
