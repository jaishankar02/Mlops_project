"""Recommendation engine using FAISS with lightweight reranking."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from PIL import Image

from .feature_extractor import get_feature_extractor

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Efficient recommendation system using FAISS."""
    
    def __init__(self, feature_dim: int = 512, faiss_index_path: str = None):
        """
        Initialize recommendation engine.
        
        Args:
            feature_dim: Dimension of feature vectors
            faiss_index_path: Path to save/load FAISS index
        """
        self.feature_dim = feature_dim
        self.faiss_index_path = faiss_index_path or "faiss_index.bin"
        self.metadata_path = str(Path(self.faiss_index_path).with_suffix('.json'))
        
        # Initialize FAISS index (cosine via inner product on L2-normalized vectors).
        self.index = faiss.IndexFlatIP(feature_dim)
        self.metadata = []
        self.feature_extractor = None

    @staticmethod
    def _clip_similarity_to_ui_range(similarity: float) -> float:
        """Convert cosine similarity from [-1, 1] to [0, 1]."""
        score = (float(similarity) + 1.0) / 2.0
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _dominant_color_rgb(image: Image.Image) -> List[float]:
        """Estimate dominant color for a cheap visual reranking feature."""
        img = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        return [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])]

    @staticmethod
    def _color_similarity(query_rgb: List[float], candidate_rgb: Optional[List[float]]) -> float:
        """Convert RGB distance into [0, 1] similarity."""
        if not candidate_rgb or len(candidate_rgb) != 3:
            return 0.5
        q = np.asarray(query_rgb, dtype=np.float32)
        c = np.asarray(candidate_rgb, dtype=np.float32)
        dist = float(np.linalg.norm(q - c))
        max_dist = np.sqrt(3.0)
        return float(max(0.0, min(1.0, 1.0 - (dist / max_dist))))

    @staticmethod
    def _dedupe_key(garment_id: str, metadata: Dict) -> str:
        """Build a stable key to avoid returning duplicate visual assets."""
        image_path = str(metadata.get("image_path", "")).strip()
        if image_path:
            return Path(image_path).stem.lower()
        filename = str(metadata.get("filename", "")).strip()
        if filename:
            return Path(filename).stem.lower()
        return garment_id.lower()
        
    def initialize_feature_extractor(self, model_name: str = "clip"):
        """Initialize feature extractor."""
        try:
            self.feature_extractor = get_feature_extractor(model_name)
            logger.info(f"Feature extractor initialized with {model_name}")
        except Exception as e:
            logger.error(f"Error initializing feature extractor: {e}")
            raise
    
    def add_garment(self, garment_id: str, garment_image: Image.Image, 
                   metadata: Dict = None) -> bool:
        """
        Add a garment to the recommendation database.
        
        Args:
            garment_id: Unique identifier for garment
            garment_image: PIL Image of garment
            metadata: Additional metadata (category, size, color, price, etc.)
            
        Returns:
            Success flag
        """
        try:
            if self.feature_extractor is None:
                self.initialize_feature_extractor()
            
            # Extract features
            features = self.feature_extractor.extract_image_features(garment_image)
            features = features.astype(np.float32)
            faiss.normalize_L2(features)
            
            # Add to FAISS index
            self.index.add(features)
            
            # Store metadata
            item_metadata = {
                "garment_id": garment_id,
                "metadata": metadata or {},
            }
            item_metadata["metadata"]["dominant_color_rgb"] = self._dominant_color_rgb(garment_image)
            self.metadata.append(item_metadata)
            
            logger.info(f"Added garment {garment_id} to index. Total items: {self.index.ntotal}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding garment: {e}")
            return False
    
    def search_similar(
        self,
        garment_image: Image.Image,
        k: int = 5,
        min_similarity: float = 0.0,
        overfetch_factor: int = 8,
        clip_weight: float = 0.85,
    ) -> List[Dict[str, object]]:
        """
        Search for similar garments.
        
        Args:
            garment_image: PIL Image to search for
            k: Number of similar items to retrieve
            
        Returns:
            List of (garment_id, similarity_score) tuples
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("Index is empty, returning empty results")
                return []
            
            if self.feature_extractor is None:
                self.initialize_feature_extractor()
            
            # Extract features from query image
            query_features = self.feature_extractor.extract_image_features(garment_image)
            query_features = query_features.astype(np.float32)
            faiss.normalize_L2(query_features)
            
            candidate_k = min(max(k * max(1, overfetch_factor), k), self.index.ntotal)
            similarities, indices = self.index.search(query_features, candidate_k)

            query_color = self._dominant_color_rgb(garment_image)
            w_clip = float(max(0.0, min(1.0, clip_weight)))
            w_color = 1.0 - w_clip

            reranked = []
            for sim, idx in zip(similarities[0], indices[0]):
                if 0 <= int(idx) < len(self.metadata):
                    garment_entry = self.metadata[int(idx)]
                    garment_id = garment_entry["garment_id"]
                    md = garment_entry.get("metadata") or {}

                    clip_score = self._clip_similarity_to_ui_range(float(sim))
                    color_score = self._color_similarity(query_color, md.get("dominant_color_rgb"))
                    blended = (w_clip * clip_score) + (w_color * color_score)

                    reranked.append(
                        {
                            "faiss_index": int(idx),
                            "garment_id": garment_id,
                            "clip_score": float(clip_score),
                            "color_score": float(color_score),
                            "similarity_score": float(blended),
                            "dedupe_key": self._dedupe_key(garment_id, md),
                        }
                    )

            reranked.sort(key=lambda item: item["similarity_score"], reverse=True)

            # Keep diverse results by suppressing duplicate assets.
            results = []
            seen_keys = set()
            threshold = float(max(0.0, min(1.0, min_similarity)))
            for item in reranked:
                if item["similarity_score"] < threshold:
                    continue
                if item["dedupe_key"] in seen_keys:
                    continue
                seen_keys.add(item["dedupe_key"])
                item.pop("dedupe_key", None)
                results.append(item)
                if len(results) >= k:
                    break

            logger.info(f"Found {len(results)} similar items")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar items: {e}")
            return []
    
    def save_index(self, save_path: str = None) -> bool:
        """Save FAISS index and metadata."""
        try:
            save_path = save_path or self.faiss_index_path
            
            # Save FAISS index
            faiss.write_index(self.index, save_path)
            
            # Save metadata
            metadata_path = str(Path(save_path).with_suffix('.json'))
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            
            logger.info(f"Index saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load_index(self, load_path: str = None) -> bool:
        """Load FAISS index and metadata."""
        try:
            load_path = load_path or self.faiss_index_path
            
            if not Path(load_path).exists():
                logger.warning(f"Index file not found at {load_path}")
                return False
            
            # Load FAISS index
            loaded_index = faiss.read_index(load_path)
            metric_type = getattr(loaded_index, "metric_type", None)
            if metric_type != faiss.METRIC_INNER_PRODUCT:
                logger.warning(
                    "Existing index metric type %s is legacy/non-cosine; rebuild required",
                    metric_type,
                )
                return False

            self.index = loaded_index
            
            # Load metadata
            metadata_path = str(Path(load_path).with_suffix('.json'))
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"Index loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def reset_index(self):
        """Reset the recommendation engine."""
        self.index = faiss.IndexFlatIP(self.feature_dim)
        self.metadata = []
        logger.info("Index reset")
