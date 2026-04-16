"""
Recommendation engine using FAISS vector database.
"""
import numpy as np
import faiss
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
import json
from .feature_extractor import get_feature_extractor
from PIL import Image

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
        self.metadata_path = str(Path(faiss_index_path).with_suffix('.json'))
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(feature_dim)
        self.metadata = []
        self.feature_extractor = None
        
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
            
            # Add to FAISS index
            self.index.add(features)
            
            # Store metadata
            item_metadata = {
                "garment_id": garment_id,
                "metadata": metadata or {}
            }
            self.metadata.append(item_metadata)
            
            logger.info(f"Added garment {garment_id} to index. Total items: {self.index.ntotal}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding garment: {e}")
            return False
    
    def search_similar(self, garment_image: Image.Image, k: int = 5) -> List[Tuple[str, float]]:
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
            
            # Search in FAISS
            distances, indices = self.index.search(query_features, min(k, self.index.ntotal))
            
            # Format results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata):
                    garment_id = self.metadata[idx]["garment_id"]
                    # Convert L2 distance to similarity score (0-1)
                    similarity = 1 / (1 + dist)
                    results.append((garment_id, float(similarity)))
            
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
            self.index = faiss.read_index(load_path)
            
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
        self.index = faiss.IndexFlatL2(self.feature_dim)
        self.metadata = []
        logger.info("Index reset")
