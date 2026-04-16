"""
Preprocessing utilities for images and data.
"""
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def preprocess_image_for_clip(image: Image.Image, target_size: int = 224) -> np.ndarray:
    """
    Preprocess image for CLIP model.
    
    Args:
        image: PIL Image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed numpy array
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize maintaining aspect ratio
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy
        img_array = np.array(image, dtype=np.uint8)
        
        # Pad to square if necessary
        if img_array.shape[0] != img_array.shape[1]:
            max_dim = max(img_array.shape[0], img_array.shape[1])
            padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            offset_h = (max_dim - img_array.shape[0]) // 2
            offset_w = (max_dim - img_array.shape[1]) // 2
            padded[offset_h:offset_h+img_array.shape[0], 
                   offset_w:offset_w+img_array.shape[1]] = img_array
            img_array = padded
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def batch_preprocess_images(image_paths: list, batch_size: int = 32):
    """
    Preprocess multiple images in batches.
    
    Args:
        image_paths: List of image file paths
        batch_size: Number of images per batch
        
    Yields:
        Batches of preprocessed images
    """
    batch = []
    for path in image_paths:
        try:
            image = Image.open(path)
            preprocessed = preprocess_image_for_clip(image)
            batch.append(preprocessed)
            
            if len(batch) == batch_size:
                yield np.stack(batch)
                batch = []
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
    
    if batch:
        yield np.stack(batch)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize feature vectors using L2 normalization.
    
    Args:
        features: Feature vectors (N, D)
        
    Returns:
        Normalized features
    """
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norm + 1e-8)
