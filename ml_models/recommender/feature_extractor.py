"""
Feature extraction using CLIP model for fashion recommendations.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class CLIPFeatureExtractor:
    """Extract features using OpenAI's CLIP model."""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize CLIP model for feature extraction."""
        try:
            import clip
            self.device = device
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            logger.info(f"CLIP model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Feature vector (numpy array)
        """
        try:
            with torch.no_grad():
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract features from text description.
        
        Args:
            text: Text description
            
        Returns:
            Feature vector (numpy array)
        """
        try:
            import clip
            with torch.no_grad():
                text_input = clip.tokenize(text).to(self.device)
                features = self.model.encode_text(text_input)
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            raise


class ResNetFeatureExtractor:
    """Alternative: Extract features using ResNet50 model."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize ResNet50 for feature extraction."""
        try:
            from torchvision.models import resnet50
            self.device = device
            self.model = resnet50(pretrained=True)
            # Remove classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(device)
            
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            logger.info(f"ResNet50 model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading ResNet50 model: {e}")
            raise
    
    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from an image using ResNet50.
        
        Args:
            image: PIL Image object
            
        Returns:
            Feature vector (numpy array, flattened)
        """
        try:
            with torch.no_grad():
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model(image_input)
                features = features.view(features.size(0), -1)  # Flatten
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize
                return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting ResNet features: {e}")
            raise


def get_feature_extractor(model_name: str = "clip", **kwargs):
    """Factory function to get appropriate feature extractor."""
    if model_name.lower() == "clip":
        return CLIPFeatureExtractor(**kwargs)
    elif model_name.lower() == "resnet":
        return ResNetFeatureExtractor(**kwargs)
    else:
        logger.warning(f"Unknown model {model_name}, defaulting to CLIP")
        return CLIPFeatureExtractor(**kwargs)
