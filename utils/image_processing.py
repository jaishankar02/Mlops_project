"""
Image processing utilities.
"""
from PIL import Image
import logging
from config.settings import settings
from typing import Tuple

logger = logging.getLogger(__name__)


def validate_image(image: Image.Image) -> bool:
    """
    Validate image format and size.
    
    Args:
        image: PIL Image object
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check format
        if image.format not in settings.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported image format: {image.format}")
            return False
        
        # Check file size (rough estimate based on image dimensions)
        width, height = image.size
        estimated_size_mb = (width * height * 3) / (1024 * 1024)  # Rough estimate
        
        if estimated_size_mb > settings.MAX_IMAGE_SIZE_MB:
            logger.warning(f"Image too large: {estimated_size_mb:.2f}MB")
            return False
        
        # Check minimum size
        if width < 100 or height < 100:
            logger.warning(f"Image too small: {width}x{height}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False


def optimize_image(image: Image.Image, max_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Optimize image for processing.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimensions
        
    Returns:
        Optimized PIL Image
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if necessary (preserve aspect ratio)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"Error optimizing image: {e}")
        return image


def crop_to_square(image: Image.Image) -> Image.Image:
    """
    Crop image to square (for consistent input to models).
    
    Args:
        image: PIL Image object
        
    Returns:
        Square PIL Image
    """
    try:
        width, height = image.size
        size = min(width, height)
        
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        return image.crop((left, top, right, bottom))
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return image


def save_image(image: Image.Image, path: str, quality: int = 85):
    """
    Save image with optimization.
    
    Args:
        image: PIL Image object
        path: Output path
        quality: JPEG quality (1-100)
    """
    try:
        image = image.convert('RGB')
        image.save(path, quality=quality, optimize=True)
        logger.debug(f"Image saved to {path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
