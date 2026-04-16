"""
Simple transformation-based try-on fallback for resource-constrained scenarios.
Provides lightweight alternative to GAN-based try-on.
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SimpleTransformTryOn:
    """Lightweight try-on implementation using geometric transformations."""
    
    def __init__(self):
        """Initialize simple transform model."""
        logger.info("SimpleTransform try-on fallback initialized")
    
    def detect_person_region(self, person_image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect approximate person region using edge detection.
        Returns: (x, y, w, h)
        """
        try:
            gray = cv2.cvtColor(person_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                h, w = person_image.shape[:2]
                return 0, 0, w, h
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return x, y, w, h
        except Exception as e:
            logger.warning(f"Error detecting person region: {e}")
            h, w = person_image.shape[:2]
            return 0, 0, w, h
    
    def overlay_garment(self, person_image: np.ndarray, garment_image: np.ndarray) -> np.ndarray:
        """
        Overlay garment on person using simple transformations.
        
        Args:
            person_image: RGB numpy array of person
            garment_image: RGB numpy array of garment
            
        Returns:
            Composite try-on image
        """
        try:
            # Get person image dimensions
            person_h, person_w = person_image.shape[:2]
            
            # Detect person bounding box
            px, py, pw, ph = self.detect_person_region(person_image)
            
            # Resize garment to fit upper body (approximately 40% of person height)
            garment_target_h = int(ph * 0.6)
            garment_scale = garment_target_h / garment_image.shape[0]
            garment_w = int(garment_image.shape[1] * garment_scale)
            
            if garment_w <= 0 or garment_target_h <= 0:
                logger.warning("Invalid garment scaling")
                return person_image
            
            garment_resized = cv2.resize(garment_image, (garment_w, garment_target_h))
            
            # Calculate placement (centered on upper body)
            garment_x = px + (pw - garment_w) // 2
            garment_y = py + int(ph * 0.1)  # Position in upper body area
            
            # Create output image
            output = person_image.copy()
            
            # Blend garment (with transparency consideration)
            if garment_resized.shape[2] == 4:  # Has alpha channel
                alpha = garment_resized[:, :, 3] / 255.0
                for c in range(3):
                    output_region = output[garment_y:garment_y+garment_target_h, 
                                          garment_x:garment_x+garment_w, c]
                    garment_region = garment_resized[:, :, c]
                    
                    # Blend with alpha
                    blended = (output_region * (1 - alpha) + garment_region * alpha).astype(np.uint8)
                    output[garment_y:garment_y+garment_target_h, 
                           garment_x:garment_x+garment_w, c] = blended
            else:
                # Simple overlay without alpha
                output[garment_y:garment_y+garment_target_h, 
                       garment_x:garment_x+garment_w] = garment_resized
            
            logger.info(f"Garment overlayed successfully. Output shape: {output.shape}")
            return output
            
        except Exception as e:
            logger.error(f"Error overlaying garment: {e}")
            return person_image
    
    def apply_color_correction(self, composite: np.ndarray) -> np.ndarray:
        """Apply basic color correction to make overlay more natural."""
        try:
            # Convert to HSV for better color adjustment
            hsv = cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # Slightly reduce saturation globally for uniformity
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.95, 0, 255).astype(np.uint8)
            
            # Convert back to RGB
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return result
            
        except Exception as e:
            logger.warning(f"Error applying color correction: {e}")
            return composite
    
    def generate_tryon(self, person_image: Image.Image, garment_image: Image.Image) -> Image.Image:
        """
        Generate try-on image.
        
        Args:
            person_image: PIL Image of person
            garment_image: PIL Image of garment
            
        Returns:
            PIL Image with try-on result
        """
        try:
            # Convert to numpy arrays
            person_np = np.array(person_image.convert('RGB'))
            garment_np = np.array(garment_image.convert('RGBA'))  # Try to preserve alpha
            
            # Overlay garment
            output = self.overlay_garment(person_np, garment_np)
            
            # Apply color correction
            output = self.apply_color_correction(output)
            
            # Convert back to PIL
            result = Image.fromarray(output.astype(np.uint8))
            logger.info("Try-on generation completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating try-on: {e}")
            return person_image


class GradientBlendTryOn(SimpleTransformTryOn):
    """Enhanced try-on with gradient blending for smoother edges."""
    
    def blend_with_gradient_mask(self, person_image: np.ndarray, 
                                garment_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Blend garment with smooth gradient mask."""
        try:
            # Create gradient mask (soft edges)
            h, w = mask.shape[:2]
            gradient = np.linspace(1, 0, h // 5)
            gradient_mask = np.ones_like(mask, dtype=np.float32)
            gradient_mask[:h//5] = gradient[:, np.newaxis]
            
            # Apply smooth blending
            output = person_image.copy().astype(np.float32)
            garment_rgb = garment_image[:, :, :3] if garment_image.shape[2] > 3 else garment_image
            
            for c in range(3):
                output[:, :, c] = (person_image[:, :, c] * (1 - gradient_mask) + 
                                  garment_rgb[:, :, c] * gradient_mask)
            
            return output.astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in gradient blending: {e}")
            return person_image
