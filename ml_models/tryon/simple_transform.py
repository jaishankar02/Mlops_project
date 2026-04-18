"""
Simple transformation-based try-on fallback for resource-constrained scenarios.
Provides lightweight alternative to GAN-based try-on.
"""
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional

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


class DemoStableTryOn:
    """Landmark-aware, deterministic try-on model for reliable demos."""

    def __init__(self):
        self._pose = None
        self._selfie_seg = None
        try:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.4)
            self._selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            logger.info("DemoStableTryOn initialized with MediaPipe pose")
        except Exception as e:
            logger.warning(f"MediaPipe unavailable for DemoStableTryOn: {e}")

    def is_available(self) -> bool:
        return True

    def _garment_mask(self, garment_rgb: np.ndarray) -> np.ndarray:
        """Extract foreground garment mask from likely plain background image."""
        hsv = cv2.cvtColor(garment_rgb, cv2.COLOR_RGB2HSV)
        # Keep colorful/darker foreground; reject bright low-saturation background.
        fg = ((hsv[:, :, 1] > 18) | (hsv[:, :, 2] < 240)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(fg)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            fg = mask
        return fg

    def _pose_box_from_image(self, image_rgb: np.ndarray, width_scale: float, height_scale: float) -> Optional[Tuple[int, int, int, int]]:
        """Get torso box from pose landmarks for arbitrary image."""
        if self._pose is None:
            return None
        h, w = image_rgb.shape[:2]
        try:
            result = self._pose.process(image_rgb)
            if not result.pose_landmarks:
                return None
            lm = result.pose_landmarks.landmark
            ls, rs = lm[11], lm[12]
            lh, rh = lm[23], lm[24]

            xs = [ls.x * w, rs.x * w, lh.x * w, rh.x * w]
            ys = [ls.y * h, rs.y * h, lh.y * h, rh.y * h]

            x1, x2 = int(max(0, min(xs))), int(min(w - 1, max(xs)))
            y1, y2 = int(max(0, min(ys))), int(min(h - 1, max(ys)))

            torso_w = max(30, x2 - x1)
            torso_h = max(40, y2 - y1)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bw = int(torso_w * width_scale)
            bh = int(torso_h * height_scale)

            bx = max(0, cx - bw // 2)
            by = max(0, cy - bh // 2)
            bw = min(w - bx, bw)
            bh = min(h - by, bh)
            if bw < 20 or bh < 20:
                return None
            return bx, by, bw, bh
        except Exception:
            return None

    def _extract_garment_region(self, garment_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a usable garment patch from either product image or full-body person image."""
        # Step 1: if pose detected, crop around torso from garment source.
        box = self._pose_box_from_image(garment_rgb, width_scale=1.8, height_scale=1.9)
        crop = garment_rgb
        if box is not None:
            bx, by, bw, bh = box
            crop = garment_rgb[by:by + bh, bx:bx + bw].copy()

        # Step 2: if selfie segmentation is available, keep person foreground in crop.
        if self._selfie_seg is not None:
            try:
                seg = self._selfie_seg.process(crop)
                if seg.segmentation_mask is not None:
                    fg = (seg.segmentation_mask > 0.35).astype(np.uint8) * 255
                    kernel = np.ones((5, 5), np.uint8)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
                    if np.count_nonzero(fg) > 200:
                        return crop, fg
            except Exception:
                pass

        # Step 3: fallback to appearance-based mask.
        return crop, self._garment_mask(crop)

    def _estimate_torso_box(self, person_rgb: np.ndarray) -> Tuple[int, int, int, int]:
        """Estimate torso box via pose landmarks, fallback to centered heuristic."""
        h, w = person_rgb.shape[:2]
        pose_box = self._pose_box_from_image(person_rgb, width_scale=1.55, height_scale=1.65)
        if pose_box is not None:
            return pose_box
        try:
            return int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.55)
        except Exception as e:
            logger.warning(f"Pose torso estimation failed: {e}")
            return int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.55)

    def _match_region_brightness(self, src_rgb: np.ndarray, dst_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Match garment brightness to torso region for more natural blending."""
        src = src_rgb.astype(np.float32)
        dst = dst_rgb.astype(np.float32)
        a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

        src_gray = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dst_gray = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        wsum = np.sum(a) + 1e-6
        src_mean = float(np.sum(src_gray * a) / wsum)
        dst_mean = float(np.sum(dst_gray * a) / wsum)
        gain = np.clip((dst_mean + 1e-6) / (src_mean + 1e-6), 0.65, 1.35)
        src = np.clip(src * gain, 0, 255)
        return src.astype(np.uint8)

    def generate_tryon(self, person_image: Image.Image, garment_image: Image.Image) -> Image.Image:
        """Generate stable try-on by placing segmented garment over estimated torso."""
        try:
            person = np.array(person_image.convert("RGB"))
            garment = np.array(garment_image.convert("RGB"))
            h, w = person.shape[:2]

            bx, by, bw, bh = self._estimate_torso_box(person)
            if bw < 20 or bh < 20:
                return person_image

            garment_src, gmask = self._extract_garment_region(garment)
            ys, xs = np.where(gmask > 0)
            if len(xs) < 20 or len(ys) < 20:
                # Garment mask failed; keep person image unchanged instead of glitching.
                return Image.fromarray(person)

            gx1, gx2 = int(xs.min()), int(xs.max())
            gy1, gy2 = int(ys.min()), int(ys.max())
            garment_crop = garment_src[gy1:gy2 + 1, gx1:gx2 + 1]
            mask_crop = gmask[gy1:gy2 + 1, gx1:gx2 + 1]

            garment_rs = cv2.resize(garment_crop, (bw, bh), interpolation=cv2.INTER_LINEAR)
            mask_rs = cv2.resize(mask_crop, (bw, bh), interpolation=cv2.INTER_LINEAR)

            alpha = cv2.GaussianBlur(mask_rs.astype(np.float32) / 255.0, (11, 11), 0)
            alpha = np.clip(alpha * 0.92, 0.0, 0.92)

            roi = person[by:by + bh, bx:bx + bw].copy()
            garment_rs = self._match_region_brightness(garment_rs, roi, alpha)

            out_roi = (
                garment_rs.astype(np.float32) * alpha[..., None]
                + roi.astype(np.float32) * (1.0 - alpha[..., None])
            )
            person[by:by + bh, bx:bx + bw] = np.clip(out_roi, 0, 255).astype(np.uint8)

            return Image.fromarray(person)
        except Exception as e:
            logger.error(f"DemoStableTryOn failed: {e}")
            return person_image
