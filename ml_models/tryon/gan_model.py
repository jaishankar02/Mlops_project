"""
GAN models for efficient virtual try-on.
Supports lightweight architectures optimized for T4 GPUs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class LightweightGANGenerator(nn.Module):
    """
    Lightweight GAN generator optimized for T4 GPUs.
    Simplified architecture for efficient try-on synthesis.
    """
    
    def __init__(self, in_channels: int = 6, out_channels: int = 3, base_channels: int = 64):
        """
        Initialize lightweight GAN generator.
        
        Args:
            in_channels: Number of input channels (person + garment)
            out_channels: Number of output channels (RGB)
            base_channels: Base number of channels
        """
        super(LightweightGANGenerator, self).__init__()
        self.base_channels = base_channels
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels, 7, 2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2, 5, 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4, 3, 2)
        
        # Residual blocks
        self.res1 = self._residual_block(base_channels * 4)
        self.res2 = self._residual_block(base_channels * 4)
        
        # Decoder with skip connections
        self.dec3 = self._deconv_block(base_channels * 4, base_channels * 2, 4, 2)
        self.dec2 = self._deconv_block(base_channels * 2 * 2, base_channels, 4, 2)  # *2 for skip concat
        self.dec1 = self._deconv_block(base_channels * 2, out_channels, 4, 2)  # *2 for skip concat
        
        self.output = nn.Tanh()
    
    def _conv_block(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        """Convolutional block with normalization and activation."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding=kernel//2),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _deconv_block(self, in_ch: int, out_ch: int, kernel: int = 4, stride: int = 2):
        """Transposed convolution block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _residual_block(self, channels: int):
        """Residual block with two convolutions."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        
        # Residual blocks
        res_out = self.res1(enc3_out)
        res_out = self.res2(res_out)
        
        # Decoder with skip connections
        dec3_out = self.dec3(res_out)
        dec2_out = self.dec2(torch.cat([dec3_out, enc2_out], dim=1))
        dec1_out = self.dec1(torch.cat([dec2_out, enc1_out], dim=1))
        
        output = self.output(dec1_out)
        return output


class GAN_TryOn:
    """GAN-based try-on model wrapper."""
    
    def __init__(self, use_fp16: bool = True, device: str = None):
        """
        Initialize GAN-based try-on.
        
        Args:
            use_fp16: Use FP16 precision for efficiency
            device: Device to run on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        self.generator = LightweightGANGenerator().to(self.device)
        self.generator.eval()
        
        if self.use_fp16:
            self.generator = self.generator.half()
        
        logger.info(f"GAN model initialized on {self.device} (FP16: {self.use_fp16})")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint."""
        try:
            if Path(checkpoint_path).exists():
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.generator.load_state_dict(state_dict)
                logger.info(f"Checkpoint loaded from {checkpoint_path}")
                return True
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def generate_tryon(self, person_image: Image.Image, garment_image: Image.Image) -> Image.Image:
        """
        Generate try-on image using GAN.
        
        Args:
            person_image: PIL Image of person
            garment_image: PIL Image of garment
            
        Returns:
            Generated try-on image
        """
        try:
            # Preprocess images
            person_tensor = self._preprocess_image(person_image)
            garment_tensor = self._preprocess_image(garment_image)
            
            # Concatenate
            input_tensor = torch.cat([person_tensor, garment_tensor], dim=1)
            
            # Generate try-on
            with torch.no_grad():
                if self.use_fp16:
                    input_tensor = input_tensor.half()
                
                output = self.generator(input_tensor)
                
                if self.use_fp16:
                    output = output.float()
            
            # Postprocess
            result_image = self._postprocess_image(output)
            
            logger.info("GAN try-on generation completed")
            return result_image
            
        except Exception as e:
            logger.error(f"Error generating GAN try-on: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image to tensor."""
        image = image.convert('RGB').resize((256, 256))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array * 2 - 1)  # Normalize to [-1, 1]
        
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor
    
    def _postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Postprocess tensor to PIL Image."""
        # Denormalize from [-1, 1] to [0, 1]
        output = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(output)


def get_tryon_model(use_gan: bool = False, use_fp16: bool = True) -> object:
    """
    Factory function to get try-on model.
    
    Args:
        use_gan: Use GAN model (if False, uses SimpleTransform)
        use_fp16: Use FP16 precision
        
    Returns:
        Try-on model instance
    """
    if use_gan:
        try:
            return GAN_TryOn(use_fp16=use_fp16)
        except Exception as e:
            logger.warning(f"Error loading GAN model, falling back to SimpleTransform: {e}")
            from .simple_transform import SimpleTransformTryOn
            return SimpleTransformTryOn()
    else:
        from .simple_transform import SimpleTransformTryOn
        return SimpleTransformTryOn()
