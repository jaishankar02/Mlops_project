"""
Unit tests for recommender system.
"""
import pytest
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new('RGB', (256, 256), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img


class TestRecommendationEngine:
    """Test recommendation engine."""
    
    def test_engine_initialization(self):
        """Test recommendation engine initialization."""
        from ml_models.recommender.recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine(feature_dim=512)
        assert engine.feature_dim == 512
        assert engine.index.ntotal == 0
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor."""
        pytest.skip("Requires CLIP model download")
        from ml_models.recommender.feature_extractor import CLIPFeatureExtractor
        
        extractor = CLIPFeatureExtractor()
        assert extractor is not None


class TestImageProcessing:
    """Test image processing utilities."""
    
    def test_image_validation(self, sample_image):
        """Test image validation."""
        from utils.image_processing import validate_image
        
        assert validate_image(sample_image) == True
    
    def test_image_optimization(self, sample_image):
        """Test image optimization."""
        from utils.image_processing import optimize_image
        
        optimized = optimize_image(sample_image)
        assert optimized is not None
        assert optimized.format == 'PNG' or optimized.mode == 'RGB'
    
    def test_invalid_image_size(self):
        """Test invalid image size handling."""
        from PIL import Image
        from utils.image_processing import validate_image
        
        # Create very small image
        small_img = Image.new('RGB', (50, 50), color='blue')
        assert validate_image(small_img) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
