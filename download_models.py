"""
Download and setup pretrained models for the MLOps pipeline.
Run: python download_models.py
"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_models.model_registry import get_model_registry, PRETRAINED_MODELS
from config.mlflow_config import get_mlflow_tracker
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_models():
    """Download all recommended pretrained models."""
    registry = get_model_registry()
    mlflow_tracker = get_mlflow_tracker()
    
    print("\n" + "="*70)
    print("StyleSync: MLOps Pipeline - Model Download & Setup")
    print("="*70)
    
    print("\n📦 Available Pretrained Models:")
    print("-" * 70)
    
    for model_name, model_info in PRETRAINED_MODELS.items():
        print(f"\n  {model_name}")
        print(f"    Name: {model_info['name']}")
        print(f"    Task: {model_info['task']}")
        print(f"    Size: {model_info['size_mb']}MB")
        print(f"    Feature Dimension: {model_info['feature_dim']}")
    
    print("\n" + "-"*70)
    print("📥 Downloading models...")
    print("-"*70)
    
    downloaded_models = {}
    failed_models = {}
    
    for model_name in ["clip-vit-b-32", "resnet50"]:  # Download core models
        try:
            print(f"\n⏳ Downloading {model_name}...")
            model_path = registry.download_pretrained(model_name, force=False)
            
            if model_path:
                downloaded_models[model_name] = str(model_path)
                print(f"✅ {model_name} downloaded successfully")
                
                # Log to MLflow
                mlflow_tracker.log_event("model_download", {
                    "model_name": model_name,
                    "path": str(model_path),
                    "size_mb": PRETRAINED_MODELS[model_name]["size_mb"]
                })
            else:
                failed_models[model_name] = "Download failed"
                
        except Exception as e:
            failed_models[model_name] = str(e)
            logger.error(f"Error downloading {model_name}: {e}")
    
    print("\n" + "="*70)
    print("📊 Download Summary")
    print("="*70)
    print(f"✅ Successfully downloaded: {len(downloaded_models)}/{len(['clip-vit-b-32', 'resnet50'])}")
    
    if downloaded_models:
        print("\nDownloaded Models:")
        for model_name, path in downloaded_models.items():
            print(f"  ✓ {model_name}: {path}")
    
    if failed_models:
        print("\nFailed Models:")
        for model_name, error in failed_models.items():
            print(f"  ✗ {model_name}: {error}")
    
    print("\n" + "="*70)
    print("✨ Setup Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Initialize recommendation engine: python -m backend.main")
    print("2. Access Streamlit UI: http://localhost:8501")
    print("3. Monitor MLflow: http://localhost:5000")
    print("\n" + "="*70 + "\n")
    
    return len(downloaded_models) > 0


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
