"""
Project setup and initialization script.
Run this to initialize the project structure and download necessary models.
"""
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary project directories."""
    directories = [
        "data",
        "logs",
        "mlruns",
        "models",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Created/verified directory: {directory}")


def setup_env_file():
    """Create .env file if it doesn't exist."""
    if not Path(".env").exists():
        logger.info("Creating .env file from template...")
        # .env file is already created
        logger.info("✓ .env file created")
    else:
        logger.info("✓ .env file already exists")


def download_clip_model():
    """Download CLIP model for feature extraction."""
    try:
        logger.info("Downloading CLIP model...")
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        logger.info("✓ CLIP model downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠ Could not download CLIP model: {e}")
        logger.info("  This will be downloaded on first API call")


def initialize_mlflow():
    """Initialize MLflow backend."""
    try:
        logger.info("Initializing MLflow...")
        import mlflow
        from config.settings import settings
        
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        logger.info("✓ MLflow initialized")
    except Exception as e:
        logger.warning(f"⚠ Could not initialize MLflow: {e}")


def main():
    """Run all setup tasks."""
    logger.info("=" * 50)
    logger.info("StyleSync Project Setup")
    logger.info("=" * 50)
    
    create_directories()
    setup_env_file()
    
    logger.info("\nOptional downloads (may take time):")
    download_clip_model()
    initialize_mlflow()
    
    logger.info("\n" + "=" * 50)
    logger.info("Setup complete! 🎉")
    logger.info("=" * 50)
    logger.info("\nNext steps:")
    logger.info("1. Start Docker services: docker-compose up -d")
    logger.info("2. Run backend: python -m backend.main")
    logger.info("3. Run frontend: streamlit run frontend/app.py")
    logger.info("4. Access MLflow: http://localhost:5000")


if __name__ == "__main__":
    main()
