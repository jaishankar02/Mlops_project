"""
Application settings and configuration.
"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit
        "http://localhost:8000",
    ]
    
    # Model Settings
    FEATURE_EXTRACTOR_MODEL: str = os.getenv("FEATURE_EXTRACTOR", "clip")  # clip or resnet
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    FAISS_METADATA_PATH: str = os.getenv("FAISS_METADATA_PATH", "data/faiss_metadata.json")

    # Dataset Settings
    DATASET_NAME: str = os.getenv("DATASET_NAME", "fashion-catalog")
    RECOMMENDER_DATASET_DIR: str = os.getenv(
        "RECOMMENDER_DATASET_DIR",
        "/data/m25csa007/datasets/high_resolution_viton_zalando/test/cloth",
    )
    TRYON_TRAIN_DATASET_DIR: str = os.getenv(
        "TRYON_TRAIN_DATASET_DIR",
        "/data/m25csa007/datasets/high_resolution_viton_zalando/train/cloth",
    )
    
    # Image Settings
    MAX_IMAGE_SIZE_MB: int = 10
    SUPPORTED_FORMATS: List[str] = ["JPEG", "PNG"]
    IMAGE_RESIZE_HEIGHT: int = 224
    IMAGE_RESIZE_WIDTH: int = 224
    TRYON_OUTPUT_MAX_WIDTH: int = int(os.getenv("TRYON_OUTPUT_MAX_WIDTH", "768"))
    TRYON_OUTPUT_MAX_HEIGHT: int = int(os.getenv("TRYON_OUTPUT_MAX_HEIGHT", "1024"))
    TRYON_WHITE_BG_PREPROCESS: bool = os.getenv("TRYON_WHITE_BG_PREPROCESS", "True").lower() == "true"
    TRYON_WHITE_BG_THRESHOLD: float = float(os.getenv("TRYON_WHITE_BG_THRESHOLD", "0.35"))
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/stylesync")
    DATABASE_ECHO: bool = os.getenv("DATABASE_ECHO", "False").lower() == "true"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "recommender-phase-1")
    MLFLOW_BACKEND_STORE_URI: str = os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db")
    MLFLOW_ARTIFACT_ROOT: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

    # Hugging Face / Weights & Biases
    HF_TRYON_REPO_ID: str = os.getenv("HF_TRYON_REPO_ID", "")
    HF_TRYON_FILENAME: str = os.getenv("HF_TRYON_FILENAME", "checkpoint.pt")
    HF_TRYON_LOCAL_PATH: str = os.getenv("HF_TRYON_LOCAL_PATH", "")
    IDM_VTON_ENABLED: bool = os.getenv("IDM_VTON_ENABLED", "True").lower() == "true"
    IDM_VTON_PRETRAINED_MODEL_NAME_OR_PATH: str = os.getenv(
        "IDM_VTON_PRETRAINED_MODEL_NAME_OR_PATH",
        "yisol/IDM-VTON",
    )
    HF_MODEL_SOURCE: str = os.getenv("HF_MODEL_SOURCE", "auto")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "stylesync")
    WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "")
    WANDB_MODE: str = os.getenv("WANDB_MODE", "disabled")
    WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")

    # Kaggle Settings
    KAGGLE_USERNAME: str = os.getenv("KAGGLE_USERNAME", "")
    KAGGLE_KEY: str = os.getenv("KAGGLE_KEY", "")
    
    # GPU Settings
    USE_GPU: bool = os.getenv("USE_GPU", "True").lower() == "true"
    DEVICE: str = "cuda" if USE_GPU else "cpu"
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/stylesync.log")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
