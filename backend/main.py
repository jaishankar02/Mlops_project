"""
FastAPI backend for StyleSync Recommender System.
Phase 1: Recommender only (GAN try-on deferred to Phase 2)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import sys
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from backend.routes import recommender
from backend.routes import tryon
from config.huggingface_config import get_huggingface_client
from config.mlflow_config import get_mlflow_tracker
from config.wandb_config import get_wandb_tracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Application startup")
    # Keep startup readiness independent of external tracking availability.
    # MLflow initialization is handled lazily during event logging.
    get_wandb_tracker().setup_wandb()
    get_huggingface_client()
    
    # Initialize the recommendation engine (bootstrap index) at startup
    from backend.routes.recommender import init_recommendation_engine
    logger.info("Bootstrapping recommendation engine...")
    init_recommendation_engine()
    
    yield
    try:
        get_wandb_tracker().finish()
    except Exception:
        pass
    logger.info("Application shutdown")


# Initialize FastAPI app
app = FastAPI(
    title="StyleSync - Recommender API",
    description="AI-Powered Visual Fashion Recommender (Phase 1)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(recommender.router, prefix="/api/recommender", tags=["recommender"])
app.include_router(tryon.router, prefix="/api/tryon", tags=["tryon"])


@app.get("/")
async def read_root():
    """Health check endpoint."""
    return {
        "status": "OK",
        "service": "StyleSync Recommender API",
        "phase": "Phase 1 - Recommender + Scoped Try-On Fallback",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check for deployment monitoring."""
    return {
        "status": "healthy",
        "service": "stylesync-recommender",
        "service_key": settings.BACKEND_SERVICE_KEY,
    }


@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics."""
    from config.mlflow_config import get_metrics
    return get_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )
