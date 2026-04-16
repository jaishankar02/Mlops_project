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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Application startup")
    yield
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


@app.get("/")
async def read_root():
    """Health check endpoint."""
    return {
        "status": "OK",
        "service": "StyleSync Recommender API",
        "phase": "Phase 1 - Recommender Only",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check for deployment monitoring."""
    return {"status": "healthy", "service": "stylesync-recommender"}


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
