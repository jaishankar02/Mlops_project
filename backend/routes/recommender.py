"""
Recommender API routes.
"""
import base64
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from PIL import Image
import logging
import time
from io import BytesIO
from typing import List, Optional
import os
from pathlib import Path

from backend.schemas import (
    GarmentUploadResponse, RecommendationResponse, 
    RecommendationResult, BulkUploadResponse, IndexStatsResponse
)
from ml_models.recommender.recommendation_engine import RecommendationEngine
from config.mlflow_config import log_recommendation_event
from config.wandb_config import log_wandb_event
from config.settings import settings
from utils.image_processing import validate_image, optimize_image

router = APIRouter()
logger = logging.getLogger(__name__)

# Global recommendation engine instance
recommendation_engine = None


GARMENT_IMAGES_DIR = Path("data/garments")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_data_dirs() -> None:
    GARMENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_index_path(index_path: str) -> Path:
    """Resolve index path so it is stable even when server starts from another cwd."""
    raw = Path(index_path)
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw)


def _bootstrap_limit_from_env() -> Optional[int]:
    """Read optional bootstrap cap. 0 or negative means no cap (index all)."""
    raw = os.getenv("RECOMMENDER_BOOTSTRAP_MAX_ITEMS", "0")
    try:
        value = int(raw)
    except ValueError:
        value = 0
    return value if value > 0 else None


def _bootstrap_index_from_dataset(max_items: Optional[int] = None) -> int:
    """Build recommender index from known cloth folders when index is empty."""
    if recommendation_engine is None:
        return 0

    candidate_dirs = []

    configured = Path(settings.RECOMMENDER_DATASET_DIR)
    candidate_dirs.append(configured)

    # Common local dataset path used in this workspace.
    candidate_dirs.append(Path("/data/m25csa007/datasets/high_resolution_viton_zalando/test/cloth"))
    candidate_dirs.append(Path("/data/m25csa007/datasets/high_resolution_viton_zalando/train/cloth"))

    seen = set()
    total_added = 0

    for cloth_dir in candidate_dirs:
        if max_items is not None and total_added >= max_items:
            break
        if not cloth_dir.exists() or not cloth_dir.is_dir():
            continue

        # Avoid indexing the same folder twice through aliases/config overlap.
        real_dir = str(cloth_dir.resolve())
        if real_dir in seen:
            continue
        seen.add(real_dir)

        images = sorted(
            [
                p for p in cloth_dir.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        for path in images:
            if max_items is not None and total_added >= max_items:
                break
            try:
                img = Image.open(path).convert("RGB")
                garment_id = f"seed_{path.stem}_{total_added + 1}"
                ok = recommendation_engine.add_garment(
                    garment_id=garment_id,
                    garment_image=img,
                    metadata={
                        "source": "dataset-bootstrap",
                        "filename": path.name,
                        "image_path": str(path),
                    },
                )
                if ok:
                    total_added += 1
            except Exception as exc:
                logger.warning("Skipping garment %s: %s", path, exc)

    return total_added


def _save_garment_image(garment_id: str, image: Image.Image) -> str:
    """Persist optimized garment image and return relative path."""
    _ensure_data_dirs()
    out_path = GARMENT_IMAGES_DIR / f"{garment_id}.png"
    image.save(out_path, format="PNG", optimize=True)
    return str(out_path)


def _get_image_preview_base64(metadata: Optional[dict]) -> Optional[str]:
    """Load persisted garment image and return base64-encoded preview."""
    if not metadata:
        return None

    image_path = metadata.get("image_path")
    if not image_path:
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as exc:
        logger.warning("Failed to load garment preview %s: %s", image_path, exc)
        return None


def init_recommendation_engine():
    """Initialize recommendation engine on startup."""
    global recommendation_engine
    if recommendation_engine is None:
        resolved_index_path = _resolve_index_path(settings.FAISS_INDEX_PATH)
        recommendation_engine = RecommendationEngine(
            feature_dim=512,
            faiss_index_path=str(resolved_index_path),
        )
        recommendation_engine.initialize_feature_extractor(model_name="clip")
        
        # Try to load existing index
        loaded_ok = False
        if resolved_index_path.exists():
            loaded_ok = recommendation_engine.load_index(str(resolved_index_path))
            if loaded_ok:
                logger.info(f"Loaded existing index with {recommendation_engine.index.ntotal} items")
            else:
                logger.warning("Existing index could not be loaded, rebuilding from dataset")
        else:
            logger.info("No existing index found, starting with empty index")

        # Auto-bootstrap from dataset when index is still empty.
        if (not loaded_ok) or recommendation_engine.index.ntotal == 0:
            recommendation_engine.reset_index()
            max_items = _bootstrap_limit_from_env()
            added = _bootstrap_index_from_dataset(max_items=max_items)
            if added > 0:
                resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
                recommendation_engine.save_index(str(resolved_index_path))
                if max_items is None:
                    logger.info("Bootstrapped recommender index with %d garments (full dataset)", added)
                else:
                    logger.info("Bootstrapped recommender index with %d garments (cap=%d)", added, max_items)
            else:
                logger.warning("Index bootstrap found no garments in dataset directories")
    
    return recommendation_engine


@router.post("/upload-garment", response_model=GarmentUploadResponse)
async def upload_garment(
    file: UploadFile = File(...),
    garment_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    color: Optional[str] = Query(None),
    size: Optional[str] = Query(None),
    price: Optional[float] = Query(None),
):
    """
    Upload a single garment image.
    
    - **file**: Garment image (JPEG, PNG)
    - **garment_id**: Unique identifier (auto-generated if not provided)
    - **category**: Garment category (e.g., shirt, pants, dress)
    - **color**: Primary color
    - **size**: Size (e.g., S, M, L, XL)
    - **price**: Price in USD
    """
    try:
        init_recommendation_engine()
        
        # Read and validate image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        if not validate_image(image):
            raise HTTPException(status_code=400, detail="Invalid image format or size")
        
        # Generate garment ID if not provided
        if not garment_id:
            import uuid
            garment_id = f"garment_{uuid.uuid4().hex[:12]}"
        
        # Optimize image
        image = optimize_image(image)
        image_path = _save_garment_image(garment_id, image)
        
        # Create metadata
        metadata = {
            "category": category,
            "color": color,
            "size": size,
            "price": price,
            "filename": file.filename,
            "image_path": image_path,
        }
        
        # Add to recommendation engine
        success = recommendation_engine.add_garment(garment_id, image, metadata)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process garment")
        
        # Log to MLflow
        log_recommendation_event("garment_upload", {
            "garment_id": garment_id,
            "category": category,
            "total_items": recommendation_engine.index.ntotal
        })
        log_wandb_event("garment_upload", {
            "total_items": recommendation_engine.index.ntotal,
        })
        
        logger.info(f"Garment {garment_id} uploaded successfully")
        
        return GarmentUploadResponse(
            garment_id=garment_id,
            message=f"Garment uploaded successfully",
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading garment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload_garments(files: List[UploadFile] = File(...)):
    """
    Bulk upload multiple garment images.
    
    - **files**: List of garment images (max 50 at a time)
    """
    try:
        init_recommendation_engine()
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 files per upload")
        
        successful = 0
        failed = 0
        failed_items = []
        
        for file in files:
            try:
                image_data = await file.read()
                image = Image.open(BytesIO(image_data))
                
                if not validate_image(image):
                    failed += 1
                    failed_items.append({"filename": file.filename, "error": "Invalid image"})
                    continue
                
                import uuid
                garment_id = f"garment_{uuid.uuid4().hex[:12]}"
                image = optimize_image(image)
                image_path = _save_garment_image(garment_id, image)

                if recommendation_engine.add_garment(
                    garment_id,
                    image,
                    {"filename": file.filename, "image_path": image_path},
                ):
                    successful += 1
                else:
                    failed += 1
                    failed_items.append({"filename": file.filename, "error": "Processing failed"})
            
            except Exception as e:
                failed += 1
                failed_items.append({"filename": file.filename, "error": str(e)})
        
        # Log to MLflow
        log_recommendation_event("bulk_upload", {
            "successful": successful,
            "failed": failed,
            "total_items": recommendation_engine.index.ntotal
        })
        log_wandb_event("bulk_upload", {
            "successful": successful,
            "failed": failed,
            "total_items": recommendation_engine.index.ntotal,
        })
        
        return BulkUploadResponse(
            total_uploaded=len(files),
            successful=successful,
            failed=failed,
            failed_items=failed_items
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=RecommendationResponse)
async def search_similar_garments(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=20, description="Number of similar items to retrieve"),
    min_similarity: float = Query(0.2, ge=0.0, le=1.0, description="Minimum blended similarity score"),
):
    """
    Search for similar garments given a query image.
    
    - **file**: Query garment image
    - **k**: Number of recommendations (1-20)
    """
    try:
        init_recommendation_engine()
        
        if recommendation_engine.index.ntotal == 0:
            raise HTTPException(status_code=400, detail="No garments in database yet")
        
        # Read and validate image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        if not validate_image(image):
            raise HTTPException(status_code=400, detail="Invalid image format or size")
        
        image = optimize_image(image)
        
        # Search
        start_time = time.time()
        results = recommendation_engine.search_similar(
            image,
            k=k,
            min_similarity=min_similarity,
        )
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Format recommendations
        recommendations = [
            RecommendationResult(
                garment_id=result["garment_id"],
                similarity_score=result["similarity_score"],
                metadata=recommendation_engine.metadata[result["faiss_index"]]["metadata"]
                    if 0 <= result["faiss_index"] < len(recommendation_engine.metadata) else None,
                image_base64=_get_image_preview_base64(
                    recommendation_engine.metadata[result["faiss_index"]]["metadata"]
                    if 0 <= result["faiss_index"] < len(recommendation_engine.metadata) else None
                ),
            )
            for result in results
        ]
        
        # Log to MLflow
        log_recommendation_event("search_query", {
            "k_results": k,
            "min_similarity": min_similarity,
            "results_count": len(recommendations),
            "processing_time_ms": processing_time,
            "top_score": recommendations[0].similarity_score if recommendations else 0
        })
        log_wandb_event("search_query", {
            "k_results": k,
            "min_similarity": min_similarity,
            "results_count": len(recommendations),
            "processing_time_ms": processing_time,
            "top_score": recommendations[0].similarity_score if recommendations else 0,
        })
        
        logger.info(f"Search completed in {processing_time:.2f}ms, found {len(recommendations)} results")
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            processing_time_ms=processing_time,
            model_used="CLIP"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=IndexStatsResponse)
async def get_index_statistics():
    """Get statistics about the recommendation index."""
    try:
        init_recommendation_engine()
        
        import os
        index_path = _resolve_index_path(settings.FAISS_INDEX_PATH)
        index_size = os.path.getsize(index_path) / (1024 * 1024) if os.path.exists(index_path) else 0
        
        return IndexStatsResponse(
            total_items=recommendation_engine.index.ntotal,
            feature_dimension=recommendation_engine.feature_dim,
            model_type="CLIP ViT-B/32",
            index_size_mb=index_size
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-index")
async def save_index_to_disk():
    """Save the current index to disk."""
    try:
        init_recommendation_engine()
        target_path = _resolve_index_path(settings.FAISS_INDEX_PATH)
        os.makedirs(target_path.parent, exist_ok=True)
        
        success = recommendation_engine.save_index(str(target_path))
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save index")
        
        logger.info("Index saved successfully")
        return {"status": "success", "message": "Index saved to disk"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-index")
async def reset_index():
    """Reset the recommendation index (use with caution)."""
    try:
        init_recommendation_engine()
        recommendation_engine.reset_index()
        
        log_recommendation_event("index_reset", {})
        log_wandb_event("index_reset", {})
        
        logger.warning("Index has been reset")
        return {"status": "success", "message": "Index has been reset"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
