"""
Recommender API routes.
"""
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
from utils.image_processing import validate_image, optimize_image

router = APIRouter()
logger = logging.getLogger(__name__)

# Global recommendation engine instance
recommendation_engine = None


def init_recommendation_engine():
    """Initialize recommendation engine on startup."""
    global recommendation_engine
    if recommendation_engine is None:
        recommendation_engine = RecommendationEngine(feature_dim=512)
        recommendation_engine.initialize_feature_extractor(model_name="clip")
        
        # Try to load existing index
        index_path = "data/faiss_index.bin"
        if Path(index_path).exists():
            recommendation_engine.load_index(index_path)
            logger.info(f"Loaded existing index with {recommendation_engine.index.ntotal} items")
        else:
            logger.info("No existing index found, starting with empty index")
    
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
        
        # Create metadata
        metadata = {
            "category": category,
            "color": color,
            "size": size,
            "price": price,
            "filename": file.filename
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
        
        logger.info(f"Garment {garment_id} uploaded successfully")
        
        return GarmentUploadResponse(
            garment_id=garment_id,
            message=f"Garment uploaded successfully",
            status="success"
        )
    
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
                
                if recommendation_engine.add_garment(garment_id, image, {"filename": file.filename}):
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
        
        return BulkUploadResponse(
            total_uploaded=len(files),
            successful=successful,
            failed=failed,
            failed_items=failed_items
        )
    
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=RecommendationResponse)
async def search_similar_garments(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=20, description="Number of similar items to retrieve"),
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
        results = recommendation_engine.search_similar(image, k=k)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Format recommendations
        recommendations = [
            RecommendationResult(
                garment_id=garment_id,
                similarity_score=score,
                metadata=recommendation_engine.metadata[idx]["metadata"] 
                    if idx < len(recommendation_engine.metadata) else None
            )
            for idx, (garment_id, score) in enumerate(results)
        ]
        
        # Log to MLflow
        log_recommendation_event("search_query", {
            "k_results": k,
            "results_count": len(recommendations),
            "processing_time_ms": processing_time,
            "top_score": recommendations[0].similarity_score if recommendations else 0
        })
        
        logger.info(f"Search completed in {processing_time:.2f}ms, found {len(recommendations)} results")
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            processing_time_ms=processing_time,
            model_used="CLIP"
        )
    
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=IndexStatsResponse)
async def get_index_statistics():
    """Get statistics about the recommendation index."""
    try:
        init_recommendation_engine()
        
        import os
        index_path = "data/faiss_index.bin"
        index_size = os.path.getsize(index_path) / (1024 * 1024) if os.path.exists(index_path) else 0
        
        return IndexStatsResponse(
            total_items=recommendation_engine.index.ntotal,
            feature_dimension=recommendation_engine.feature_dim,
            model_type="CLIP ViT-B/32",
            index_size_mb=index_size
        )
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-index")
async def save_index_to_disk():
    """Save the current index to disk."""
    try:
        init_recommendation_engine()
        os.makedirs("data", exist_ok=True)
        
        success = recommendation_engine.save_index("data/faiss_index.bin")
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save index")
        
        logger.info("Index saved successfully")
        return {"status": "success", "message": "Index saved to disk"}
    
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
        
        logger.warning("Index has been reset")
        return {"status": "success", "message": "Index has been reset"}
    
    except Exception as e:
        logger.error(f"Error resetting index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
