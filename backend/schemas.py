"""
Pydantic schemas for API requests/responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class GarmentUploadResponse(BaseModel):
    """Response for garment upload."""
    garment_id: str = Field(..., description="Unique identifier for the garment")
    message: str
    status: str = Field("success", description="Upload status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RecommendationResult(BaseModel):
    """Single recommendation result."""
    garment_id: str = Field(..., description="ID of recommended garment")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score (0-1)")
    metadata: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    """Response for recommendation request."""
    query_garment_id: Optional[str] = None
    recommendations: List[RecommendationResult]
    total_count: int
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: str = Field(default="CLIP", description="Feature extractor model used")


class BulkUploadResponse(BaseModel):
    """Response for bulk garment upload."""
    total_uploaded: int
    successful: int
    failed: int
    failed_items: List[Dict[str, str]] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GarmentMetadata(BaseModel):
    """Metadata for a garment item."""
    category: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class SearchHistoryItem(BaseModel):
    """Item in search history."""
    query_id: str
    timestamp: datetime
    k_results: int
    top_result_score: Optional[float] = None
    processing_time_ms: float


class IndexStatsResponse(BaseModel):
    """Statistics about the recommendation index."""
    total_items: int
    feature_dimension: int
    model_type: str
    index_size_mb: float
    last_updated: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
