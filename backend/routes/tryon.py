"""Virtual try-on API routes using IDM-VTON only."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from backend.schemas import TryOnResponse
from config.mlflow_config import log_recommendation_event
from config.settings import settings
from config.wandb_config import log_wandb_event
from ml_models.tryon.idm_vton_wrapper import get_idm_vton_wrapper
from utils.image_processing import optimize_image, validate_image

router = APIRouter()
logger = logging.getLogger(__name__)


async def _load_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    image = Image.open(BytesIO(data))
    if not validate_image(image):
        raise HTTPException(status_code=400, detail=f"Invalid image: {upload.filename}")
    return optimize_image(image)


def _limit_output_image_size(image: Image.Image) -> Image.Image:
    max_width = max(1, int(settings.TRYON_OUTPUT_MAX_WIDTH))
    max_height = max(1, int(settings.TRYON_OUTPUT_MAX_HEIGHT))

    if image.width <= max_width and image.height <= max_height:
        return image

    resized = image.copy()
    resized.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return resized


def _is_degenerate_tryon_output(image: Image.Image) -> bool:
    return image.width < 8 or image.height < 8


def _get_idm_model():
    idm_vton = get_idm_vton_wrapper()
    if not idm_vton.is_available():
        raise HTTPException(status_code=503, detail="IDM-VTON is not available")
    return idm_vton


@router.post("/generate", response_model=TryOnResponse)
async def generate_tryon(
    person_file: UploadFile = File(...),
    garment_file: UploadFile = File(...),
    use_gan: bool = Query(False, description="Ignored; IDM-VTON is the only supported model"),
    hf_repo_id: Optional[str] = Query(None, description="Ignored"),
    hf_filename: Optional[str] = Query(None, description="Ignored"),
):
    """Generate a try-on result using the official IDM-VTON demo."""
    del use_gan, hf_repo_id, hf_filename

    start_time = time.time()
    person_image = await _load_image(person_file)
    garment_image = await _load_image(garment_file)
    idm_vton = _get_idm_model()

    try:
        result_image = idm_vton.generate_tryon(person_image, garment_image)
    except Exception as exc:
        logger.error("IDM-VTON inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"IDM-VTON inference failed: {exc}")

    if result_image is None:
        raise HTTPException(status_code=500, detail="IDM-VTON returned no result")

    if _is_degenerate_tryon_output(result_image):
        raise HTTPException(status_code=422, detail="IDM-VTON output failed quality checks")

    result_image = _limit_output_image_size(result_image)
    buffer = BytesIO()
    result_image.save(buffer, format="PNG", optimize=True)
    result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    processing_time = (time.time() - start_time) * 1000

    log_recommendation_event("tryon_request", {
        "model_used": "IDM-VTON",
        "processing_time_ms": processing_time,
    })
    log_wandb_event("tryon_request", {
        "model_used": "IDM-VTON",
        "processing_time_ms": processing_time,
    })

    return TryOnResponse(
        model_used="IDM-VTON",
        fallback_used=False,
        processing_time_ms=processing_time,
        result_image_base64=result_b64,
    )


@router.post("/generate-streaming")
async def generate_tryon_streaming(
    person_file: UploadFile = File(...),
    garment_file: UploadFile = File(...),
    use_gan: bool = Query(False, description="Ignored; IDM-VTON is the only supported model"),
    hf_repo_id: Optional[str] = Query(None, description="Ignored"),
    hf_filename: Optional[str] = Query(None, description="Ignored"),
):
    """Generate a try-on result and stream progress updates."""
    del use_gan, hf_repo_id, hf_filename

    async def progress_stream():
        try:
            start_time = time.time()
            yield json.dumps({"status": "loading_images", "progress": 5, "message": "Loading person and garment images..."}) + "\n"

            person_image = await _load_image(person_file)
            garment_image = await _load_image(garment_file)

            yield json.dumps({"status": "model_ready", "progress": 15, "message": "Using IDM-VTON official demo..."}) + "\n"
            yield json.dumps({"status": "inferencing", "progress": 20, "message": "Starting IDM-VTON inference (official Hugging Face pipeline)...", "estimated_seconds": 120}) + "\n"

            idm_vton = _get_idm_model()
            try:
                result_image = idm_vton.generate_tryon(person_image, garment_image)
            except Exception as exc:
                yield json.dumps({"status": "error", "progress": 0, "message": f"Inference failed: {exc}"}) + "\n"
                return

            if result_image is None:
                yield json.dumps({"status": "error", "progress": 0, "message": "IDM-VTON returned no result"}) + "\n"
                return

            yield json.dumps({"status": "quality_check", "progress": 85, "message": "Checking output quality...", "elapsed_seconds": round(time.time() - start_time, 1)}) + "\n"

            if _is_degenerate_tryon_output(result_image):
                yield json.dumps({"status": "error", "progress": 0, "message": "IDM-VTON output failed quality checks"}) + "\n"
                return

            yield json.dumps({"status": "finalizing", "progress": 95, "message": "Finalizing output..."}) + "\n"

            result_image = _limit_output_image_size(result_image)
            buffer = BytesIO()
            result_image.save(buffer, format="PNG", optimize=True)
            result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            processing_time = (time.time() - start_time) * 1000

            log_recommendation_event("tryon_request", {
                "model_used": "IDM-VTON",
                "processing_time_ms": processing_time,
            })
            log_wandb_event("tryon_request", {
                "model_used": "IDM-VTON",
                "processing_time_ms": processing_time,
            })

            yield json.dumps({
                "status": "success",
                "progress": 100,
                "message": "Complete!",
                "model_used": "IDM-VTON",
                "processing_time_ms": round(processing_time, 1),
                "result_image_base64": result_b64,
            }) + "\n"
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Request validation failed"
            yield json.dumps({"status": "error", "progress": 0, "message": detail}) + "\n"
        except Exception as exc:
            logger.error("Error in streaming try-on: %s", exc)
            yield json.dumps({"status": "error", "progress": 0, "message": f"Error: {exc}"}) + "\n"

    return StreamingResponse(progress_stream(), media_type="application/x-ndjson")


@router.get("/health")
async def tryon_health():
    """Report IDM-VTON availability status."""
    idm_wrapper = get_idm_vton_wrapper()
    return {
        "status": "healthy",
        "idm_vton_enabled": settings.IDM_VTON_ENABLED,
        "idm_vton_available": idm_wrapper.is_available(),
        "idm_vton_model_id": getattr(idm_wrapper, "model_id", settings.IDM_VTON_PRETRAINED_MODEL_NAME_OR_PATH),
        "idm_vton_space_id": getattr(idm_wrapper, "space_id", "yisol/IDM-VTON"),
        "idm_vton_snapshot_path": str(getattr(idm_wrapper, "_snapshot_path", None)) if getattr(idm_wrapper, "_snapshot_path", None) else None,
        "idm_vton_demo_path": str(idm_wrapper.demo_path),
        "idm_vton_seed": int(os.getenv("IDM_VTON_SEED", "42")),
        "idm_vton_steps": int(os.getenv("IDM_VTON_STEPS", "30")),
    }