"""Try-on backend selection helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from config.settings import settings
from .hr_viton_wrapper import get_hr_viton_wrapper
from .idm_vton_wrapper import get_idm_vton_wrapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TryOnSelection:
    model_name: str
    fallback_used: bool


def get_free_vram_mb() -> float:
    """Return free VRAM in megabytes for the current CUDA device."""
    if not torch.cuda.is_available():
        return 0.0

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return free_bytes / (1024 * 1024)
    except Exception:
        try:
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            allocated = torch.cuda.memory_allocated(device_index)
            reserved = torch.cuda.memory_reserved(device_index)
            free_bytes = max(props.total_memory - max(allocated, reserved), 0)
            return free_bytes / (1024 * 1024)
        except Exception:
            return 0.0


def should_fallback_to_hr_viton() -> bool:
    if not torch.cuda.is_available():
        return False
    return get_free_vram_mb() < float(settings.TRYON_VRAM_FALLBACK_THRESHOLD_MB)


def select_tryon_backend(preferred_backend: str | None = None) -> TryOnSelection:
    """Select the best available try-on backend for the current GPU memory budget.

    preferred_backend can be "hr_viton", "idm_vton", or "auto".
    """
    hr_viton = get_hr_viton_wrapper()
    idm_vton = get_idm_vton_wrapper()

    preferred = (preferred_backend or "auto").strip().lower()
    if preferred == "hr_viton" and settings.HR_VITON_ENABLED and hr_viton.is_available():
        return TryOnSelection(model_name="HR-VITON", fallback_used=True)
    if preferred == "idm_vton" and idm_vton.is_available():
        return TryOnSelection(model_name="IDM-VTON", fallback_used=False)
    if preferred == "hr_viton":
        return TryOnSelection(model_name="HR-VITON", fallback_used=True)
    if preferred == "idm_vton":
        return TryOnSelection(model_name="IDM-VTON", fallback_used=False)

    if settings.HR_VITON_ENABLED and should_fallback_to_hr_viton() and hr_viton.is_available():
        logger.info("Selected HR-VITON fallback due to low VRAM: %.0f MB free", get_free_vram_mb())
        return TryOnSelection(model_name="HR-VITON", fallback_used=True)

    if idm_vton.is_available():
        return TryOnSelection(model_name="IDM-VTON", fallback_used=False)

    if settings.HR_VITON_ENABLED and hr_viton.is_available():
        return TryOnSelection(model_name="HR-VITON", fallback_used=True)

    return TryOnSelection(model_name="IDM-VTON", fallback_used=False)


def get_selected_tryon_wrapper():
    """Return the selected try-on wrapper instance."""
    selection = select_tryon_backend()
    if selection.model_name == "HR-VITON":
        return get_hr_viton_wrapper(), selection
    return get_idm_vton_wrapper(), selection
