"""Weights & Biases configuration and lightweight event logging."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency wiring
    wandb = None


class WandBTracker:
    """Small singleton wrapper around WandB with safe disabled fallback."""

    def __init__(self) -> None:
        self._run = None
        self._initialized = False

    def _init_run(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        if wandb is None:
            logger.warning("wandb is unavailable; tracking is disabled")
            return

        mode = settings.WANDB_MODE or "disabled"
        if settings.WANDB_API_KEY and not os.getenv("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = settings.WANDB_API_KEY
        init_kwargs = {
            "project": settings.WANDB_PROJECT,
            "mode": mode,
            "reinit": True,
            "settings": wandb.Settings(start_method="thread"),
        }
        if settings.WANDB_ENTITY:
            init_kwargs["entity"] = settings.WANDB_ENTITY

        try:
            self._run = wandb.init(**init_kwargs)
            logger.info("WandB initialized in %s mode", mode)
        except Exception as exc:
            logger.warning("WandB initialization failed: %s", exc)
            self._run = None

    def setup_wandb(self) -> None:
        """Public helper for eager initialization."""
        self._init_run()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self._init_run()
        if self._run is None or wandb is None:
            return
        try:
            wandb.log(metrics, step=step)
        except Exception as exc:
            logger.warning("WandB metric logging failed: %s", exc)

    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        payload = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **event_data,
        }
        numeric_metrics = {
            f"event/{key}": float(value)
            for key, value in event_data.items()
            if isinstance(value, (int, float))
        }
        if numeric_metrics:
            self.log_metrics(numeric_metrics)
        self._init_run()
        if self._run is None or wandb is None:
            return
        try:
            wandb.log({f"event/{event_type}/count": 1})
            if wandb.run is not None:
                wandb.run.summary[f"event/{event_type}_payload"] = str(payload)
        except Exception as exc:
            logger.warning("WandB event logging failed: %s", exc)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        self._init_run()
        if self._run is None or wandb is None:
            return
        try:
            artifact = wandb.Artifact(name or path.split("/")[-1], type="artifact")
            artifact.add_file(path)
            wandb.log_artifact(artifact)
        except Exception as exc:
            logger.warning("WandB artifact logging failed: %s", exc)

    def finish(self) -> None:
        if self._run is not None and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass


_wandb_tracker: Optional[WandBTracker] = None


def get_wandb_tracker() -> WandBTracker:
    global _wandb_tracker
    if _wandb_tracker is None:
        _wandb_tracker = WandBTracker()
    return _wandb_tracker


def log_wandb_event(event_type: str, event_data: Dict[str, Any]) -> None:
    get_wandb_tracker().log_event(event_type, event_data)


def setup_wandb() -> None:
    """Initialize WandB at startup when available."""
    get_wandb_tracker().setup_wandb()
