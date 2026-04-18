"""Hugging Face Hub helpers for optional model and checkpoint downloads."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except Exception:  # pragma: no cover - optional dependency wiring
    HfApi = None
    hf_hub_download = None
    snapshot_download = None


class HuggingFaceHubClient:
    """Minimal helper around the Hugging Face Hub download APIs."""

    def __init__(self, cache_dir: str = "models/hf") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self,
        repo_id: str,
        filename: str,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> Optional[Path]:
        if hf_hub_download is None:
            logger.warning("huggingface_hub is unavailable; skipping download")
            return None

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                revision=revision,
                token=settings.HF_TOKEN or None,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
            )
            return Path(downloaded_path)
        except Exception as exc:
            logger.warning("Hugging Face file download failed: %s", exc)
            return None

    def download_snapshot(
        self,
        repo_id: str,
        allow_patterns: Optional[list[str]] = None,
        revision: Optional[str] = None,
    ) -> Optional[Path]:
        if snapshot_download is None:
            logger.warning("huggingface_hub is unavailable; skipping snapshot download")
            return None

        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                revision=revision,
                token=settings.HF_TOKEN or None,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
            )
            return Path(snapshot_path)
        except Exception as exc:
            logger.warning("Hugging Face snapshot download failed: %s", exc)
            return None

    def upload_file(
        self,
        repo_id: str,
        local_path: str,
        path_in_repo: Optional[str] = None,
        repo_type: str = "model",
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
    ) -> bool:
        if HfApi is None:
            logger.warning("huggingface_hub is unavailable; skipping upload")
            return False

        try:
            api = HfApi(token=settings.HF_TOKEN or None)
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo or Path(local_path).name,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            return True
        except Exception as exc:
            logger.warning("Hugging Face file upload failed: %s", exc)
            return False

    def upload_folder(
        self,
        repo_id: str,
        folder_path: str,
        repo_type: str = "model",
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
    ) -> bool:
        if HfApi is None:
            logger.warning("huggingface_hub is unavailable; skipping upload")
            return False

        try:
            api = HfApi(token=settings.HF_TOKEN or None)
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            return True
        except Exception as exc:
            logger.warning("Hugging Face folder upload failed: %s", exc)
            return False


_hf_client: Optional[HuggingFaceHubClient] = None


def get_huggingface_client() -> HuggingFaceHubClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = HuggingFaceHubClient()
    return _hf_client