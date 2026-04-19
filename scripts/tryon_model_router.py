#!/usr/bin/env python3
"""Inspect available VRAM and print the try-on backend that StyleSync will select."""
from __future__ import annotations

from ml_models.tryon.selector import get_free_vram_mb, select_tryon_backend


def main() -> None:
    selection = select_tryon_backend()
    print(f"selected={selection.model_name}")
    print(f"fallback_used={selection.fallback_used}")
    print(f"free_vram_mb={get_free_vram_mb():.0f}")


if __name__ == "__main__":
    main()
