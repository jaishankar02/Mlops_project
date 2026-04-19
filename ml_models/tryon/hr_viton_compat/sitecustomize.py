"""Compatibility shims for running official HR-VITON code on modern NumPy."""
import numpy as np

# Official HR-VITON uses legacy NumPy aliases (e.g. np.float).
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "object"):
    np.object = object  # type: ignore[attr-defined]
