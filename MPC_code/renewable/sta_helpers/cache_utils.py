"""Utilities for disk caching of offline MPC computations.

This module provides deterministic hashing of inputs and simple
pickle-based load/save helpers for caching expensive preprocessing
results such as constraint tightening and terminal-set calculations.
"""

from __future__ import annotations

import json
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np


CACHE_VERSION = "mpc_offline_v1"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_ndarray(arr: np.ndarray) -> dict[str, Any]:
    """Return a stable digest description of a NumPy array."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return {
        "__ndarray__": True,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "sha256": h.hexdigest(),
    }


def _normalize_for_hash(obj: Any) -> Any:
    """Convert Python / NumPy objects into deterministic JSON-safe data."""
    if isinstance(obj, np.ndarray):
        return _hash_ndarray(obj)

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): _normalize_for_hash(obj[k]) for k in sorted(obj.keys())}

    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback for objects that are not directly serializable.
    return repr(obj)


def make_cache_key(name: str, inputs: Any) -> str:
    """Create a deterministic cache key from a function name and inputs."""
    payload = {
        "cache_version": CACHE_VERSION,
        "name": name,
        "inputs": _normalize_for_hash(inputs),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cache_path(name: str, key: str, cache_dir: Path | None = None) -> Path:
    """Construct a cache filename for a given function name and key."""
    base = cache_dir if cache_dir is not None else CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}_{key}.pkl"


def load_cache(path: Path) -> Any | None:
    """Load a cached object from disk, or return None if it does not exist."""
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(path: Path, obj: Any) -> None:
    """Save an object to disk using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
