"""SHA-256 helpers for files, arrays and JSON-serialisable objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(X: np.ndarray, y: np.ndarray) -> str:
    """Return the hex SHA-256 of ``X.tobytes() + y.tobytes()`` (RESULTS_SCHEMA §3)."""
    digest = hashlib.sha256()
    digest.update(np.asarray(X).tobytes())
    digest.update(np.asarray(y).tobytes())
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` as canonical JSON: sorted keys, no whitespace, no NaN/inf."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj: Any) -> str:
    """Return the hex SHA-256 of the canonical JSON form of ``obj`` (used for the config hash)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
