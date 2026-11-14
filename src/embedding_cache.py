"""Persistent, request-exact embedding cache; stores only SHA-256 keys and float64 vectors.

No API calls. No raw titles or memo contents are written to the cache database.
Cache keys are versioned and include ALL effective embedding request parameters.
"""

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from src import config as app_config

_SCHEMA_VERSION = 1


def _cache_path():
    configured = os.getenv("EMBED_CACHE_PATH", "").strip()
    return Path(configured).expanduser() if configured else (
        app_config.BASE_DIR / "data" / "cache" / "embedding_vectors_v1.sqlite3"
    )


def request_key(*, text, model, dim, style, task_type, use_task_type):
    """A digest of the exact request, including styles and task-type mode.

    Caller passes build_request()'s final text (including any formatted title).
    Deliberately keeps legacy and documented embeddings separate even when
    the resulting text happens to be identical.
    """
    payload = {
        "cache_schema": _SCHEMA_VERSION,
        "model": str(model),
        "dim": int(dim),
        "style": str(style),
        "final_text": str(text),
        "task_type": str(task_type) if use_task_type else None,
        "use_task_type": bool(use_task_type),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _connect():
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return connection


def read(key, dim):
    """Read a defensive copy; return None on misses, corruption or I/O error."""
    try:
        with closing(_connect()) as con:
            row = con.execute(
                "SELECT dim, vector FROM embeddings WHERE cache_key=?", (key,)
            ).fetchone()
        if row is None or row[0] != dim or len(row[1]) != dim * 8:
            return None
        vector = np.frombuffer(row[1], dtype="<f8").copy()
        return vector if np.isfinite(vector).all() else None
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return None  # Cache failure must not corrupt the embedding path.


def write(key, vector, dim):
    """Store only successfully obtained finite vectors as exact float64 bytes."""
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or len(value) != dim or not np.isfinite(value).all():
        return False
    try:
        with closing(_connect()) as con:
            with con:
                con.execute(
                    "INSERT OR REPLACE INTO embeddings(cache_key, dim, vector) VALUES (?, ?, ?)",
                    (key, dim, sqlite3.Binary(value.astype("<f8", copy=False).tobytes())),
                )
        return True
    except (OSError, sqlite3.Error):
        return False
