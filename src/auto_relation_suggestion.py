
"""Auto-suggest relations after a saved memo; never auto-approve."""
import math
import os
import threading

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.data_manager import load_memos
from src.auto_relation_classifier import (
    GeminiRelationClassifier, parse_time
)

router = APIRouter()
_BUSY = threading.BoundedSemaphore(1)
_CACHE = {}
_CACHE_LOCK = threading.Lock()
_TYPES = {"revision", "development", "branch", "related"}


def _select_candidates(memos, child, limit=2):
    """Select older, different, high-cosine memos."""
    try:
        cv = np.asarray(child["vector"], dtype=float)
        ct = parse_time(child)
        cn = float(np.linalg.norm(cv))
        if cv.ndim != 1 or cn <= 0 or not np.isfinite(cv).all():
            return []
    except (ValueError, TypeError, KeyError):
        return []

    existing = {
        str(link.get("parent_id"))
        for link in (child.get("thought_links") or [])
        if isinstance(link, dict)
    }
    existing.update(str(i) for i in (child.get("supersedes") or []))
    matches = []

    for parent in memos:
        if parent.get("id") == child.get("id"):
            continue
        if parent.get("status") == "deleted":
            continue
        if str(parent.get("id")) in existing:
            continue
        if (
            str(parent.get("content") or "").strip()
            == str(child.get("content") or "").strip()
        ):
            continue

        try:
            if parse_time(parent) >= ct:
                continue
            pv = np.asarray(parent["vector"], dtype=float)
            if pv.shape != cv.shape or not np.isfinite(pv).all():
                continue
            pn = float(np.linalg.norm(pv))
            if pn <= 0:
                continue
            sim = float(np.dot(cv, pv) / (cn * pn))
        except (ValueError, TypeError, KeyError):
            continue

        if math.isfinite(sim) and sim >= 0.80:
            matches.append((sim, str(parent["id"]), parent))

    matches.sort(key=lambda item: (-item[0], item[1]))
    return [(memo, sim) for sim, _, memo in matches[:limit]]


@router.post("/api/auto-relation-suggest")
async def auto_relation_suggest(request: Request):
    # Explicitly disabled by default and during frozen study sessions.
    if os.getenv("THESIS_STUDY_SEARCH_LOCK") == "1":
        return JSONResponse(status_code=403, content={"error": "Study mode locked"})

    if os.getenv("AUTO_RELATION_SUGGEST_ENABLED") != "1":
        return JSONResponse(status_code=503, content={"error": "Auto suggestion disabled"})

    # Local development only. Public deployment needs user authentication,
    # persistent quotas and abuse controls before this can be enabled.
    client_host = request.client.host if request.client else ""
    if (
        client_host not in {"127.0.0.1", "::1"}
        or request.url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or os.getenv("RENDER")
    ):
        return JSONResponse(status_code=403, content={"error": "Local access only"})

    origin = request.headers.get("origin")
    if origin:
        from urllib.parse import urlparse
        if urlparse(origin).hostname not in {
            "127.0.0.1", "localhost", "::1"
        }:
            return JSONResponse(status_code=403, content={"error": "Invalid origin"})

    try:
        payload = await request.json()
        memo_id = payload.get("memo_id")
        if not isinstance(memo_id, str) or not (
            1 <= len(memo_id) <= 128
        ):
            return JSONResponse(status_code=400, content={"error": "Invalid memo ID"})
    except (ValueError, AttributeError, TypeError):
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    with _CACHE_LOCK:
        cached = _CACHE.get(memo_id)
    if cached is not None:
        return cached

    if not _BUSY.acquire(blocking=False):
        return JSONResponse(status_code=429, content={"error": "Classifier busy"})

    try:
        memos = load_memos()
        child = next(
            (m for m in memos if str(m.get("id")) == memo_id),
            None
        )
        if child is None or child.get("status") == "deleted":
            return JSONResponse(status_code=404, content={"error": "Memo not found"})

        candidates = _select_candidates(memos, child, limit=2)
        if not candidates:
            result = {
                "status": "success",
                "memo_id": memo_id,
                "candidates": [],
                "failed_count": 0
            }
        else:
            model = os.getenv(
                "GEMMA_RELATION_MODEL", "gemma-4-26b-a4b-it"
            )
            classifier = GeminiRelationClassifier(
                model=model,
                max_retries=1,
                format_retries=0
            )
            suggestions = []
            failed = 0

            for parent, cosine in candidates:
                try:
                    answer = await run_in_threadpool(
                        classifier.classify, parent, child
                    )
                except Exception as exc:
                    failed += 1
                    print(
                        "Auto relation classification error:",
                        type(exc).__name__
                    )
                    continue

                relation = answer.get("relation")
                confidence = answer.get("confidence")
                if relation not in _TYPES:
                    continue
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not math.isfinite(float(confidence))
                ):
                    continue

                suggestions.append({
                    "parent_id": str(parent["id"]),
                    "child_id": memo_id,
                    "parent_title": str(parent.get("title") or ""),
                    "relation": relation,
                    "confidence": float(confidence),
                    "reason": str(answer.get("reason") or ""),
                    "cosine": round(cosine, 4),
                    "low_confidence": float(confidence) < 0.75
                })

            if failed == len(candidates):
                return JSONResponse(status_code=503, content={"error": "Gemma unavailable; memo saved"})

            result = {
                "status": "success",
                "memo_id": memo_id,
                "candidates": suggestions,
                "failed_count": failed
            }

        with _CACHE_LOCK:
            if len(_CACHE) >= 256:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[memo_id] = result
        return result

    finally:
        _BUSY.release()
