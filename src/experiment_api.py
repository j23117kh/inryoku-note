"""Opt-in API for experimental search and Gemma relation proposals."""
import os
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.embedding import get_vector
from src.data_manager import load_memos, live_memos
from src.search_experiment import search_experiment, MODES
from src.auto_relation_classifier import (
    GeminiRelationClassifier, parse_time
)

router = APIRouter()


def blocked():
    return os.getenv("THESIS_STUDY_SEARCH_LOCK") == "1"


@router.post("/api/experiment-search")
async def experiment_search_api(request: Request):
    if blocked():
        return JSONResponse(
            status_code=403,
            content={"error": "参加者実験では使用できません"}
        )
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("JSONオブジェクトが必要です")
        query = data.get("text", "")
        mode = data.get("mode", "B_stale080")
        top_k = data.get("top_k", 10)
        threshold = data.get("threshold", 30)

        if not isinstance(query, str) or not query.strip():
            raise ValueError("質問を入力してください")
        if len(query) > 3000:
            raise ValueError("質問は3000文字以内です")
        if mode not in MODES:
            raise ValueError("検索条件が不正です")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("閾値は数値で指定してください")

        memos = load_memos()
        if not any(
            m.get("status") != "deleted"
            and m.get("vector") is not None
            for m in memos
        ):
            return search_experiment(
                query, memos, np.asarray([]),
                mode=mode, top_k=top_k,
                threshold=float(threshold) / 100,
                as_of=data.get("as_of")
            )

        vector = get_vector(query, "RETRIEVAL_QUERY")
        if vector is None:
            return JSONResponse(
                status_code=503,
                content={"error": "埋め込みAPIの取得に失敗しました"}
            )

        return search_experiment(
            query, memos, vector,
            mode=mode,
            top_k=top_k,
            threshold=float(threshold) / 100,
            as_of=data.get("as_of")
        )

    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@router.post("/api/experiment-relations")
async def classify_real_memos(request: Request):
    """Explicit opt-in. Does not write any classified relation."""
    if blocked():
        return JSONResponse(
            status_code=403,
            content={"error": "参加者実験では使用できません"}
        )

    # Disabled by default and requires administrator token.
    if os.getenv("GEMMA_RELATIONS_ENABLED") != "1":
        return JSONResponse(
            status_code=403,
            content={"error": "Gemma classification disabled"}
        )

    from hmac import compare_digest
    secret = os.getenv("GEMMA_ADMIN_TOKEN", "")
    supplied = request.headers.get("x-gemma-admin-token", "")

    if not secret or not compare_digest(secret, supplied):
        return JSONResponse(
            status_code=403,
            content={"error": "Unauthorized"}
        )

    model = os.getenv("GEMMA_RELATION_MODEL", "").strip()
    if not model:
        return JSONResponse(
            status_code=503,
            content={
                "error": "GEMMA_RELATION_MODELを設定してください"
            }
        )

    try:
        data = await request.json()
        child_id = data.get("memo_id")
        if not isinstance(child_id, str) or not child_id:
            raise ValueError("memo_idが必要です")

        memos = live_memos(load_memos())
        by_id = {str(m["id"]): m for m in memos}
        child = by_id.get(child_id)

        if not child or child.get("vector") is None:
            raise ValueError("対象メモが見つかりません")

        cv = np.asarray(child["vector"], dtype=float)
        cn = np.linalg.norm(cv)
        if not cn or not np.isfinite(cv).all():
            raise ValueError("対象メモのベクトルが不正です")

        candidates = []
        for parent in memos:
            if parent["id"] == child_id:
                continue
            if parent.get("vector") is None:
                continue
            try:
                if parse_time(parent) >= parse_time(child):
                    continue
            except ValueError:
                continue
            pv = np.asarray(parent["vector"], dtype=float)
            if pv.shape != cv.shape or not np.isfinite(pv).all():
                continue
            pn = np.linalg.norm(pv)
            if not pn:
                continue
            sim = float(np.dot(cv, pv) / (cn * pn))
            if sim >= 0.80:
                candidates.append((sim, parent))

        candidates.sort(
            key=lambda x: (-x[0], str(x[1]["id"]))
        )
        classifier = GeminiRelationClassifier(model=model, max_output_tokens=768, thinking_level="minimal", max_retries=3, base_delay=5.0, max_delay=30.0, format_retries=0)
        results = []

        # At most two external model calls per explicit request.
        for similarity, parent in candidates[:1]:
            answer = await run_in_threadpool(
                classifier.classify, parent, child
            )
            results.append({
                "parent_id": str(parent["id"]),
                "child_id": child_id,
                "parent_title": parent.get("title", ""),
                "similarity": round(similarity, 6),
                "relation": answer["relation"],
                "confidence": answer["confidence"],
                "reason": answer["reason"],
                "approved": False
            })

        return {
            "status": "success",
            "model": model,
            "candidates": results,
            "saved": False
        }

    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception:
        return JSONResponse(
            status_code=502,
            content={"error": "Gemmaによる分類に失敗しました"}
        )
