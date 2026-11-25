"""Production UI adapter for verified experiment B ranking."""
import math
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.data_manager import load_memos
from src.embedding import get_vector
from src.search_experiment import search_experiment
from src.search_policy import (
    APP_DEFAULT_THRESHOLD_PERCENT, APP_DEFAULT_TOP_K
)

router = APIRouter()


async def _search(request, array_response=False):
    if (os.getenv("THESIS_STUDY_SEARCH_LOCK") == "1"
            and os.getenv("THESIS_STUDY_ALLOW_PROPOSAL_SEARCH") != "1"):
        return JSONResponse(
            status_code=403,
            content={"error": "Study session is locked"}
        )

    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("JSON object required")

        query = data.get("text", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("???????????")
        query = query.strip()
        if len(query) > 3000:
            raise ValueError("????3000??????")

        threshold = data.get(
            "threshold", APP_DEFAULT_THRESHOLD_PERCENT
        )
        top_k = data.get("top_k", APP_DEFAULT_TOP_K)

        if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float)
        ):
            raise ValueError("???????")
        threshold = float(threshold)

        if not math.isfinite(threshold) or not 0 <= threshold <= 100:
            raise ValueError("???0?100??")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k?????")
        if not 1 <= top_k <= 30:
            raise ValueError("top_k?1?30??")

        memos = load_memos()
        has_vectors = any(
            m.get("status") != "deleted"
            and m.get("vector") is not None
            for m in memos
        )

        if has_vectors:
            vector = get_vector(query, "RETRIEVAL_QUERY")
            if vector is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Embedding API unavailable"}
                )
        else:
            vector = []

        result = search_experiment(
            query, memos, vector,
            mode="B_stale080",
            as_of=data.get("as_of"),
            top_k=top_k,
            threshold=threshold / 100.0
        )

        return result["results"] if array_response else result

    except (ValueError, TypeError) as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)}
        )


@router.post("/api/proposal-gravity")
async def proposal_gravity(request: Request):
    # Same list response shape as the original /api/gravity.
    return await _search(request, array_response=True)


@router.post("/api/proposal-search")
async def proposal_search(request: Request):
    # Same object response shape as /api/integrated-search.
    return await _search(request, array_response=False)


# AUTO_GEMMA_ON_SAVE_V1
