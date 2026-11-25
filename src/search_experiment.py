"""850v3 fixed9 + revision stale experiment, for real memo data.

No gold labels, no Oracle, no API calls, no database writes.
Ranking logic transcribed from run_remaining_parameters.py,
run_gemma_joint27_850v3.py and run_gemma_stale080_850v3.py.
"""
from collections import deque, defaultdict
from datetime import date
import math
import numpy as np

from src.bm25 import BM25
from src.data_manager import live_memos, superseded_map
from src.intent_search import _relation_neighbors
from src.integrated_search import (
    resolve_search_intent, _current_version_precedence
)
from src.similarity import time_align

MODES = (
    "A_fixed9", "B_stale080", "C_precedence", "D_both"
)
VERSION = "gemma_850v3_fixed9_stale080_v1"

PARAMS = {
    "sem_w": .80,
    "event_boost": .035,
    "stale_multiplier": .92,
    "threshold": .30,
    "max_hops": 3,
    "max_anchors": 4,
    "max_visited": 32,
    "semantic_floor": .15,
    "anchor_min": .30,
    "hop_decay": .70,
    "rev_current": .10,
    "rev_change": .045,
    "rev_past": .032,
    "rev_backward": -.06,
    "dev_forward": .20,
    "dev_change_forward": .022,
    "dev_change_back": .012,
    "branch_change": .0,
    "branch_other": .007,
    "delta_min": -.06,
    "delta_max": .11,
}


def rerank(pool, adjacency, semantic, hybrid, intent, params):
    n = len(pool)
    ids = {str(m["id"]): i for i, m in enumerate(pool)}
    anchors = sorted(
        range(n), key=lambda i: (-float(hybrid[i]), i)
    )[:params["max_anchors"]]
    anchors = [
        i for i in anchors if hybrid[i] >= params["anchor_min"]
    ]
    delta = np.zeros(n, dtype=float)
    kind = intent["intent"]
    as_of = intent["as_of"]

    for ai in anchors:
        aid = str(pool[ai]["id"])
        queue = deque([(aid, 0)])
        visited = {aid}

        while queue and len(visited) < params["max_visited"]:
            cid, hop = queue.popleft()
            if hop >= params["max_hops"]:
                continue

            for nid, rel, direction, source in adjacency.get(cid, ()):
                if nid in visited or nid not in ids:
                    continue
                if len(visited) >= params["max_visited"]:
                    continue

                i = ids[nid]
                if semantic[i] < params["semantic_floor"]:
                    continue

                visited.add(nid)
                dist = hop + 1
                queue.append((nid, dist))
                factor = params["hop_decay"] ** (dist - 1)

                if rel == "revision":
                    if kind == "change":
                        adjustment = params["rev_change"] * factor
                    elif kind == "past" and as_of is None:
                        adjustment = (
                            params["rev_past"]
                            if direction == "backward" else 0
                        ) * factor
                    elif kind == "past":
                        adjustment = (
                            params["rev_past"]
                            if direction == "forward" else 0
                        ) * factor
                    else:
                        adjustment = (
                            params["rev_current"]
                            if direction == "forward"
                            else params["rev_backward"]
                        ) * factor

                elif rel == "development":
                    if kind == "past":
                        adjustment = 0.0
                    elif direction == "forward":
                        adjustment = (
                            params["dev_change_forward"]
                            if kind == "change"
                            else params["dev_forward"]
                        ) * factor
                    else:
                        adjustment = (
                            params["dev_change_back"]
                            if kind == "change" else 0.0
                        ) * factor

                elif rel == "branch":
                    adjustment = (
                        params["branch_change"]
                        if kind == "change"
                        else params["branch_other"]
                    ) * factor
                else:
                    raise ValueError("未対応の関係種類: " + str(rel))

                if adjustment > 0:
                    delta[i] = max(delta[i], adjustment)
                elif adjustment < 0:
                    delta[i] = min(delta[i], adjustment)

    return np.clip(
        hybrid + np.clip(
            delta, params["delta_min"], params["delta_max"]
        ), 0, 1
    )


def revision_mapping(pool, adjacency):
    by_id = {str(m["id"]): m for m in pool}
    children = defaultdict(set)

    for parent_id, neighbors in adjacency.items():
        if parent_id not in by_id:
            continue
        for child_id, relation, direction, source in neighbors:
            if (
                relation == "revision"
                and direction == "forward"
                and child_id in by_id
                and child_id != parent_id
            ):
                children[parent_id].add(child_id)

    successors = {}
    for parent_id, child_ids in children.items():
        successors[parent_id] = max(
            child_ids,
            key=lambda mid: (
                str(by_id[mid].get("created_at") or ""), mid
            ),
        )
    stale = np.asarray(
        [str(m["id"]) in successors for m in pool], dtype=bool
    )
    return successors, stale


def search_experiment(
    query, memos, query_vector, *,
    mode="B_stale080", as_of=None,
    top_k=10, threshold=.30
):
    if mode not in MODES:
        raise ValueError("未対応の検索条件です")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 30:
        raise ValueError("top_kは1～30の整数です")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("thresholdは0～1です")

    intent = resolve_search_intent(query, as_of=as_of)
    cutoff = (
        date.fromisoformat(intent["as_of"])
        if intent["as_of"] else None
    )
    pool = [
        m for m in live_memos(memos, cutoff)
        if m.get("vector") is not None
    ]

    if not pool:
        return {
            "query": query, "classification": intent,
            "mode": mode, "scoring": VERSION,
            "results": [], "total_candidates": 0
        }

    matrix = np.asarray(
        [m["vector"] for m in pool], dtype=float
    )
    qv = np.asarray(query_vector, dtype=float)

    if matrix.ndim != 2 or qv.shape != (matrix.shape[1],):
        raise ValueError("メモと質問のベクトル次元が不一致です")
    if not np.isfinite(matrix).all() or not np.isfinite(qv).all():
        raise ValueError("ベクトルに不正な値があります")

    ids = [str(m["id"]) for m in pool]
    semantic = np.clip(matrix @ qv, 0, 1)
    lexical = BM25().fit(
        f"{m.get('title', '')} {m.get('content', '')}"
        for m in pool
    ).scores_normalized(query)

    p = dict(PARAMS)
    p["threshold"] = float(threshold)

    hybrid = p["sem_w"] * semantic + (1 - p["sem_w"]) * lexical
    adjacency = _relation_neighbors(pool)

    related = (
        rerank(pool, adjacency, semantic, hybrid, intent, p)
        if adjacency else hybrid.copy()
    )

    target = intent["event_target_date"]
    event = (
        np.asarray([
            float(time_align(m, target, target) or 0.0)
            for m in pool
        ], dtype=float)
        if target else np.zeros(len(pool), dtype=float)
    )
    scores = np.clip(related + p["event_boost"] * event, 0, 1)

    superseded = superseded_map(pool)
    stale = np.asarray(
        [mid in superseded for mid in ids], dtype=bool
    )
    multipliers = np.ones(len(pool), dtype=float)

    if intent["intent"] == "current":
        if mode != "A_fixed9":
            successors, inferred_stale = revision_mapping(
                pool, adjacency
            )
            if mode in ("B_stale080", "D_both"):
                stale = inferred_stale
                p["stale_multiplier"] = .80
            else:
                stale = np.zeros(len(pool), dtype=bool)
                p["stale_multiplier"] = 1.0

            superseded = (
                successors
                if mode in ("C_precedence", "D_both")
                else {}
            )

        multipliers = np.where(
            stale, p["stale_multiplier"], 1.0
        )
        scores = np.clip(scores * multipliers, 0, 1)

        if target is None:
            scores, _ = _current_version_precedence(
                pool, semantic, scores,
                superseded, p["threshold"]
            )

    order = np.argsort(-scores, kind="stable")
    selected = [
        int(i) for i in order
        if scores[i] >= p["threshold"]
    ][:top_k]

    results = []
    for i in selected:
        m = pool[i]
        results.append({
            "id": ids[i],
            "title": m.get("title", ""),
            "content": m.get("content", ""),
            "date": m.get("date", ""),
            "created_at": m.get("created_at"),
            "event_date": m.get("event_date"),
            "supersedes": m.get("supersedes", []),
            "stale": bool(stale[i]),
            "superseded_by": superseded.get(ids[i]),
            "related_ids": [],
            "score": int(round(float(scores[i]) * 100)),
            "raw_score": float(scores[i]),
            "parts": {
                "sem": round(float(semantic[i]), 4),
                "bm25": round(float(lexical[i]), 4),
                "hybrid": round(float(hybrid[i]), 4),
                "lineage_delta": round(
                    float(related[i] - hybrid[i]), 4
                ),
                "event_time_bonus": round(
                    float(p["event_boost"] * event[i]), 4
                ),
                "stale_multiplier": float(multipliers[i])
            },
            "lineage_intent": intent["intent"],
            "relation_source_policy": "saved",
            "search_scoring": VERSION
        })

    return {
        "query": query,
        "classification": intent,
        "mode": mode,
        "scoring": VERSION,
        "total_candidates": len(pool),
        "results": results
    }
