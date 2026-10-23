import networkx as nx
import numpy as np
from collections import deque
from src.bm25 import BM25
from src.config import (
    PPR_ALPHA,
    STALE_PENALTY,
)

from src.data_manager import (
    load_memos,
    live_memos,
    superseded_map,
)

from src.embedding import get_vector
from src.graph import build_graph
from src.similarity import time_align


MODES = (
    "keyword", "semantic", "graph", "time", "full",
    "thought_evolution",          # ラベルなし（類似度＋作成日時で系列を推定）
    "thought_evolution_oracle",   # 評価用の上限値（chain_id/version を使う）
)

# (semantic, graph, time) の重み
WEIGHTS = {
    "semantic": (1.0, 0.0, 0.0),
    "graph": (0.7, 0.3, 0.0),
    "time": (0.7, 0.0, 0.3),
    "full": (0.6, 0.2, 0.2),
}

LINEAGE_TAU = 0.78        # 「同じ思考の系列」とみなすコサイン類似度の下限（推定リンク）
LINEAGE_MAX_HOPS = 3
LINEAGE_ANCHORS = 5


def _thought_evolution_oracle_scores(candidates, semantic_scores, superseded_by, top_anchors=5):
    n = len(candidates)
    scores = np.zeros(n, dtype=float)
    anchor_ids = set()

    if n == 0:
        return scores, anchor_ids

    order = np.argsort(-semantic_scores)[:min(top_anchors, n)]
    anchor_indices = [int(i) for i in order if semantic_scores[int(i)] > 0.63]
    anchor_ids = {candidates[i]["id"] for i in anchor_indices}

    by_id = {m["id"]: m for m in candidates}

    def ancestor_distance(start_id, target_id, max_depth=20):
        frontier = [(start_id, 0)]
        seen = {start_id}
        while frontier:
            current, dist = frontier.pop(0)
            memo = by_id.get(current)
            if not memo:
                continue
            for old_id in memo.get("supersedes", []) or []:
                if old_id == target_id:
                    return dist + 1
                if old_id in by_id and old_id not in seen and dist + 1 < max_depth:
                    seen.add(old_id)
                    frontier.append((old_id, dist + 1))
        return None

    for i, memo in enumerate(candidates):
        if memo["id"] in anchor_ids:
            scores[i] = float(semantic_scores[i])
            continue

        best = 0.0

        for anchor_index in anchor_indices:
            anchor = candidates[anchor_index]
            anchor_score = float(semantic_scores[anchor_index])
            relation = 0.0

            if (
                memo.get("chain_id") is not None
                and memo.get("chain_id") == anchor.get("chain_id")
                and memo.get("version") is not None
                and anchor.get("version") is not None
            ):
                try:
                    delta = int(anchor["version"]) - int(memo["version"])
                except (TypeError, ValueError):
                    delta = 0
                if delta > 0:
                    relation = 0.65 / delta

            if relation == 0.0:
                dist = ancestor_distance(anchor["id"], memo["id"])
                if dist is not None:
                    relation = 0.85 / dist

            if relation > 0:
                value = relation * (0.65 + 0.35 * float(semantic_scores[i]))
                value *= 0.70 + 0.30 * anchor_score
                best = max(best, value)

        scores[i] = min(best, 1.0)

    return scores, anchor_ids


def _infer_parent_links(candidates, vectors, tau):
    """
    ラベルなしで思考のつながり（子 → より過去の親）を推定する。
    """
    n = len(candidates)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1e-10, norms)
    sim = unit @ unit.T
    created = [m["created_at"] for m in candidates]
    index_of = {m["id"]: i for i, m in enumerate(candidates)}

    parents = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and created[j] < created[i] and sim[i, j] >= tau:
                parents[i].add(j)
        for old_id in candidates[i].get("supersedes", []) or []:
            if old_id in index_of:
                parents[i].add(index_of[old_id])
    return parents


def _thought_evolution_scores(
    candidates,
    vectors,
    semantic_scores,
    top_anchors=None,
    tau=None,
    max_hops=None,
    include_anchors=False,
):
    n = len(candidates)
    scores = np.zeros(n, dtype=float)
    if n == 0:
        return scores, set()

    top_anchors = LINEAGE_ANCHORS if top_anchors is None else top_anchors
    tau = LINEAGE_TAU if tau is None else tau
    max_hops = LINEAGE_MAX_HOPS if max_hops is None else max_hops

    order = np.argsort(-semantic_scores)[:min(top_anchors, n)]
    anchors = [int(i) for i in order if semantic_scores[int(i)] > 0.63]
    anchor_ids = {candidates[i]["id"] for i in anchors}

    parents = _infer_parent_links(candidates, vectors, tau)

    for a in anchors:
        anchor_score = float(semantic_scores[a])
        hops = {a: 0}
        frontier = [a]
        for depth in range(1, max_hops + 1):
            nxt = []
            for cur in frontier:
                for p in parents[cur]:
                    if p not in hops:
                        hops[p] = depth
                        nxt.append(p)
            frontier = nxt

        for i, h in hops.items():
            if i == a:
                continue
            relation = 0.85 / h
            value = relation * (0.65 + 0.35 * float(semantic_scores[i]))
            value *= 0.70 + 0.30 * anchor_score
            scores[i] = max(scores[i], min(value, 1.0))

    if include_anchors:
        for a in anchors:
            scores[a] = max(scores[a], float(semantic_scores[a]))
    else:
        for a in anchors:
            scores[a] = 0.0

    return scores, anchor_ids


def search_core(
    query,
    mode="full",
    threshold=0.3,
    top_k=10,
    as_of=None,
    time_from=None,
    time_to=None,
    memos=None,
    query_vector=None,
    stale_penalty=None,
    use_stale_penalty=True,  # ★評価実験・ペナルティ切り替え用に追加
    graph=None,
    weights=None,
    ppr_alpha=None,
    bm25_k1=None,
    bm25_b=None,
    lineage_tau=None,
    lineage_include_anchors=True,
):
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")

    if stale_penalty is None:
        stale_penalty = STALE_PENALTY

    if ppr_alpha is None:
        ppr_alpha = PPR_ALPHA

    if weights is not None:
        if len(weights) != 3 or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError(f"weights は (semantic, graph, time) の3つの非負の値である必要があります: {weights!r}")

    if memos is None:
        memos = load_memos()

    pool = live_memos(memos, as_of)
    superseded_by = superseded_map(pool)
    candidates = [m for m in pool if "vector" in m]

    if not candidates or not query:
        return []

    dims = {len(m["vector"]) for m in candidates}
    if len(dims) > 1:
        raise ValueError(f"メモのベクトルの次元が混在しています: {sorted(dims)}。")

    vectors = np.array([m["vector"] for m in candidates])
    count = len(candidates)
    # ⬇️⬇️⬇️ ここから追加: メモ同士の自動リンク(類似度)を計算 ⬇️⬇️⬇️
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / np.where(norms == 0, 1e-10, norms)
    memo_sim_matrix = unit_vectors @ unit_vectors.T
    RELATED_TAU = 0.5  # ★意味的関連とみなす類似度の閾値 (0.7〜0.85あたりで調整)
    # ⬆️⬆️⬆️ ここまで追加 ⬆️⬆️⬆️

    # ---------- Keyword (BM25) ----------
    keyword_scores = np.zeros(count)
    if mode == "keyword":
        bm25_args = {}
        if bm25_k1 is not None: bm25_args["k1"] = bm25_k1
        if bm25_b is not None: bm25_args["b"] = bm25_b
        bm25 = BM25(**bm25_args).fit(
            f'{m.get("title", "")} {m.get("content", "")}' for m in candidates
        )
        keyword_scores = bm25.scores_normalized(query)

    # ---------- Semantic ----------
    semantic_scores = np.zeros(count)
    if mode != "keyword":
        if query_vector is None:
            query_vector = get_vector(query, "RETRIEVAL_QUERY")
        if query_vector is None:
            return None
        if len(query_vector) != vectors.shape[1]:
            raise ValueError(f"クエリ次元({len(query_vector)})とメモ次元({vectors.shape[1]})が不一致です。")

        semantic_scores = np.clip(
            vectors @ np.asarray(query_vector, dtype=float), 0, 1
        )

    # ---------- Graph (Personalized PageRank) ----------
    graph_scores = np.zeros(count)
    if mode in ("graph", "full") and semantic_scores.sum() > 0:
        if graph is None:
            graph = build_graph(candidates, vectors)

        personalization = {m["id"]: float(s) for m, s in zip(candidates, semantic_scores)}
        try:
            pagerank = nx.pagerank(
                graph, alpha=ppr_alpha, personalization=personalization, weight="weight"
            )
            values = np.array([pagerank[m["id"]] for m in candidates])
            if values.max() > 0:
                graph_scores = values / values.max()
        except Exception as e:
            print(f"PPRエラー: {e}")

    # ---------- Thought Evolution ----------
    evolution_scores = np.zeros(count)
    evolution_anchor_ids = set()

    if mode == "thought_evolution":
        evolution_scores, evolution_anchor_ids = _thought_evolution_scores(
            candidates, vectors, semantic_scores, tau=lineage_tau, include_anchors=lineage_include_anchors
        )
    elif mode == "thought_evolution_oracle":
        evolution_scores, evolution_anchor_ids = _thought_evolution_oracle_scores(
            candidates, semantic_scores, superseded_by
        )

    # ---------- 統合 ----------
    use_time = time_from is not None or time_to is not None
    results = []

    for i, memo in enumerate(candidates):
        stale = memo["id"] in superseded_by
        time_score = None

        if mode in ("time", "full") and use_time:
            time_score = time_align(memo, time_from, time_to)

        if mode == "keyword":
            score = keyword_scores[i]
        elif mode in ("thought_evolution", "thought_evolution_oracle"):
            score = evolution_scores[i]
        else:
            w_sem, w_graph, w_time = WEIGHTS[mode] if weights is None else weights
            if time_score is None:
                w_sem += w_time
                w_time = 0.0
                time_value = 0.0
            else:
                time_value = time_score

            score = w_sem * semantic_scores[i] + w_graph * graph_scores[i] + w_time * time_value

            # ★ use_stale_penalty フラグによる制御
            if mode == "full" and stale and use_stale_penalty:
                score *= stale_penalty

        if score < threshold:
            continue

        # ⬇️⬇️⬇️ ここから追加: 自分以外の類似メモを抽出 ⬇️⬇️⬇️
        related_indices = np.where(memo_sim_matrix[i] >= RELATED_TAU)[0]
        auto_related = [
            candidates[j]["id"]
            for j in related_indices if i != j
        ]
        # 元々のデータ(もしあれば)と、自動抽出したIDを重複なく合体
        final_related_ids = list(set(memo.get("related_ids", [])) | set(auto_related))
        # ⬆️⬆️⬆️ ここまで追加 ⬆️⬆️

        results.append(
            {
                "id": memo["id"],
                "date": memo["date"],
                "created_at": memo["created_at"],
                "title": memo["title"],
                "content": memo["content"],
                "event_date": memo.get("event_date"),
                "stale": stale,
                "superseded_by": superseded_by.get(memo["id"]),
                "supersedes": memo.get("supersedes", []),
                "related_ids": final_related_ids,
                "thought_evolution": mode in ("thought_evolution", "thought_evolution_oracle"),
                "evolution_anchor": memo["id"] in evolution_anchor_ids,
                "score": int(round(float(score) * 100)),
                "raw_score": float(score),
                "parts": {
                    "sem": round(float(semantic_scores[i]), 3),
                    "graph": round(float(graph_scores[i]), 3),
                    "evolution": round(float(evolution_scores[i]), 3),
                    "time": None if time_score is None else round(float(time_score), 3),
                },
            }
        )

    results.sort(key=lambda x: x["raw_score"], reverse=True)
    return results[:top_k]


# ============================================================
# 多階層（マルチホップ）可視化・探索用 API 関数
# ============================================================

def get_multi_hop_lineage(
    start_memo_id: str,
    memos=None,
    max_hops: int = 3,
    tau: float = LINEAGE_TAU,
    as_of=None,
):
    """
    指定メモ（start_memo_id）を起点とし、
    過去にさかのぼる思考の系譜を max_hops 階層まで探索して
    vis-network などの可視化用 JSON 構造 (nodes, edges) で返す。
    """
    if memos is None:
        memos = load_memos()

    pool = live_memos(memos, as_of)
    candidates = [m for m in pool if "vector" in m]
    index_of = {m["id"]: i for i, m in enumerate(candidates)}

    if start_memo_id not in index_of:
        return {"nodes": [], "edges": []}

    start_idx = index_of[start_memo_id]
    vectors = np.array([m["vector"] for m in candidates])
    parents = _infer_parent_links(candidates, vectors, tau)

    # 幅優先探索 (BFS)
    visited = {start_idx: {"depth": 0, "parent": None}}
    queue = deque([(start_idx, 0)])

    edges = []

    while queue:
        curr_idx, depth = queue.popleft()
        if depth >= max_hops:
            continue

        for p_idx in parents[curr_idx]:
            if p_idx not in visited:
                visited[p_idx] = {"depth": depth + 1, "parent": curr_idx}
                queue.append((p_idx, depth + 1))

            # エッジの作成
            edges.append(
                {
                    "from": candidates[curr_idx]["id"],
                    "to": candidates[p_idx]["id"],
                    "label": f"hop {depth + 1}",
                }
            )

    nodes = []
    for idx, info in visited.items():
        m = candidates[idx]
        nodes.append(
            {
                "id": m["id"],
                "label": m.get("title") or m.get("content", "")[:15] + "...",
                "depth": info["depth"],
                "level": info["depth"],  # vis-network の階層表示用
                "date": m.get("created_at") or m.get("date", ""),
                "group": f"depth_{info['depth']}",
            }
        )

    return {"nodes": nodes, "edges": edges}