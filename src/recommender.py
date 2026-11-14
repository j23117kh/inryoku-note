import networkx as nx
import numpy as np
from collections import deque
from src.bm25 import BM25
from datetime import datetime, timezone
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
from src.semantic_lineage import rerank as rerank_semantic_lineage


# ============================================================
# ハイパーパラメータのデフォルト値 (実験での探索・調整対象)
# ============================================================
DEFAULT_RELATED_TAU = 0.5         # 自動リンクの類似度閾値
DEFAULT_ANCHOR_THRESHOLD = 0.63   # Anchorとする類似度の下限
DEFAULT_VERSION_DECAY = 0.65      # oracleにおけるバージョン差分の減衰率
DEFAULT_DISTANCE_DECAY = 0.85     # グラフ上のホップ数（距離）の減衰率
DEFAULT_SEM_BASE = 0.65           # 意味類似度のベース重み
DEFAULT_SEM_WEIGHT = 0.35         # 意味類似度の追加重み
DEFAULT_ANC_BASE = 0.70           # Anchorスコアのベース重み
DEFAULT_ANC_WEIGHT = 0.30         # Anchorスコアの追加重み


MODES = (
    "keyword", "semantic", "graph", "time", "full",
    "semantic_lineage",        # 主提案: Semantic候補維持＋関係再順位付け
    "thought_evolution",          # ラベルなし（類似度＋作成日時で系列を推定）
    "thought_evolution_oracle",   # 評価用の上限値（chain_id/version を使う）
    "thought_approved",          # 明示的に承認された関係のみを使う推薦
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






def _parse_created_at(memo):
    value = memo.get("created_at") or memo.get("date")
    if not value:
        return None

    try:
        if not isinstance(value, str):
            return None

        dt = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except (TypeError, ValueError):
        return None


def _thought_evolution_oracle_scores(
    candidates, semantic_scores, superseded_by, top_anchors=5,
    anchor_threshold=DEFAULT_ANCHOR_THRESHOLD,
    version_decay=DEFAULT_VERSION_DECAY,
    distance_decay=DEFAULT_DISTANCE_DECAY,
    sem_base=DEFAULT_SEM_BASE,
    sem_weight=DEFAULT_SEM_WEIGHT,
    anc_base=DEFAULT_ANC_BASE,
    anc_weight=DEFAULT_ANC_WEIGHT
):
    n = len(candidates)
    scores = np.zeros(n, dtype=float)
    anchor_ids = set()

    if n == 0:
        return scores, anchor_ids

    order = np.argsort(-semantic_scores)[:min(top_anchors, n)]
    anchor_indices = [int(i) for i in order if semantic_scores[int(i)] > anchor_threshold]
    anchor_ids = {candidates[i]["id"] for i in anchor_indices}

    def update_distance(start_id, target_id, max_depth=20):
        """superseded_by（旧メモID → 更新後メモID）をたどって更新距離を返す。"""
        current_id = start_id
        seen = {start_id}

        for distance in range(1, max_depth + 1):
            next_id = superseded_by.get(current_id)
            if next_id is None:
                return None
            if next_id == target_id:
                return distance
            if next_id in seen:
                return None

            seen.add(next_id)
            current_id = next_id

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
                    relation = version_decay / delta  # 【修正箇所1】

            if relation == 0.0:
                dist = update_distance(memo["id"], anchor["id"])
                if dist is not None:
                    relation = distance_decay / dist  # 【修正箇所2】

            if relation > 0:
                # 【修正箇所3, 4】
                value = relation * (sem_base + sem_weight * float(semantic_scores[i]))
                value *= anc_base + anc_weight * anchor_score
                best = max(best, value)

        scores[i] = min(best, 1.0)

    return scores, anchor_ids


def _infer_parent_links(candidates, vectors, tau=None, max_parents=None, similarity_matrix=None):
    """
    ラベルなしで思考のつながり（子 → より過去の親）を推定する。
    系列推定の類似度閾値は、ベクトル次元にかかわらず LINEAGE_TAU に固定する。
    親ノード数の上限は、従来どおりベクトル次元に応じて設定する。
    """
    n = len(candidates)
    if n == 0:
        return []

    vectors = np.asarray(vectors)
    dim = vectors.shape[1] if vectors.ndim > 1 else 256

    # 次元数による比較条件の変化を避け、閾値を固定する。
    tau = LINEAGE_TAU if tau is None else tau

    if max_parents is None:
        # ハブ化防止の次数制限は従来の次元別設定を維持する。
        max_parents = 2 if dim >= 3072 else (3 if dim >= 768 else 4)

    if similarity_matrix is None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit = vectors / np.where(norms == 0, 1e-10, norms)
        sim = unit @ unit.T
    else:
        sim = np.asarray(similarity_matrix)
        if sim.shape != (n, n):
            raise ValueError("親候補推定の類似度行列の形状が不正です")

    # 日時の厳密な前後関係を整数の順位に変換。
    # Python datetime の比較規則を保ち、同日時は親にしない。
    created = [_parse_created_at(m) for m in candidates]
    date_ranks = {dt: rank for rank, dt in enumerate(sorted({d for d in created if d is not None}))}
    ranks = np.asarray([date_ranks.get(dt, -1) for dt in created], dtype=np.int64)

    parents = [set() for _ in range(n)]
    if max_parents <= 0:
        return parents
    for i in range(n):
        if ranks[i] < 0:
            continue
        # strict < を使い、同時刻のメモと自分自身は除外。
        older = (ranks >= 0) & (ranks < ranks[i])
        valid_indices = np.flatnonzero(older & (sim[i] >= tau))
        if valid_indices.size == 0:
            continue
        # Python の安定ソートと同じ: 類似度が同点なら元のメモ順。
        order = np.argsort(-sim[i, valid_indices], kind="stable")
        parents[i].update(int(j) for j in valid_indices[order[:max_parents]])

    return parents


def _thought_evolution_scores(
    candidates,
    vectors,
    semantic_scores,
    top_anchors=None,
    tau=None,
    max_hops=None,
    include_anchors=False,
    similarity_matrix=None,
    # 【追加】評価実験で外部から調整するための引数
    anchor_threshold=DEFAULT_ANCHOR_THRESHOLD,
    distance_decay=DEFAULT_DISTANCE_DECAY,
    sem_base=DEFAULT_SEM_BASE,
    sem_weight=DEFAULT_SEM_WEIGHT,
    anc_base=DEFAULT_ANC_BASE,
    anc_weight=DEFAULT_ANC_WEIGHT
):
    n = len(candidates)
    scores = np.zeros(n, dtype=float)
    if n == 0:
        return scores, set()

    top_anchors = LINEAGE_ANCHORS if top_anchors is None else top_anchors
    tau = LINEAGE_TAU if tau is None else tau
    max_hops = LINEAGE_MAX_HOPS if max_hops is None else max_hops

    order = np.argsort(-semantic_scores)[:min(top_anchors, n)]
    # 【修正箇所1】 0.63 -> anchor_threshold
    anchors = [int(i) for i in order if semantic_scores[int(i)] > anchor_threshold]
    anchor_ids = {candidates[i]["id"] for i in anchors}

    parents = _infer_parent_links(candidates, vectors, tau=tau, similarity_matrix=similarity_matrix)

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
            # 【修正箇所2】 0.85 -> distance_decay
            relation = distance_decay / h
            
            # 【修正箇所3, 4】 0.65 -> sem_base, 0.35 -> sem_weight
            value = relation * (sem_base + sem_weight * float(semantic_scores[i]))
            
            # 【修正箇所5, 6】 0.70 -> anc_base, 0.30 -> anc_weight
            value *= anc_base + anc_weight * anchor_score
            
            scores[i] = max(scores[i], min(value, 1.0))

    if include_anchors:
        for a in anchors:
            scores[a] = max(scores[a], float(semantic_scores[a]))
    else:
        for a in anchors:
            scores[a] = 0.0

    return scores, anchor_ids



def _approved_lineage_scores(
    candidates,
    semantic_scores,
    *,
    top_anchors=LINEAGE_ANCHORS,
    max_hops=LINEAGE_MAX_HOPS,
    include_anchors=True,
    anchor_threshold=DEFAULT_ANCHOR_THRESHOLD,
    distance_decay=DEFAULT_DISTANCE_DECAY,
    sem_base=DEFAULT_SEM_BASE,
    sem_weight=DEFAULT_SEM_WEIGHT,
    anc_base=DEFAULT_ANC_BASE,
    anc_weight=DEFAULT_ANC_WEIGHT,
):
    """User-approved directed links only. Never infer from labels/similarity.

    Follow child -> parent (past-directed) on revision/development/branch.
    'related' is intentionally excluded: it does not assert ancestry.
    Returns scores, anchor ids, and transparent provenance paths.
    """
    n = len(candidates)
    scores = np.zeros(n, dtype=float)
    paths = {}
    if not n:
        return scores, set(), paths
    id_to_index = {m["id"]: i for i, m in enumerate(candidates)}
    order = sorted(range(n), key=lambda i: (-float(semantic_scores[i]), str(candidates[i]["id"])))
    anchors = [i for i in order[:min(top_anchors, n)]
               if float(semantic_scores[i]) > anchor_threshold]
    anchor_ids = {candidates[i]["id"] for i in anchors}

    parents = [[] for _ in candidates]
    for child_i, child in enumerate(candidates):
        for link in child.get("thought_links") or []:
            if not isinstance(link, dict):
                continue
            relation = link.get("relation")
            parent_id = link.get("parent_id")
            if relation not in ("revision", "development", "branch"):
                continue
            parent_i = id_to_index.get(parent_id)
            if parent_i is None or parent_i == child_i:
                continue
            # Defensive check for corrupt/imported data that reverses time.
            parent_date = _parse_created_at(candidates[parent_i])
            child_date = _parse_created_at(child)
            if parent_date is not None and child_date is not None and parent_date > child_date:
                continue
            parents[child_i].append((parent_i, relation))
        parents[child_i].sort(key=lambda pair: (str(candidates[pair[0]]["id"]), pair[1]))

    for anchor_i in anchors:
        anchor_id = candidates[anchor_i]["id"]
        anchor_score = float(semantic_scores[anchor_i])
        visited = {anchor_i}
        queue = deque([(anchor_i, 0, [anchor_id], [])])
        while queue:
            node_i, hops, node_path, rel_path = queue.popleft()
            if hops >= max_hops:
                continue
            for parent_i, relation in parents[node_i]:
                if parent_i in visited:
                    continue
                visited.add(parent_i)
                next_hops = hops + 1
                path = node_path + [candidates[parent_i]["id"]]
                relations = rel_path + [relation]
                queue.append((parent_i, next_hops, path, relations))
                relation_score = distance_decay / next_hops
                value = relation_score * (
                    sem_base + sem_weight * float(semantic_scores[parent_i])
                ) * (anc_base + anc_weight * anchor_score)
                value = min(1.0, float(value))
                if value > scores[parent_i]:
                    scores[parent_i] = value
                    paths[candidates[parent_i]["id"]] = {
                        "anchor_id": anchor_id,
                        "memo_ids": path,
                        "relations": relations,
                        "hops": next_hops,
                    }

    if include_anchors:
        for i in anchors:
            if float(semantic_scores[i]) >= scores[i]:
                scores[i] = float(semantic_scores[i])
                paths.pop(candidates[i]["id"], None)
    else:
        for i in anchors:
            scores[i] = 0.0
            paths.pop(candidates[i]["id"], None)

    return scores, anchor_ids, paths


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
    # 【追加】評価実験で外から調整するための引数
    related_tau=DEFAULT_RELATED_TAU,
    anchor_threshold=DEFAULT_ANCHOR_THRESHOLD,
    version_decay=DEFAULT_VERSION_DECAY,
    distance_decay=DEFAULT_DISTANCE_DECAY,
    sem_base=DEFAULT_SEM_BASE,
    sem_weight=DEFAULT_SEM_WEIGHT,
    anc_base=DEFAULT_ANC_BASE,
    anc_weight=DEFAULT_ANC_WEIGHT,
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

    # 検索対象として利用できるメモの更新関係だけを使う。
    # as_of が指定されている場合、pool はその時点までのメモに限定される。
    pool_ids = {m["id"] for m in pool}

    superseded_by = superseded_map(pool)

    # 更新先が検索対象に存在しない場合は、古いメモとして扱わない。
    superseded_by = {
        old_id: new_id
        for old_id, new_id in superseded_by.items()
        if old_id in pool_ids and new_id in pool_ids
    }

    candidates = [m for m in pool if "vector" in m]

    if not candidates or not query:
        return []

    dims = {len(m["vector"]) for m in candidates}
    if len(dims) > 1:
        raise ValueError(f"メモのベクトルの次元が混在しています: {sorted(dims)}。")

    vectors = np.array([m["vector"] for m in candidates])
    count = len(candidates)
    
    # 全メモ間の類似度行列は必要な方式だけで作成する。
    # semantic_lineage の再順位付けは全行列を使用するため必ず維持する。
    # thought_evolution は親候補推定と関連メモ抽出で行列を共有する。
    memo_sim_matrix = None
    unit_vectors = None
    if mode in ("semantic_lineage", "thought_evolution"):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / np.where(norms == 0, 1e-10, norms)
        memo_sim_matrix = unit_vectors @ unit_vectors.T

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
            raise ValueError(f"クエリ次元({len(query_vector)})とメモ次元({vectors.shape[1]})が一致しません。")

        semantic_scores = np.clip(
            vectors @ np.asarray(query_vector, dtype=float), 0, 1
        )

    # ---------- Graph (Personalized PageRank) ----------
    graph_scores = np.zeros(count)

    if mode in ("graph", "full") and semantic_scores.sum() > 0:
        candidate_ids = {m["id"] for m in candidates}

        if graph is None:
            graph = build_graph(candidates, vectors)
        else:
            graph = graph.subgraph(candidate_ids).copy()

        personalization = {
            m["id"]: float(s)
            for m, s in zip(candidates, semantic_scores)
        }

        try:
            # 候補ノードだけで PageRank を計算する
            pagerank = nx.pagerank(
                graph,
                alpha=ppr_alpha,
                personalization=personalization,
                weight="weight",
            )

            values = np.array([
                pagerank.get(m["id"], 0.0)
                for m in candidates
            ])

            if values.max() > 0:
                graph_scores = values / values.max()

        except Exception as e:
            print(f"PPRエラー: {e}")

    # ---------- Thought Evolution ----------
    evolution_scores = np.zeros(count)
    evolution_anchor_ids = set()
    approved_paths = {}
    lineage_scores = np.zeros(count, dtype=float)
    lineage_extra_eligible = set()
    lineage_reasons = {}
    lineage_delta = np.zeros(count, dtype=float)
    lineage_intent = None
    if mode == "semantic_lineage":
        lineage_scores, lineage_extra_eligible, lineage_reasons, lineage_delta, lineage_intent = (
            rerank_semantic_lineage(
                candidates, semantic_scores, memo_sim_matrix,
                threshold=threshold, query=query,
                time_from=time_from, time_to=time_to, time_align=time_align,
            )
        )

    if mode == "thought_evolution":
        evolution_scores, evolution_anchor_ids = _thought_evolution_scores(
            candidates, vectors, semantic_scores, tau=lineage_tau, include_anchors=lineage_include_anchors,
            anchor_threshold=anchor_threshold, distance_decay=distance_decay,
            similarity_matrix=memo_sim_matrix,
            sem_base=sem_base, sem_weight=sem_weight, anc_base=anc_base, anc_weight=anc_weight # 【追加】
        )
    elif mode == "thought_evolution_oracle":
        evolution_scores, evolution_anchor_ids = _thought_evolution_oracle_scores(
            candidates, semantic_scores, superseded_by,
            anchor_threshold=anchor_threshold, version_decay=version_decay, distance_decay=distance_decay,
            sem_base=sem_base, sem_weight=sem_weight, anc_base=anc_base, anc_weight=anc_weight # 【追加】
        )
    elif mode == "thought_approved":
        evolution_scores, evolution_anchor_ids, approved_paths = _approved_lineage_scores(
            candidates, semantic_scores, include_anchors=lineage_include_anchors,
            anchor_threshold=anchor_threshold, distance_decay=distance_decay,
            sem_base=sem_base, sem_weight=sem_weight, anc_base=anc_base, anc_weight=anc_weight,
        )

    
    # ---------- 統合 ----------
    use_time = time_from is not None or time_to is not None
    results = []

    for i, memo in enumerate(candidates):
        stale = memo["id"] in superseded_by
        time_score = None

        if mode in ("time", "full") and use_time:
            time_score = time_align(memo, time_from, time_to)

        # ---------- モード別スコア計算 ----------
        if mode == "keyword":
            score = keyword_scores[i]

        elif mode == "semantic_lineage":
            score = lineage_scores[i]

        elif mode in ("thought_evolution", "thought_evolution_oracle", "thought_approved"):
            score = evolution_scores[i]

        else:
            # 共通の統合処理 (semantic, graph, time, full)
            base_w_sem, base_w_graph, base_w_time = (
                WEIGHTS[mode] if weights is None else weights
            )

            # メモごとに重みを初期化する
            w_sem = base_w_sem
            w_graph = base_w_graph
            w_time = base_w_time

            if time_score is None:
                # 時間スコアがない場合は、その重みを意味類似度へ移す
                w_sem += w_time
                w_time = 0.0
                time_value = 0.0
            else:
                time_value = time_score

            # 重み付き合算スコア
            score = (
                w_sem * semantic_scores[i]
                + w_graph * graph_scores[i]
                + w_time * time_value
            )

        # ---------- 古いメモへのペナルティ ----------
        if (
            mode in ("full", "thought_evolution", "thought_evolution_oracle", "thought_approved")
            and stale
            and use_stale_penalty
        ):
            score *= stale_penalty

        # ---------- 閾値による除外 ----------
        if mode == "semantic_lineage":
            # Keep ALL baseline Semantic-eligible candidates even when
            # confirmed revisions reduce their final score below threshold.
            if semantic_scores[i] < threshold and i not in lineage_extra_eligible:
                continue
        elif score < threshold:
            continue

        # ---------- 検索結果の作成 ----------
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
                "related_ids": [],  # 上位確定後に作る
                "_memo_index": i,   # 返却前に削除する内部キー
                "thought_evolution": mode in (
                    "thought_evolution",
                    "thought_evolution_oracle",
                    "thought_approved",
                ),
                "evolution_anchor": memo["id"] in evolution_anchor_ids,
                "approved_lineage": mode == "thought_approved",
                "approved_path": approved_paths.get(memo["id"]) if mode == "thought_approved" else None,
                "lineage_intent": lineage_intent if mode == "semantic_lineage" else None,
                "lineage_evidence": lineage_reasons.get(i, []) if mode == "semantic_lineage" else [],
                "lineage_delta": float(lineage_delta[i]) if mode == "semantic_lineage" else 0.0,
                "score": int(round(float(score) * 100)),
                "raw_score": float(score),
                "parts": {
                    "sem": round(float(semantic_scores[i]), 3),
                    "graph": round(float(graph_scores[i]), 3),
                    "evolution": round(float(evolution_scores[i]), 3),
                    "lineage_delta": round(float(lineage_delta[i]), 3) if mode == "semantic_lineage" else 0.0,
                    "time": (
                        None
                        if time_score is None
                        else round(float(time_score), 3)
                    ),
                },
            }
        )

    # ---------- スコア順に並び替え ----------
    results.sort(
        key=lambda x: x["raw_score"],
        reverse=True
    )

    selected = results[:top_k]
    if not selected:
        return selected

    # 順位確定後、表示対象に限って従来と同じ関連IDを計算する。
    selected_indices = np.asarray([item.pop("_memo_index") for item in selected], dtype=int)
    if memo_sim_matrix is not None:
        selected_rows = memo_sim_matrix[selected_indices]
    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / np.where(norms == 0, 1e-10, norms)
        selected_rows = unit_vectors[selected_indices] @ unit_vectors.T

    for result, idx, row in zip(selected, selected_indices, selected_rows):
        auto_related = [
            candidates[int(j)]["id"]
            for j in np.where(row >= related_tau)[0]
            if int(j) != int(idx)
        ]
        existing_related = candidates[int(idx)].get("related_ids") or []
        result["related_ids"] = list(dict.fromkeys(list(existing_related) + auto_related))

    return selected



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
