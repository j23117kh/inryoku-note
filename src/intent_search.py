"""Production-safe candidate Hybrid + bounded stored-relationship search.

Distinct from the v4 *experimental* search method. No oracle, answer labels,
question type metadata, chain_id/version, or learned parameters are accessed.
The ranking never builds an N×N memo similarity matrix. Existing Web modes are
unmodified; this opt-in route does not modify the stored memos.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
import math
import numpy as np

from src.bm25 import BM25
from src.data_manager import live_memos, superseded_map
from src.query_intent import classify_query

MAX_HOPS = 3
MAX_ANCHORS = 7
MAX_VISITED = 32
HYBRID_SEM_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3
RELEVANT_SEM_FLOOR = 0.15


def _relation_neighbors(memos):
    """Build bounded traversal graph from saved links; related is non-ancestral."""
    from src.data_manager import parse_created_date
    by_id = {str(m['id']): m for m in memos}
    edges = {}
    for child in memos:
        cid = str(child['id'])
        cday = parse_created_date(child.get('created_at'))
        def add(pid, relation, source):
            pid = str(pid)
            parent = by_id.get(pid)
            if parent is None or pid == cid:
                return
            pday = parse_created_date(parent.get('created_at'))
            if pday and cday and pday > cday:
                return
            old = edges.get((pid, cid))
            if old is not None and old[0] == 'revision' and relation != 'revision':
                return  # Supersedes cannot become a mere branch accidentally.
            if relation in ('revision', 'development', 'branch'):
                edges[(pid, cid)] = (relation, source)
        for pid in child.get('supersedes') or []:
            if isinstance(pid, str):
                add(pid, 'revision', 'stored_update')
        for link in child.get('thought_links') or []:
            if isinstance(link, dict) and link.get('relation') in ('revision', 'development', 'branch'):
                pid = link.get('parent_id')
                if isinstance(pid, str):
                    add(pid, link['relation'], 'stored_thought_link')
    adj = defaultdict(list)
    for (p, c), (rel, source) in sorted(edges.items()):
        adj[p].append((c, rel, 'forward', source))
        adj[c].append((p, rel, 'backward', source))
    return adj


def rerank_stored(memos, hybrid, semantic, intent, *, as_of=None):
    """Bounded rank adjustment. A saved path may expand 3 hops, never guessed.

    A small score delta is applied to neighbors of top Hybrid anchors. For
    repeated paths the strongest single positive/negative delta wins, preventing
    link-count inflation. Only supported memo relationships are used.
    """
    n = len(memos)
    hybrid = np.asarray(hybrid, dtype=float)
    semantic = np.asarray(semantic, dtype=float)
    if hybrid.shape != (n,) or semantic.shape != (n,):
        raise ValueError('スコアの件数がメモ数と一致しません')
    if n == 0:
        return hybrid.copy(), {}
    ids = {str(m['id']): i for i, m in enumerate(memos)}
    adjacency = _relation_neighbors(memos)
    anchors = sorted(range(n), key=lambda i: (-float(hybrid[i]), i))[:MAX_ANCHORS]
    anchors = [i for i in anchors if hybrid[i] >= 0.3]
    delta = np.zeros(n, dtype=float)
    evidence = defaultdict(list)
    for anchor_idx in anchors:
        anchor_id = str(memos[anchor_idx]['id'])
        queue = deque([(anchor_id, 0)])
        visited = {anchor_id}
        while queue and len(visited) < MAX_VISITED:
            cid, hop = queue.popleft()
            if hop >= MAX_HOPS:
                continue
            for nid, relation, direction, source in adjacency.get(cid, ()): 
                if nid in visited or nid not in ids or len(visited) >= MAX_VISITED:
                    continue
                i = ids[nid]
                if semantic[i] < RELEVANT_SEM_FLOOR:
                    continue
                visited.add(nid)
                distance = hop + 1
                queue.append((nid, distance))
                factor = 0.7 ** (distance - 1)
                if relation == 'revision':
                    if intent == 'change':
                        adjustment = 0.045 * factor
                    elif intent == 'past' and as_of is None:
                        adjustment = (0.032 if direction == 'backward' else 0.0) * factor
                    elif intent == 'past':
                        adjustment = (0.032 if direction == 'forward' else 0.0) * factor
                    else:
                        adjustment = (0.055 if direction == 'forward' else -0.01) * factor
                elif relation == 'development':
                    adjustment = (0.022 if direction == 'forward' else (0.012 if intent == 'change' else 0)) * factor
                else:
                    adjustment = (0.014 if intent == 'change' else 0.007) * factor
                if adjustment > 0:
                    delta[i] = max(delta[i], adjustment)
                elif adjustment < 0:
                    delta[i] = min(delta[i], adjustment)
                evidence[i].append({'anchor_id': anchor_id, 'hops': distance,
                                    'relation': relation, 'direction': direction,
                                    'source': source})
    return np.clip(hybrid + np.clip(delta, -0.06, 0.08), 0, 1), dict(evidence)


def search_question(query: str, *, memos, query_vector, as_of=None,
                    top_k=10, threshold=0.30) -> dict:
    """Pure search function for testability, with precomputed query vector.

    The caller must run classify_query and pass its parsed as_of date string.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 30:
        raise ValueError('top_k は1～30の整数にしてください')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold は0～1の数値にしてください')
    cutoff = date.fromisoformat(as_of) if as_of else None
    pool = live_memos(memos, cutoff)
    candidates = [m for m in pool if m.get('vector') is not None]
    if not candidates:
        return {'results': [], 'total_candidates': 0, 'scoring': 'hybrid_saved_3hop_v1'}
    dims = {len(m['vector']) for m in candidates}
    if len(dims) > 1:
        raise ValueError('保存メモのベクトル次元が混在しています')
    matrix = np.asarray([m['vector'] for m in candidates], dtype=float)
    qvector = np.asarray(query_vector, dtype=float)
    if qvector.shape != (matrix.shape[1],) or not np.isfinite(qvector).all():
        raise ValueError('クエリとメモのベクトル次元が一致しません')
    if not np.isfinite(matrix).all():
        raise ValueError('メモのベクトルに不正値があります')
    # Legacy scoring parity: clip cosine/dot scores identically to search_core.
    semantic = np.clip(matrix @ qvector, 0, 1)
    bm25 = BM25().fit(f"{m.get('title', '')} {m.get('content', '')}" for m in candidates)
    lexical = bm25.scores_normalized(query)
    hybrid = HYBRID_SEM_WEIGHT * semantic + HYBRID_BM25_WEIGHT * lexical
    detected = classify_query(query, as_of=as_of)
    improved, evidence = rerank_stored(candidates, hybrid, semantic, detected['intent'], as_of=as_of)
    superseded = superseded_map(pool)
    order = sorted(range(len(candidates)), key=lambda i: (-float(improved[i]), i))
    selected = [i for i in order if improved[i] >= threshold][:top_k]
    results = []
    for i in selected:
        memo = candidates[i]
        m_id = str(memo['id'])
        results.append({
            'id': m_id, 'title': memo.get('title',''), 'content': memo.get('content',''),
            'date': memo.get('date',''), 'created_at': memo.get('created_at',''),
            'event_date': memo.get('event_date'), 'stale': m_id in superseded,
            'supersedes': memo.get('supersedes', []), 'superseded_by': superseded.get(m_id),
            'related_ids': [], 'score': int(round(float(improved[i])*100)),
            'raw_score': float(improved[i]),
            'parts': {'sem': round(float(semantic[i]), 3), 'bm25': round(float(lexical[i]),3),
                      'hybrid': round(float(hybrid[i]),3), 'lineage_delta': round(float(improved[i]-hybrid[i]),3)},
            'thought_path_evidence': evidence.get(i, []),
            'lineage_intent': detected['intent'], 'relation_source_policy': 'saved_only',
        })
    return {'results': results, 'total_candidates': len(candidates),
            'scoring': 'hybrid_saved_3hop_v1'}


def classify_and_search(query: str, *, memos, query_vector, as_of=None,
                        top_k=10, threshold=.30) -> dict:
    classified = classify_query(query, as_of=as_of)
    found = search_question(query, memos=memos, query_vector=query_vector,
                            as_of=classified['as_of'], top_k=top_k, threshold=threshold)
    return {'query': query, 'classification': classified, **found}
