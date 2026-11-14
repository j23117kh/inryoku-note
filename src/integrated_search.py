"""Question-aware Hybrid + approved thought links + temporal/stale handling.

This *new* live search option is distinct from all previous experimental modes.
It reads only memo text, created_at/event_date, approved thought_links and
supersedes; it never reads oracle annotations, gold labels or query.type.

Retains a fast O(Nd + corpus-text + saved-edges) candidate stage and bounded
3-hop relationship exploration. No N^2 memo similarity graph.
"""
from __future__ import annotations

from datetime import date
import math
import re
from typing import Any

import numpy as np

from src.bm25 import BM25
from src.data_manager import live_memos, superseded_map
from src.intent_search import rerank_stored
from src.query_intent import classify_query
from src.similarity import time_align

SEM_W = 0.7
LEX_W = 0.3
STALE_MULTIPLIER = 0.92  # retained from pilot v1; do not tune on pilot questions
SCORING_VERSION = "hybrid_thought_time_stale_v2"
EVENT_BOOST = 0.035
MAX_K = 30
# Only use event_date when the question explicitly asks about an event or schedule.
_EVENT_TERMS = re.compile(r'開催|予定|行事|イベント|会議|予約|締切|締め切り|期限|出発|到着|開始日|実施日|受診|面接日|試験日|訪問日')
_SNAPSHOT_TERMS = re.compile(r'時点|当時|その頃|そのころ|以前|変更前|修正前|過去|当初|までに|その日には')


def resolve_search_intent(query: str, as_of: str | None = None) -> dict:
    """Disambiguate *event date* from *snapshot date*, preserving explicit as_of.

    All inferred values originate exclusively in the user's query string.
    """
    base = dict(classify_query(query, as_of=as_of))
    date_mentioned = base.get('mentioned_date')
    is_event = (bool(date_mentioned) and bool(_EVENT_TERMS.search(query))
                and not bool(_SNAPSHOT_TERMS.search(query))
                and base['intent'] != 'change'
                and base.get('date_source') != 'explicit_field')
    if is_event:
        base['intent'] = 'current'  # schedules are not snapshots of memo creation
        base['as_of'] = None
        base['date_source'] = 'event_date_in_question'
        base['needs_date_clarification'] = False
    base['time_reference'] = 'event' if is_event else ('snapshot' if base['as_of'] else 'none')
    base['event_target_date'] = date_mentioned if is_event else None
    base['classifier'] = 'rule_based_integrated_v1'
    return base


def _event_bonus(memo: dict, event_day: str | None) -> float:
    if not event_day:
        return 0.0
    val = time_align(memo, event_day, event_day)
    if val is None:
        return 0.0  # Missing event_date is NOT evidence of a mismatch.
    return EVENT_BOOST * max(0.0, min(1.0, float(val)))


def _validated_matrix(memos: list, query_vector: Any):
    vectors = [m.get('vector') for m in memos]
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise ValueError('保存済みメモのベクトル次元が混在しています')
    mat = np.asarray(vectors, dtype=float)
    vec = np.asarray(query_vector, dtype=float)
    if mat.ndim != 2 or mat.shape[1] <= 0 or vec.shape != (mat.shape[1],):
        raise ValueError('クエリとメモのベクトル次元が一致しません')
    if not np.isfinite(mat).all() or not np.isfinite(vec).all():
        raise ValueError('埋め込みに不正な数値が含まれています')
    return mat, vec



def _current_version_precedence(candidates, semantic, scores, superseded,
                                threshold):
    """Constrain scores within *confirmed* supersedes chains, not all old notes.

    The current answer may cite a historical memo, but a confirmed later
    replacement should outrank that old memo if the replacement is itself a
    viable query match. No oracle labels, inferred links, event dates or memo
    version metadata are used. No boost is given to unrelated successors.

    Strictly cap only the old memo score just under its terminal successor.
    This O(N) pass has no effect for past/change, absent successors, branches,
    or replacements that fail the original selection threshold.
    """
    adjusted = np.asarray(scores, dtype=float).copy()
    caps = np.zeros(len(candidates), dtype=float)
    by_id = {str(m['id']): i for i, m in enumerate(candidates)}
    for old_id, newest_id in superseded.items():
        old_i, new_i = by_id.get(str(old_id)), by_id.get(str(newest_id))
        if old_i is None or new_i is None or old_i == new_i:
            continue
        # A successor must independently qualify for a result: relevance is
        # not established by the stored update link alone.
        if scores[new_i] < threshold or semantic[new_i] < 0.15:
            continue
        if adjusted[old_i] >= adjusted[new_i]:
            adjusted[old_i] = max(0.0, math.nextafter(float(adjusted[new_i]), -math.inf))
            caps[old_i] = float(adjusted[old_i] - scores[old_i])
    return adjusted, caps


def integrated_search(query: str, *, memos: list[dict], query_vector,
                      as_of: str | None = None, top_k: int = 10,
                      threshold: float = 0.30,
                      use_lexical=True, use_relations=True, use_time=True,
                      use_stale=True) -> dict:
    """Rank for the current/past/change task using ONLY stored user data.

    Stage 1: Semantic (0.7) + normalized BM25 (0.3).
    Stage 2: approved revision/development/branch graph, 3 hops max.
    Stage 3: optional event_date proximity, current-only stale multiplier,
             and precedence for relevant confirmed newer replacements.

    Past snapshots apply created_at cutoff *before* all ranking stages.
    Change queries retain older versions without stale penalties.
    """
    classified = resolve_search_intent(query, as_of=as_of)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_K:
        raise ValueError('top_k は1～30の整数にしてください')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold は0～1の数値にしてください')
    cutoff = date.fromisoformat(classified['as_of']) if classified['as_of'] else None
    pool = live_memos(memos, cutoff)
    candidates = [m for m in pool if m.get('vector') is not None]
    if not candidates:
        return {'query': query, 'classification': classified, 'results': [],
                'total_candidates': 0, 'scoring': SCORING_VERSION}

    matrix, qvector = _validated_matrix(candidates, query_vector)
    semantic = np.clip(matrix @ qvector, 0.0, 1.0)
    lexical = (BM25().fit(f"{m.get('title', '')} {m.get('content', '')}" for m in candidates).scores_normalized(query)
               if use_lexical else np.zeros(len(candidates)))
    base = SEM_W * semantic + LEX_W * lexical if use_lexical else semantic.copy()
    if use_relations:
        related, evidence = rerank_stored(candidates, base, semantic, classified['intent'],
                                         as_of=classified['as_of'])
    else:
        related, evidence = base.copy(), {}
    superseded = superseded_map(pool)

    event_scores = np.asarray([_event_bonus(m, classified['event_target_date']) if use_time else 0.0 for m in candidates])
    scores_before_stale = np.clip(related + event_scores, 0.0, 1.0)
    current_only_stale = use_stale and (classified['intent'] == 'current')
    multipliers = np.ones(len(candidates), dtype=float)
    if current_only_stale:
        for i, memo in enumerate(candidates):
            if str(memo['id']) in superseded:
                multipliers[i] = STALE_MULTIPLIER
    final = np.clip(scores_before_stale * multipliers, 0, 1)
    precedence_caps = np.zeros(len(candidates), dtype=float)
    # Event-date questions are not necessarily requests for the current state.
    # Do not apply version-precedence to a dated event look-up.
    if current_only_stale and classified['event_target_date'] is None:
        final, precedence_caps = _current_version_precedence(
            candidates, semantic, final, superseded, threshold)
    order = sorted(range(len(candidates)), key=lambda i: (-float(final[i]), i))
    selected = [i for i in order if final[i] >= threshold][:top_k]

    output = []
    for i in selected:
        memo = candidates[i]
        mid = str(memo['id'])
        output.append({
            'id': mid, 'title': memo.get('title', ''), 'content': memo.get('content', ''),
            'date': memo.get('date', ''), 'created_at': memo.get('created_at'),
            'event_date': memo.get('event_date'),
            'stale': mid in superseded, 'supersedes': memo.get('supersedes', []),
            'superseded_by': superseded.get(mid),
            'related_ids': [], 'score': int(round(float(final[i]) * 100)),
            'raw_score': float(final[i]),
            'parts': {
                'sem': round(float(semantic[i]), 4),
                'bm25': round(float(lexical[i]), 4),
                'hybrid': round(float(base[i]), 4),
                'lineage_delta': round(float(related[i] - base[i]), 4),
                'event_time_bonus': round(float(event_scores[i]), 4),
                'stale_multiplier': round(float(multipliers[i]), 4),
                'version_precedence_delta': round(float(precedence_caps[i]), 4),
            },
            'thought_path_evidence': evidence.get(i, []),
            'lineage_intent': classified['intent'],
            'relation_source_policy': 'saved_only',
            'search_scoring': SCORING_VERSION,
        })
    return {'query': query, 'classification': classified, 'results': output,
            'total_candidates': len(candidates), 'scoring': SCORING_VERSION}
