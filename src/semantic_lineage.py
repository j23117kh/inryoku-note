"""Conservative semantic-first, evidence-aware lineage reranking (research v1).

This module does not mutate memo records. It never uses chain_id/version,
synthetic ground-truth labels, or query relevance labels. Accepted relations
are sourced from stored thought_links/supersedes; similarity-based ancestry
is always marked as tentative and never establishes a revision/stale status.

Scores use fixed engineering defaults, NOT weights optimized on the already
inspected 188+16 evaluation questions. This is a new exploratory method.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import re
import numpy as np

# Untrained engineering defaults, NOT experimentally optimized.
APPROVED_REVISION_BOOST = 0.10
APPROVED_DEVELOPMENT_BOOST = 0.07
APPROVED_BRANCH_BOOST = 0.06
APPROVED_PARENT_BOOST = 0.025
TENTATIVE_BOOST = 0.025
REVISION_OLD_PENALTY = 0.10
TIME_WINDOW_BONUS = 0.05
INFERRED_TAU = 0.90
EXPANSION_SEM_FLOOR = 0.15
MAX_ANCHORS = 7
MAX_INFERRED_NEIGHBORS = 2

_HISTORY_RE = re.compile(r"以前|昔|過去|当時|元々|もともと|旧版|変更前|修正前|前の考え|前回の")
_TRANSITION_RE = re.compile(r"変化|変遷|経緯|どう変わ|どのように変わ|推移|比較|変更履歴|修正履歴|更新履歴|分岐の流れ")
_RELATIONS = {"revision", "development", "branch"}


def query_intent(query: str) -> str:
    """Three conservative intent classes. as_of is enforced in search_core."""
    if _TRANSITION_RE.search(query or ""):
        return "transition"
    if _HISTORY_RE.search(query or ""):
        return "past"
    return "current"


def _date(memo):
    value = memo.get('created_at') or memo.get('date')
    if not isinstance(value, str):
        return None
    try:
        d = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except ValueError:
        return None


def _inferred_context(parent, child):
    """Avoid treating templated near-duplicates as an inferred thought lineage.

    Strong similarity is a *candidate signal*, not proof of a relation.
    Require a stable, non-generic shared subject title or an explicit
    reference by the newer note to its earlier version.
    """
    import unicodedata
    def normalized_title(m):
        return ''.join(unicodedata.normalize('NFKC', str(m.get('title') or '')).lower().split())
    a, b = normalized_title(parent), normalized_title(child)
    if len(a) >= 4 and a == b and a not in {'無題の思考','タイトルなし','新規メモ','メモ帳','untitled'}:
        return True
    content = str(child.get('content') or '')
    return any(x in content for x in ('前回の考え','以前の方針','前の案','前のメモ','方針を変更','訂正する','これまでの案'))


def _approved_edges(candidates):
    """(parent-index, child-index, relation, provenance) unique ordered edges."""
    index = {str(m['id']): i for i, m in enumerate(candidates)}
    dates = [_date(m) for m in candidates]
    edges = {}
    for ci, child in enumerate(candidates):
        cid = str(child['id'])
        for old_id in child.get('supersedes') or []:
            pi = index.get(str(old_id))
            if pi is not None and pi != ci and not (dates[pi] and dates[ci] and dates[pi] > dates[ci]):
                edges[(pi, ci)] = ('revision', 'confirmed_supersedes')
        for link in child.get('thought_links') or []:
            if not isinstance(link, dict) or link.get('relation') not in _RELATIONS:
                continue
            pi = index.get(str(link.get('parent_id')))
            if pi is None or pi == ci or (dates[pi] and dates[ci] and dates[pi] > dates[ci]):
                continue
            # An explicitly accepted relationship takes priority over supersedes
            # when imported records disagree. Preserve that disagreement in UI.
            edges[(pi, ci)] = (link['relation'], 'approved_thought_link')
    return [(p,c,rel,source) for (p,c),(rel,source) in sorted(edges.items())]


def rerank(
    candidates,
    semantic_scores,
    similarities,
    *,
    threshold: float,
    query: str,
    time_from=None,
    time_to=None,
    time_align=None,
):
    """Return (reranked scores, extra eligible indices, relation metadata, deltas).

    Every candidate meeting the original Semantic threshold remains eligible,
    even after an update penalty. Approved one-hop successors can also become
    eligible when anchored by a qualifying Semantic hit. No low-anchor failure.
    """
    n = len(candidates)
    base = np.asarray(semantic_scores, dtype=float)
    sim = np.asarray(similarities, dtype=float)
    if base.shape != (n,) or sim.shape != (n,n):
        raise ValueError('Semantic-lineage score/similarity dimensions disagree')
    if not (np.isfinite(base).all() and np.isfinite(sim).all()):
        raise ValueError('Non-finite score in Semantic-lineage inputs')
    if len({str(m['id']) for m in candidates}) != n:
        raise ValueError('Duplicate memo IDs in Semantic-lineage candidates')

    result = base.copy()
    delta = np.zeros(n, dtype=float)
    reasons = defaultdict(list)
    eligible_extras = set()
    intent = query_intent(query)
    anchor_order = sorted(range(n), key=lambda i: (-float(base[i]), str(candidates[i]['id'])))
    anchors = {i for i in anchor_order[:MAX_ANCHORS] if base[i] >= threshold}

    def change(i, value, reason, *, allow_expansion=False):
        if abs(value) < 1e-12:
            return
        delta[i] += value
        reasons[i].append(reason)
        if allow_expansion and base[i] >= EXPANSION_SEM_FLOOR:
            eligible_extras.add(i)

    edges = _approved_edges(candidates)
    for parent, child, rel, source in edges:
        if parent not in anchors and child not in anchors:
            continue
        evidence = {'type': rel, 'source': source,
                    'parent_id': str(candidates[parent]['id']),
                    'child_id': str(candidates[child]['id'])}
        if rel == 'revision':
            if intent == 'past':
                change(parent, APPROVED_REVISION_BOOST, evidence, allow_expansion=True)
            elif intent == 'transition':
                change(parent, APPROVED_PARENT_BOOST, evidence, allow_expansion=True)
                change(child, APPROVED_REVISION_BOOST, evidence, allow_expansion=True)
            else:
                change(child, APPROVED_REVISION_BOOST, evidence, allow_expansion=True)
                # Deprioritize only when an accepted replacement exists in the pool.
                change(parent, -REVISION_OLD_PENALTY, evidence)
        elif rel in ('development', 'branch'):
            bonus = APPROVED_DEVELOPMENT_BOOST if rel == 'development' else APPROVED_BRANCH_BOOST
            change(child, bonus, evidence, allow_expansion=True)
            if intent in ('transition','past'):
                change(parent, APPROVED_PARENT_BOOST, evidence, allow_expansion=True)

    # Tentative relations are deliberately weak and cannot invalidate old memos.
    # Candidate pairs with identical/general phrasing are not certified as
    # revisions, branches or developments.
    dates = [_date(m) for m in candidates]
    approved_pairs = {(p,c) for p,c,_,_ in edges}
    for anchor in sorted(anchors):
        if dates[anchor] is None:
            continue
        neighbors = []
        for j in range(n):
            if j == anchor or dates[j] is None or dates[j] == dates[anchor]:
                continue
            s = float(sim[anchor,j])
            if s < INFERRED_TAU:
                continue
            parent, child = (j,anchor) if dates[j] < dates[anchor] else (anchor,j)
            if (parent, child) in approved_pairs:
                continue
            if not _inferred_context(candidates[parent], candidates[child]):
                continue
            neighbors.append((s,j,parent,child))
        neighbors.sort(key=lambda x: (-x[0], str(candidates[x[1]]['id'])))
        for s,j,parent,child in neighbors[:MAX_INFERRED_NEIGHBORS]:
            # Restrict speculative boosts to previously eligible Semantic hits.
            if base[j] < threshold:
                continue
            # Newer version can get a small preference, but is NEVER marked
            # as superseding or stale on a similarity-only guess.
            target = parent if intent == 'past' else child
            evidence = {'type':'inferred_related', 'source':'tentative_similarity',
                        'parent_id':str(candidates[parent]['id']),
                        'child_id':str(candidates[child]['id']),
                        'similarity':round(s,4)}
            change(target, TENTATIVE_BOOST, evidence)

    if (time_from is not None or time_to is not None) and time_align is not None:
        for i,m in enumerate(candidates):
            alignment = time_align(m, time_from, time_to)
            if alignment is not None and math.isfinite(float(alignment)):
                change(i, TIME_WINDOW_BONUS * max(0.,min(float(alignment),1.)),
                       {'type':'time_window', 'source':'explicit_query_range'})

    # Bound the *total* relation influence so multiple links cannot dominate
    # the semantic score by accumulation of weak/duplicate links.
    delta = np.clip(delta, -REVISION_OLD_PENALTY, APPROVED_REVISION_BOOST)
    result = np.clip(base + delta, 0., 1.)
    return result, eligible_extras, dict(reasons), delta, intent
