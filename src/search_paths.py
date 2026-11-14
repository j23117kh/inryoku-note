"""Read-only, on-demand thought-relationship neighborhoods for real searches.

Search scores and embeddings are deliberately not involved. Only saved
thought_links and legacy supersedes edges are displayed; no oracle, labels,
or similarity-based relationships can be exposed as confirmed paths.
"""
from collections import deque
from datetime import datetime

from src.data_manager import live_memos, superseded_map
from src.thought_links import RELATIONS

_MAX_HOPS = 3
_MAX_NODES = 40
_MAX_EDGES = 80


def _time(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        # A date-only comparison is consistent with existing application's
        # created_at/as_of handling and avoids naive/aware comparison errors.
        return dt.date()
    except ValueError:
        return None


def _valid_temporal_edge(parent, child):
    p = _time(parent.get('created_at'))
    c = _time(child.get('created_at'))
    return not (p and c and p > c)


def _would_cycle(edges, parent, child):
    """Fail closed for bad imported directed relations; related is undirected."""
    if parent == child:
        return True
    adjacency = {}
    for edge in edges.values():
        if edge['relation'] == 'related':
            continue
        adjacency.setdefault(edge['from'], set()).add(edge['to'])
    pending, seen = [child], set()
    while pending:
        current = pending.pop()
        if current == parent:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, ()))
    return False


def build_search_path(memos, focus_id, *, hops=2, as_of=None,
                      max_nodes=_MAX_NODES, max_edges=_MAX_EDGES):
    """Return a small saved-relationship neighborhood around a search hit.

    Edges are from older/parent -> newer/child; traversal is bidirectional,
    so both prior thoughts and later branches can be viewed. 'related' is not
    ancestry. No reranking, external API calls, or memo mutation occurs.
    """
    if isinstance(hops, bool) or not isinstance(hops, int) or not 0 <= hops <= _MAX_HOPS:
        raise ValueError('hops は 0～3 の整数にしてください')
    if not isinstance(focus_id, str) or not focus_id.strip():
        raise ValueError('メモIDが不正です')
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= _MAX_NODES:
        raise ValueError('max_nodes が不正です')
    if isinstance(max_edges, bool) or not isinstance(max_edges, int) or not 1 <= max_edges <= _MAX_EDGES:
        raise ValueError('max_edges が不正です')

    index = {
        memo['id']: memo for memo in live_memos(memos, as_of)
        if isinstance(memo.get('id'), str) and memo['id']
    }
    if focus_id not in index:
        raise ValueError('メモが存在しないか、削除または期間外です')

    # Read-only snapshots: never update memos or infer unapproved edges.
    edges = {}
    for child_id in sorted(index):
        child = index[child_id]
        parents = child.get('supersedes') or []
        if isinstance(parents, (list, tuple)):
            for parent_id in sorted(set(x for x in parents if isinstance(x, str))):
                if (parent_id in index and parent_id != child_id
                        and _valid_temporal_edge(index[parent_id], child)
                        and not _would_cycle(edges, parent_id, child_id)):
                    edges[(parent_id, child_id)] = {
                        'from': parent_id, 'to': child_id,
                        'relation': 'revision', 'source': 'saved_revision',
                        'approved': False,
                    }

    for child_id in sorted(index):
        child = index[child_id]
        for link in child.get('thought_links') or []:
            if not isinstance(link, dict):
                continue
            parent_id = link.get('parent_id')
            relation = link.get('relation')
            if (not isinstance(parent_id, str) or parent_id not in index
                    or parent_id == child_id or relation not in RELATIONS):
                continue
            if relation != 'related' and not _valid_temporal_edge(index[parent_id], child):
                continue
            key = (parent_id, child_id)
            old = edges.get(key)
            if (relation != 'related' and not (old and old['relation'] != 'related')
                    and _would_cycle(edges, parent_id, child_id)):
                continue
            edges[key] = {
                'from': parent_id, 'to': child_id,
                'relation': relation, 'source': 'approved_link',
                'approved': True,
            }

    adjacency = {mid: set() for mid in index}
    for p, c in edges:
        adjacency[p].add(c)
        adjacency[c].add(p)

    depths = {focus_id: 0}
    queue = deque([focus_id])
    truncated = False
    while queue:
        current = queue.popleft()
        if depths[current] >= hops:
            continue
        for other in sorted(adjacency[current]):
            if other not in depths:
                if len(depths) >= max_nodes:
                    truncated = True
                    continue
                depths[other] = depths[current] + 1
                queue.append(other)

    selected = set(depths)
    neighborhood_edges = [v for (p, c), v in sorted(edges.items())
                          if p in selected and c in selected]
    if len(neighborhood_edges) > max_edges:
        neighborhood_edges = neighborhood_edges[:max_edges]
        truncated = True
    # superseded status is determined by stored revision edges, not by branch.
    superseded = superseded_map(list(index.values()))
    nodes = [{
        'id': mid,
        'title': index[mid].get('title') or '無題の思考',
        'created_at': index[mid].get('created_at') or index[mid].get('date') or '',
        'excerpt': str(index[mid].get('content') or '')[:160],
        'stale': mid in superseded,
        'depth': depths[mid],
    } for mid in sorted(selected, key=lambda key: (depths[key], key))]

    return {
        'focus_id': focus_id,
        'hops': hops,
        'nodes': nodes,
        'edges': neighborhood_edges,
        'has_links': bool(neighborhood_edges),
        'truncated': truncated,
        'source_policy': 'saved_links_only',
    }
