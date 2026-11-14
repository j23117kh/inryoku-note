"""Approved thought relationships, separate from search-time inferred lineage.

Each link is stored on the newer/child memo as {parent_id, relation}.
Only 'revision' supersedes the parent: branches and developments must not
silently mark the parent stale.
"""
from collections import deque

RELATIONS = frozenset({'revision', 'development', 'branch', 'related'})
DIRECTED = RELATIONS - {'related'}


def _live_index(memos):
    return {m['id']: m for m in memos if m.get('status') != 'deleted'}


def _links(memo):
    return memo.get('thought_links') or []


def validate_link(memos, child_id, parent_id, relation):
    if relation not in RELATIONS:
        raise ValueError('関係の種類が不正です')
    index = _live_index(memos)
    if child_id not in index or parent_id not in index:
        raise ValueError('メモが見つからないか、削除されています')
    if child_id == parent_id:
        raise ValueError('同じメモ同士は接続できません')
    child = index[child_id]
    if relation in DIRECTED:
        # Historical entries sometimes contain incomplete dates. Do not infer
        # ordering from ID: validate explicit timestamps when both are present.
        parent_date = index[parent_id].get('created_at')
        child_date = child.get('created_at')
        if parent_date and child_date and parent_date > child_date:
            raise ValueError('親メモは子メモより後に作成されています')
        adj = {key: set() for key in index}
        for item in index.values():
            for link in _links(item):
                if link.get('relation') in DIRECTED:
                    p = link.get('parent_id')
                    if p in index and not (item['id'] == child_id and p == parent_id):
                        adj[p].add(item['id'])
            for p in item.get('supersedes', []):
                if p in index:
                    adj[p].add(item['id'])
        # A new parent -> child edge would create a cycle if parent is
        # already reachable from child.
        pending, seen = [child_id], set()
        while pending:
            current = pending.pop()
            if current == parent_id:
                raise ValueError('思考関係に循環が発生します')
            if current not in seen:
                seen.add(current)
                pending.extend(adj[current] - seen)
    return child


def set_link(memos, child_id, parent_id, relation):
    child = validate_link(memos, child_id, parent_id, relation)
    current = [dict(x) for x in _links(child) if x.get('parent_id') != parent_id]
    current.append({'parent_id': parent_id, 'relation': relation})
    child['thought_links'] = current
    supersedes = set(child.get('supersedes', []))
    if relation == 'revision':
        supersedes.add(parent_id)
    else:
        supersedes.discard(parent_id)
    child['supersedes'] = sorted(supersedes)
    return current


def remove_link(memos, child_id, parent_id):
    index = _live_index(memos)
    if child_id not in index:
        raise ValueError('メモが見つかりません')
    child = index[child_id]
    before = _links(child)
    after = [x for x in before if x.get('parent_id') != parent_id]
    if len(before) == len(after):
        raise ValueError('該当する関係がありません')
    child['thought_links'] = after
    child['supersedes'] = [x for x in child.get('supersedes', []) if x != parent_id]
    return after


def relationship_graph(memos, start_id=None, hops=3):
    index = _live_index(memos)
    if start_id is not None and start_id not in index:
        raise ValueError('メモが見つかりません')
    if not isinstance(hops, int) or not 0 <= hops <= 20:
        raise ValueError('hops は 0～20 の整数です')
    edges = {}
    for child in index.values():
        # Legacy update edges are retained even without thought_links.
        for parent_id in child.get('supersedes', []):
            if parent_id in index and parent_id != child['id']:
                edges[(parent_id, child['id'])] = 'revision'
        for link in _links(child):
            parent_id = link.get('parent_id')
            rel = link.get('relation')
            if parent_id in index and parent_id != child['id'] and rel in RELATIONS:
                edges[(parent_id, child['id'])] = rel
    if start_id is None:
        selected = set(index)
    else:
        adj = {key: set() for key in index}
        for p, c in edges:
            adj[p].add(c)
            adj[c].add(p)
        selected = {start_id}
        queue = deque([(start_id, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth < hops:
                for neighbor in sorted(adj[node] - selected):
                    selected.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return {
        'nodes': [{'id': key, 'title': index[key].get('title', ''),
                   'created_at': index[key].get('created_at')}
                  for key in sorted(selected)],
        'edges': [{'from': p, 'to': c, 'relation': rel, 'approved': True}
                  for (p, c), rel in sorted(edges.items())
                  if p in selected and c in selected],
    }
