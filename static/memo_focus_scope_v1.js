/* THESIS_AUTO_RELATED_UI_V2
 * UI-only automatic focus. No change to API requests, ranking, saved memos,
 * evaluation labels or the approved thought-relation graph.
 *
 * A question/draft search focuses the first ranked result. Clicking an archive
 * memo, a search-result title or a graph memo selects that exact memo. Up to
 * three links of persisted development/revision/branch/supersedes are followed.
 * Generic 'related' links are deliberately NOT treated as thought evolution.
 * Search results remain in original relative rank order. Related archives
 * outside the ranked hits appear in the MAP only and are never scored hits.
 */
(() => {
    'use strict';
    if (window.__thesisAutoRelatedUI) return;

    const MAX_HOPS = 3;
    const MAX_GRAPH_MEMOS = 30;
    let focusId = '';
    let focusOrigin = '';
    let source = 'initial';
    let boundNetwork = null;

    function archive() {
        return Array.isArray(window.archiveData) ? window.archiveData : [];
    }
    function recordMap() {
        const map = new Map();
        const add = m => {
            if (m && m.id != null && !map.has(String(m.id))) map.set(String(m.id), m);
        };
        for (const m of archive()) add(m);
        if (typeof savedRelationGraph !== 'undefined' && savedRelationGraph) {
            for (const m of (savedRelationGraph.nodes || [])) add(m);
        }
        if (typeof lastResults !== 'undefined' && Array.isArray(lastResults)) {
            for (const m of lastResults) add(m);
        }
        return map;
    }
    function adjacency() {
        const records = recordMap();
        const next = new Map([...records.keys()].map(id => [id, new Set()]));
        const connect = (a, b) => {
            if (a == null || b == null) return;
            a = String(a); b = String(b);
            if (a !== b && next.has(a) && next.has(b)) {
                next.get(a).add(b);
                next.get(b).add(a);
            }
        };
        for (const memo of records.values()) {
            const id = String(memo.id);
            for (const link of (Array.isArray(memo.thought_links) ? memo.thought_links : [])) {
                if (link && ['development', 'revision', 'branch'].includes(link.relation)) {
                    connect(link.parent_id, id);
                }
            }
            const olds = Array.isArray(memo.supersedes) ? memo.supersedes :
                memo.supersedes == null ? [] : [memo.supersedes];
            for (const old of olds) {
                connect(old && typeof old === 'object' ? (old.id ?? old.memo_id) : old, id);
            }
        }
        if (typeof savedRelationGraph !== 'undefined' && savedRelationGraph) {
            for (const edge of (savedRelationGraph.edges || [])) {
                if (edge && ['development', 'revision', 'branch'].includes(edge.relation)) {
                    connect(edge.from, edge.to);
                }
            }
        }
        return next;
    }
    function visibleIds() {
        if (!focusId) return null;
        const links = adjacency();
        // Preserve the selected memo even if archive data is still loading.
        const ids = new Set([focusId]);
        let frontier = [focusId];
        for (let step = 0; step < MAX_HOPS && frontier.length; step++) {
            const following = [];
            for (const id of frontier) {
                for (const other of (links.get(id) || [])) {
                    if (!ids.has(other)) { ids.add(other); following.push(other); }
                }
            }
            frontier = following;
        }
        return ids;
    }
    function status() {
        const el = document.getElementById('memo-auto-focus-status');
        if (!el) return;
        if (!focusId) {
            el.textContent = '関連メモは検索・メモ選択に応じて自動表示します。';
            return;
        }
        const m = recordMap().get(focusId);
        const count = visibleIds()?.size || 1;
        const from = focusOrigin === 'ranked' ? '検索の最上位メモ' : '選択したメモ';
        el.textContent = `自動表示：${from}「${String(m?.title || focusId)}」と、保存済み思考関係でつながるメモ（最大${MAX_HOPS}段階・${count}件）`;
    }
    function setFocus(id, origin, repaint = false) {
        const next = id == null ? '' : String(id);
        const changed = next !== focusId;
        focusId = next;
        focusOrigin = next ? String(origin || '') : '';
        status();
        if (repaint && (changed || source === 'selected') && typeof redrawSearchResults === 'function') {
            redrawSearchResults();
        }
    }
    function choose(id, origin, repaint = true) {
        source = 'selected';
        setFocus(id, origin, repaint);
    }
    function addStatus() {
        const select = document.getElementById('intent-search-method');
        if (!select || document.getElementById('memo-auto-focus-status')) return;
        // No dropdown, no manual '解除' button: scope is always automatic.
        const el = document.createElement('p');
        el.id = 'memo-auto-focus-status';
        el.className = 'mt-1 text-[10px] text-emerald-200/80 leading-relaxed';
        el.setAttribute('role', 'status');
        el.setAttribute('aria-live', 'polite');
        select.parentNode.insertAdjacentElement('afterend', el);
        status();
    }
    function resultIds(data) {
        return new Set((Array.isArray(data) ? data : [])
            .filter(m => m && m.id != null).map(m => String(m.id)));
    }
    function onRankedResults(data, kind) {
        if (kind === 'question') {
            source = 'question';
            const first = (Array.isArray(data) ? data : []).find(m => m && m.id != null);
            setFocus(first ? first.id : '', 'ranked', false);
        } else if (kind === 'draft' && source !== 'selected') {
            source = 'draft';
            const first = (Array.isArray(data) ? data : []).find(m => m && m.id != null);
            setFocus(first ? first.id : '', 'ranked', false);
        }
    }

    // The API's ranked results, scores and relative order remain unchanged.
    if (typeof resultsForDisplay === 'function') {
        const original = resultsForDisplay;
        resultsForDisplay = function(data) {
            const items = original(data);
            const permitted = visibleIds();
            return permitted ? items.filter(m => m && permitted.has(String(m.id))) : items;
        };
    }
    if (typeof getSavedRelationItems === 'function') {
        const original = getSavedRelationItems;
        getSavedRelationItems = function(id) {
            const items = original(id);
            const permitted = visibleIds();
            return permitted ? items.filter(item => permitted.has(String(item.otherId))) : items;
        };
    }
    if (typeof buildResultMapModel === 'function') {
        const original = buildResultMapModel;
        buildResultMapModel = function(data, mode, style) {
            const permitted = visibleIds();
            if (!permitted) return original(data, mode, style);

            const ranked = (Array.isArray(data) ? data : []).filter(m => m && permitted.has(String(m.id)));
            const rankedIds = resultIds(ranked);
            const records = recordMap();
            const supplemental = [...permitted].filter(id => !rankedIds.has(id) && records.has(id))
                .sort((a,b) => (a === focusId ? -1 : b === focusId ? 1 : a.localeCompare(b)))
                .slice(0, Math.max(0, MAX_GRAPH_MEMOS - ranked.length));
            // Supplement the visualization with saved memos that were NOT in
            // the top ranked hits, without adding them to the result LIST.
            const input = ranked.concat(supplemental.map(id => records.get(id)));
            const model = original(input, mode, style);
            if (!model || !Array.isArray(model.nodes) || !Array.isArray(model.edges)) return model;
            const keep = new Set((model.nodes || [])
                .filter(n => permitted.has(String(n.id))).map(n => String(n.id)));
            if (ranked.some(m => keep.has(String(m.id))) &&
                model.nodes.some(n => String(n.id) === '__query__')) keep.add('__query__');
            for (const n of model.nodes) {
                const id = String(n.id);
                if (!id.startsWith('__topic_')) continue;
                if (model.edges.some(e => (String(e.from) === id && keep.has(String(e.to))) ||
                    (String(e.to) === id && keep.has(String(e.from))))) keep.add(id);
            }
            model.nodes = model.nodes.filter(n => keep.has(String(n.id)));
            model.edges = model.edges.filter(e => {
                const from = String(e.from), to = String(e.to);
                if (!keep.has(from) || !keep.has(to)) return false;
                // Archive-only additions must not be misrepresented as ranked hits.
                if (from === '__query__' && !rankedIds.has(to)) return false;
                if (to === '__query__' && !rankedIds.has(from)) return false;
                return true;
            });
            model.detail = `自動表示：関連するメモ${model.nodes.filter(n => permitted.has(String(n.id))).length}件`
                + `（検索ヒット${ranked.length}件、保存済み関係を最大${MAX_HOPS}段階）`;
            return model;
        };
    }
    if (typeof drawList === 'function') {
        const original = drawList;
        drawList = function(data) {
            const result = original(data);
            const list = document.getElementById('association-list');
            if (!list || !list.querySelectorAll) { status(); return result; }
            // Clicking a result's TITLE selects its actual saved memo.
            for (const card of list.querySelectorAll('div.p-3\\.5')) {
                const quote = card.querySelector('button[onclick^="quoteAt("]');
                const match = quote?.getAttribute('onclick')?.match(/quoteAt\((\d+)\)/);
                const memo = match ? displayedMemoResults[Number(match[1])] : null;
                const title = card.querySelector('h4');
                if (!memo?.id || !title || title.dataset.autoFocusBound) continue;
                title.dataset.autoFocusBound = '1';
                title.tabIndex = 0;
                title.setAttribute('role','button');
                title.setAttribute('aria-label', `「${memo.title || '無題'}」を中心に関連メモを表示`);
                title.title = 'クリックするとこのメモの思考系列を表示します';
                title.classList.add('cursor-pointer', 'hover:text-emerald-300');
                title.addEventListener('click', () => choose(memo.id,'result'));
                title.addEventListener('keydown', ev => {
                    if (ev.key === 'Enter' || ev.key === ' ') {
                        ev.preventDefault(); choose(memo.id,'result');
                    }
                });
            }
            status();
            return result;
        };
    }
    // Starting a new draft frees an old selected memo. The next completed
    // ranked response will establish the new automatic focus.
    document.getElementById('editor')?.addEventListener('input', () => {
        source = 'draft';
        setFocus('', '', false);
    });
    // Click a graph memo to recenter its saved thought lineage. Preserve the
    // text detail that the original graph click handler presents.
    if (typeof drawMap === 'function') {
        const original = drawMap;
        drawMap = function(...args) {
            const result = original(...args);
            if (typeof memoNetwork === 'undefined' || !memoNetwork || memoNetwork === boundNetwork) return result;
            boundNetwork = memoNetwork;
            memoNetwork.on('click', params => {
                if (!Array.isArray(params?.nodes) || params.nodes.length !== 1) return;
                const id = String(params.nodes[0]);
                if (id.startsWith('__') || !recordMap().has(id)) return;
                // Run after the graph's own click-to-detail callback.
                queueMicrotask(() => {
                    const detail = document.getElementById('result-map-detail');
                    const content = document.getElementById('result-map-detail-content');
                    const description = content?.textContent || '';
                    choose(id, 'graph', true);
                    if (detail && content && description) {
                        content.textContent = description;
                        detail.classList.remove('hidden');
                    }
                });
            });
            return result;
        };
    }

    addStatus();
    window.__thesisAutoRelatedUI = Object.freeze({
        version: '2', onRankedResults, focusMemo: (id) => choose(id,'archive',true),
        getFocusedId: () => focusId,
        getVisibleIds: () => { const s = visibleIds(); return s ? [...s] : null; },
        getSource: () => source
    });
    // Compatibility for existing UI inspection scripts.
    window.__thesisFocusScope = window.__thesisAutoRelatedUI;
})();
