/* THESIS_FOCUS_SCOPE_UI_V1
 * Presentation-only filter. It never changes API query, score, ranking,
 * stored memos, or the frozen study evaluation contract.
 * A focus memo is selected explicitly, or inferred only from an EXACT,
 * unambiguous memo title contained in the question. Unrelated topic guesses
 * are deliberately prohibited.
 */
(() => {
    'use strict';
    const SELECT_ID = 'memo-focus-select';
    let focusId = '';
    let focusOrigin = '';

    function allArchiveMemos() {
        return Array.isArray(window.archiveData) ? window.archiveData : [];
    }
    function activeMemoIds() {
        if (!focusId) return null;
        const archive = allArchiveMemos();
        const allowed = new Set(archive.map(m => String(m.id)));
        if (!allowed.has(focusId)) return new Set();
        const adjacency = new Map([...allowed].map(id => [id, new Set()]));
        const connect = (p, c) => {
            p = String(p); c = String(c);
            if (allowed.has(p) && allowed.has(c) && p !== c) {
                adjacency.get(p).add(c);
                adjacency.get(c).add(p);
            }
        };
        // Persisted directed thought lineage, ignoring generic 'related'
        // edges that could connect unrelated stories through topic similarity.
        for (const m of archive) {
            const child = String(m.id);
            for (const edge of (Array.isArray(m.thought_links) ? m.thought_links : [])) {
                if (edge && ['development', 'revision', 'branch'].includes(edge.relation) && edge.parent_id != null) {
                    connect(edge.parent_id, child);
                }
            }
            for (const old of (Array.isArray(m.supersedes) ? m.supersedes : [])) {
                connect(typeof old === 'object' ? (old.id ?? old.memo_id) : old, child);
            }
        }
        if (typeof savedRelationGraph !== 'undefined' && Array.isArray(savedRelationGraph?.edges)) {
            for (const edge of savedRelationGraph.edges) {
                if (edge && ['development', 'revision', 'branch'].includes(edge.relation)) {
                    connect(edge.from, edge.to);
                }
            }
        }
        const component = new Set([focusId]);
        const pending = [focusId];
        for (let i = 0; i < pending.length; i++) {
            for (const id of adjacency.get(pending[i]) || []) {
                if (!component.has(id)) { component.add(id); pending.push(id); }
            }
        }
        return component;
    }

    function updateFocusDescription() {
        const el = document.getElementById('memo-focus-hint');
        if (!el) return;
        if (!focusId) {
            el.textContent = '全体表示。特定メモを選ぶと、そのメモと保存済みの思考系列だけを表示します。';
            return;
        }
        const m = allArchiveMemos().find(x => String(x.id) === focusId);
        const ids = activeMemoIds();
        el.textContent = '表示対象：' + (m?.title || focusId) + '（' + (ids?.size || 0)
            + '件の関係系列）。検索のスコア・順位・上位10件の取得条件は変更しません。';
    }
    function setFocus(id, origin = 'manual', repaint = true) {
        focusId = String(id || '');
        focusOrigin = focusId ? origin : '';
        const select = document.getElementById(SELECT_ID);
        if (select) select.value = focusId;
        updateFocusDescription();
        if (repaint && typeof redrawSearchResults === 'function') redrawSearchResults();
    }
    function fillMemoOptions() {
        const select = document.getElementById(SELECT_ID);
        if (!select) return;
        const previous = focusId;
        select.replaceChildren(new Option('全体表示（絞り込みなし）', ''));
        const memos = allArchiveMemos().filter(m => m && m.id != null && m.status !== 'deleted');
        memos.sort((a, b) => String(a.title || '').localeCompare(String(b.title || ''), 'ja') || String(a.id).localeCompare(String(b.id)));
        for (const m of memos) {
            select.add(new Option(`${String(m.title || '無題')}［${String(m.id).slice(0,12)}］`, String(m.id)));
        }
        if (previous && !memos.some(m => String(m.id) === previous)) {
            focusId = '';
            focusOrigin = '';
        }
        select.value = focusId;
        updateFocusDescription();
    }
    function maybeAutoFocus(question) {
        if (focusOrigin === 'manual') return;
        const q = String(question || '').normalize('NFKC').toLocaleLowerCase();
        const hits = allArchiveMemos().filter(m => {
            const title = String(m.title || '').trim().normalize('NFKC').toLocaleLowerCase();
            return title.length >= 5 && q.includes(title);
        });
        if (!hits.length) {
            if (focusOrigin === 'auto') setFocus('', '', false);
            return;
        }
        const longest = Math.max(...hits.map(m => String(m.title).length));
        const exact = hits.filter(m => String(m.title).length === longest);
        if (exact.length === 1) setFocus(exact[0].id, 'auto', false);
        else if (focusOrigin === 'auto') setFocus('', '', false);
    }

    function addFocusControl() {
        const methodSelect = document.getElementById('intent-search-method');
        if (!methodSelect || document.getElementById(SELECT_ID)) return;
        const panel = document.createElement('div');
        panel.className = 'mt-2 border-t border-slate-700/70 pt-2';
        const label = document.createElement('label');
        label.htmlFor = SELECT_ID;
        label.className = 'block text-[11px] font-bold text-emerald-300 mb-1';
        label.textContent = '対象メモで絞り込む（任意）';
        const controls = document.createElement('div');
        controls.className = 'flex gap-1';
        const select = document.createElement('select');
        select.id = SELECT_ID;
        select.className = 'flex-1 min-w-0 text-[11px] rounded bg-slate-900 border border-slate-700 p-1 text-slate-100';
        select.setAttribute('aria-label', '表示対象メモ');
        select.addEventListener('change', () => setFocus(select.value));
        const clear = document.createElement('button');
        clear.type = 'button';
        clear.className = 'rounded bg-slate-700 hover:bg-slate-600 px-2 py-1 text-[11px]';
        clear.textContent = '解除';
        clear.addEventListener('click', () => setFocus(''));
        controls.append(select, clear);
        const hint = document.createElement('p');
        hint.id = 'memo-focus-hint';
        hint.className = 'mt-1 text-[10px] text-slate-400 leading-relaxed';
        panel.append(label, controls, hint);
        methodSelect.parentNode.insertAdjacentElement('afterend', panel);
        fillMemoOptions();
    }

    // Wrap existing display entrypoints, preserving the API-side results.
    if (typeof resultsForDisplay === 'function') {
        const originalResultsForDisplay = resultsForDisplay;
        resultsForDisplay = function(results) {
            const shown = originalResultsForDisplay(results);
            const ids = activeMemoIds();
            return ids ? shown.filter(m => ids.has(String(m.id))) : shown;
        };
    }
    if (typeof getSavedRelationItems === 'function') {
        const oldGetSavedRelationItems = getSavedRelationItems;
        getSavedRelationItems = function(id) {
            const entries = oldGetSavedRelationItems(id);
            const ids = activeMemoIds();
            return ids ? entries.filter(entry => ids.has(String(entry.otherId))) : entries;
        };
    }
    if (typeof buildResultMapModel === 'function') {
        const originalBuildResultMapModel = buildResultMapModel;
        buildResultMapModel = function(data, mode, style) {
            const model = originalBuildResultMapModel(data, mode, style);
            const ids = activeMemoIds();
            if (!ids) return model;
            const permitted = new Set(ids);
            const groupIds = new Set((model.nodes || []).filter(n => !ids.has(String(n.id)) &&
                (model.edges || []).some(e => ids.has(String(e.from)) && String(e.to) === String(n.id) ||
                    ids.has(String(e.to)) && String(e.from) === String(n.id))).map(n => String(n.id)));
            if ((model.nodes || []).some(n => String(n.id) === '__query__')) permitted.add('__query__');
            for (const g of groupIds) if (g.startsWith('__')) permitted.add(g);
            model.nodes = (model.nodes || []).filter(n => permitted.has(String(n.id)));
            model.edges = (model.edges || []).filter(e => permitted.has(String(e.from)) && permitted.has(String(e.to)));
            model.detail = `対象の思考系列 ${ids.size}件（検索結果のうち一致したメモのみ表示）`;
            return model;
        };
    }
    if (typeof drawList === 'function') {
        const originalDrawList = drawList;
        drawList = function(data) {
            const result = originalDrawList(data);
            const list = document.getElementById('association-list');
            if (!list) return result;
            // Use the existing quoteAt(index) binding to handle grouped cards.
            for (const card of list.querySelectorAll('div.p-3\\.5')) {
                const quote = card.querySelector('button[onclick^="quoteAt("]');
                const match = quote?.getAttribute('onclick')?.match(/quoteAt\((\d+)\)/);
                const memo = match ? displayedMemoResults[Number(match[1])] : null;
                if (!memo?.id || card.querySelector('.focus-memo-card-button')) continue;
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'focus-memo-card-button mt-2 w-full rounded border border-emerald-900/80 bg-emerald-950/30 px-2 py-1.5 text-[10px] font-bold text-emerald-200';
                btn.textContent = focusId === String(memo.id) ? 'このメモが表示対象です' : 'このメモの思考系列に絞る';
                btn.addEventListener('click', () => setFocus(memo.id));
                card.appendChild(btn);
            }
            return result;
        };
    }
    if (typeof loadArchive === 'function') {
        const originalLoadArchive = loadArchive;
        loadArchive = async function(...args) {
            const result = await originalLoadArchive(...args);
            fillMemoOptions();
            return result;
        };
        window.onload = loadArchive;
    }
    if (typeof searchQuestionIntent === 'function') {
        const originalSearchQuestionIntent = searchQuestionIntent;
        searchQuestionIntent = async function(...args) {
            const question = document.getElementById('intent-query')?.value.trim() || '';
            maybeAutoFocus(question);
            updateFocusDescription();
            return originalSearchQuestionIntent(...args);
        };
    }

    addFocusControl();
    updateFocusDescription();
    window.__thesisFocusScope = Object.freeze({
        getFocusedId: () => focusId,
        getVisibleIds: () => activeMemoIds() ? [...activeMemoIds()] : null
    });
})();
