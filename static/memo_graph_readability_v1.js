/* THESIS_GRAPH_READABILITY_V1
 * Presentation-only changes to all three result-map layouts.
 * Never changes /api responses, ranking, scores, links, or data files.
 * Intended to run AFTER memo_focus_scope_v1.js if installed.
 */
(() => {
    'use strict';
    const MARK = 'THESIS_GRAPH_READABILITY_V1';
    if (window.__thesisGraphReadability) return;
    if (typeof buildResultMapModel !== 'function' || typeof drawMap !== 'function') {
        console.warn(MARK + ': map functions unavailable; graph UI unmodified');
        return;
    }
    let lastReadablePosition = null;
    const compact = (str, max) => {
        const s = String(str || '').trim();
        return [...s].length > max ? [...s].slice(0, max).join('') + '…' : s;
    };
    const memoTitle = (model, id) => {
        const memo = model.visibleMemoById?.get(String(id));
        return String(memo?.title || memo?.content || id).trim().slice(0, 70);
    };
    const fullEdgeExplanation = (model, edge) => {
        const from = String(edge.from), to = String(edge.to);
        const relationship = String(edge.__readableOriginalLabel || edge.title || '関係情報');
        const target = from.startsWith('__') ? '現在の検索' : memoTitle(model, from);
        const dest = to.startsWith('__') ? '現在の検索' : memoTitle(model, to);
        const original = String(edge.title || relationship);
        return `${target}\n↓ ${relationship}\n${dest}\n\n${original}`;
    };
    const fonts = {color:'#f1f5f9', size:15, face:'Noto Sans JP, sans-serif',
                   strokeWidth:3, strokeColor:'#020617', vadjust:7};
    const setNodeFont = (node, size = 15) => {
        node.font = {...(node.font || {}), ...fonts, size};
        // Full titles remain accessible on hover and click, even when labels are shortened.
        if (!node.title) node.title = String(node.label || node.id);
    };
    const deClutterEdges = (model) => {
        for (const edge of model.edges || []) {
            const original = String(edge.label || '');
            edge.__readableOriginalLabel = original || String(edge.title || '関係');
            // The recorded relation is always kept in the tooltip and edge-click panel.
            edge.title = String(edge.title || original || '関係');
            edge.label = '';
            edge.font = {...(edge.font || {}), size:11, color:'#cbd5e1'};
            if (edge.from === '__query__' || edge.to === '__query__') {
                edge.width = Math.min(Number(edge.width || 1), 1.2);
                edge.color = {...(typeof edge.color === 'object' ? edge.color : {color:edge.color}), opacity:0.22};
            }
        }
    };
    function overview(model) {
        for (const n of model.nodes || []) {
            if (String(n.id) === '__query__') {
                setNodeFont(n, 15);
                n.label = '質問';
                n.size = 23;
            } else {
                setNodeFont(n, 15);
                n.label = compact(String(n.label || ''), 11);
                n.size = Math.max(19, Math.min(Number(n.size || 20), 23));
            }
        }
        model.options = {
            ...model.options,
            layout:{...(model.options?.layout || {}), improvedLayout:true},
            physics:{enabled:true, solver:'forceAtlas2Based',
                stabilization:{enabled:true, iterations:230, updateInterval:35, fit:true},
                forceAtlas2Based:{gravitationalConstant:-160, centralGravity:0.025,
                    springLength:245, springConstant:0.038, damping:0.65, avoidOverlap:1},
                minVelocity:0.4},
            edges:{smooth:{enabled:true,type:'continuous',roundness:0.14}}
        };
        model.legend += ' ／線をクリックすると種類と接続先を表示';
        model.detail += ' ／メモ名は短縮表示（丸をクリックで全文）';
        // Leave overview default fit alone; for small panels "読みやすく" restores readable scale.
        lastReadablePosition = {position:{x:0,y:0},scale:0.9, mode:'hybrid'};
    }
    function timeline(model) {
        const memoNodes = (model.nodes || []).filter(n => !String(n.id).startsWith('__'));
        memoNodes.forEach((node,i) => {
            const oldLines = String(node.label || '').split('\n');
            node.label = compact(oldLines[0],12) + (oldLines.length > 1 ? '\n' + oldLines.slice(1).join(' ') : '');
            setNodeFont(node, 16);
            node.shape = 'box';
            node.margin = {top:14,bottom:14,left:13,right:13};
            node.widthConstraint = {minimum:145, maximum:215};
            node.x = i * 285;
            node.y = (i % 2) * 190;
            node.fixed = {x:true,y:true};
        });
        model.options = {
            ...model.options,physics:false,
            layout:{...(model.options?.layout || {}),improvedLayout:false},
            edges:{smooth:{enabled:true,type:'cubicBezier',roundness:0.16}}
        };
        const scale = 1.02;
        const anchor = {x:Math.min(2,Math.max(0,memoNodes.length-1))*285,y:95};
        model.initialPosition = {x:anchor.x,y:anchor.y,scale};
        lastReadablePosition = {position:anchor,scale,mode:'time'};
        model.legend += ' ／横にドラッグすると続きが見えます';
        model.detail += ' ／「読みやすく」で文字サイズを復元';
    }
    function topics(model) {
        const groups = (model.nodes || []).filter(n => String(n.id).startsWith('__topic_'));
        const nodesById = new Map((model.nodes || []).map(n => [String(n.id), n]));
        let startX = 0;
        const placed = new Set();
        for (const group of groups) {
            const groupId = String(group.id);
            const memberIds = (model.edges || []).filter(e => String(e.from) === groupId && nodesById.has(String(e.to)))
                .map(e => String(e.to));
            const members = memberIds.map(id=>nodesById.get(id)).filter(Boolean);
            const columns = members.length > 7 ? 3 : members.length > 2 ? 2 : 1;
            const columnWidth = 255, rowHeight = 152;
            const blockWidth = Math.max(285, columns * columnWidth);
            const mid = startX + blockWidth / 2;
            group.x=mid; group.y=0; group.fixed={x:true,y:true};
            group.label = compact(String(group.label || ''), 18);
            setNodeFont(group, 17);
            group.margin={top:13,bottom:13,left:14,right:14};
            group.widthConstraint={minimum:200,maximum:Math.min(blockWidth-30,460)};
            members.forEach((node,i) => {
                const col=i%columns, row=Math.floor(i/columns);
                node.x=startX+(col+0.5)*columnWidth;
                node.y=145+row*rowHeight;
                node.fixed={x:true,y:true};
                node.label=compact(String(node.label || ''), 11);
                node.size=20;
                setNodeFont(node, 15);
                placed.add(String(node.id));
            });
            startX+=blockWidth+145;
        }
        // Defensive fallback: retain any node not assigned to a topic.
        let leftovers=0;
        for (const node of model.nodes || []) {
            const id=String(node.id);
            if (id.startsWith('__topic_') || placed.has(id)) continue;
            node.x=startX+(leftovers%2)*255;
            node.y=145+Math.floor(leftovers/2)*152;
            node.fixed={x:true,y:true};
            node.label=compact(String(node.label || ''),11);
            setNodeFont(node,15);
            leftovers++;
        }
        model.options = {
            ...model.options, physics:false,
            layout:{...(model.options?.layout || {}),improvedLayout:false},
            edges:{smooth:{enabled:true,type:'cubicBezier',roundness:0.12}}
        };
        const anchorX = groups.length ? Number(groups[0].x || 0) : 0;
        model.initialPosition = {x:anchorX,y:230,scale:0.85};
        lastReadablePosition = {position:{x:anchorX,y:230},scale:0.9,mode:'topic'};
        model.legend += ' ／同じ色は話題の分類であり、更新関係とは別';
        model.detail += ' ／枠で話題を分離（線をクリックで関係を確認）';
    }
    const oldBuild = buildResultMapModel;
    buildResultMapModel = function(data, mode, style) {
        // The focus add-on, when installed, has already filtered results and map nodes.
        const model=oldBuild(data,mode,style);
        if (!model || !Array.isArray(model.nodes) || !Array.isArray(model.edges)) return model;
        deClutterEdges(model);
        if (style==='time') timeline(model);
        else if (style==='topic') topics(model);
        else overview(model);
        return model;
    };

    function readableZoom() {
        if (typeof memoNetwork === 'undefined' || !memoNetwork) return;
        const style = typeof currentMapStyle === 'string' ? currentMapStyle : 'hybrid';
        const p=lastReadablePosition?.mode === style ? lastReadablePosition : null;
        const center=p?.position || (typeof memoNetwork.getViewPosition === 'function' ? memoNetwork.getViewPosition() : {x:0,y:0});
        const scale=p?.scale || 0.9;
        memoNetwork.moveTo({position:center,scale,animation:{duration:230,easingFunction:'easeInOutQuad'}});
    }
    function addReadableButton() {
        if (document.getElementById('map-readable-zoom-button')) return;
        const fitBtn = document.querySelector('#view-map button[onclick="fitResultMap()"]');
        if (!fitBtn) return;
        const btn=document.createElement('button');
        btn.id='map-readable-zoom-button';
        btn.type='button';
        btn.textContent='読みやすく';
        btn.title='文字が読める倍率へ戻す（全体表示では縮小されることがあります）';
        btn.className='px-2 py-1 rounded-md bg-slate-700 text-white text-[11px] font-bold hover:bg-slate-600';
        btn.addEventListener('click',readableZoom);
        fitBtn.parentElement.insertBefore(btn,fitBtn);
    }
    const oldDraw=drawMap;
    drawMap=function(...args) {
        const outcome=oldDraw(...args);
        if (typeof memoNetwork === 'undefined' || !memoNetwork) return outcome;
        // Edge labels stay in the original tooltips and are accessible by click.
        memoNetwork.on('click',params => {
            if ((params.nodes || []).length || (params.edges || []).length !== 1) return;
            const edge = memoNetwork?.body?.data?.edges?.get(params.edges[0]);
            if (!edge) return;
            const detail=document.getElementById('result-map-detail');
            const content=document.getElementById('result-map-detail-content');
            if (!detail || !content) return;
            const from = String(edge.from);
            const to = String(edge.to);
            const clean = id => id === '__query__' ? '検索要求' :
                ((Array.isArray(window.archiveData) ? window.archiveData : []).find(m => String(m.id) === id)?.title || id);
            content.textContent=`接続元：${clean(from)}\n接続先：${clean(to)}\n関係：${String(edge.__readableOriginalLabel || edge.title || '関係情報')}\n\n${String(edge.title || '')}`;
            detail.classList.remove('hidden');
        });
        return outcome;
    };
    // Optional styles: canvas tooltip is readable in both fullscreen and panel mode.
    const css=document.createElement('style');
    css.id='thesis-graph-readability-style-v1';
    css.textContent=`
        #view-map .vis-tooltip, div.vis-tooltip {
            max-width: min(420px, 85vw) !important;
            white-space: pre-wrap !important;
            overflow-wrap: anywhere !important;
            background: #0f172a !important;
            color: #f1f5f9 !important;
            border: 1px solid #64748b !important;
            border-radius: 8px !important;
            padding: 9px 12px !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            box-shadow: 0 12px 28px rgba(0,0,0,0.45) !important;
        }
        #view-map #map-count, #view-map #map-legend { font-size: 11px; line-height: 1.45; }
        #view-map #result-map-detail { z-index: 24; }
        @media (max-width: 550px) {
            #map-readable-zoom-button { font-size: 10px; padding: 3px 5px; }
            #view-map #map-legend { max-height: 42px; overflow: hidden; }
        }
    `;
    document.head.appendChild(css);
    addReadableButton();
    window.__thesisGraphReadability=Object.freeze({resetReadableZoom:readableZoom,version:'1'});
})();
