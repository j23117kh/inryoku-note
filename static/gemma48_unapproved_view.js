(() => {
  'use strict';
  const DATA_URL = '/static/gemma48_unapproved_results.json';
  const TYPE_LABEL = { development: '発展', revision: '修正・置換', branch: '分岐', related: '関連', none: '関係なし' };
  const CSS = `
  #gm48-open {margin:0 8px;padding:7px 12px;background:#0f766e;border:1px solid #5eead4;border-radius:8px;color:#fff;cursor:pointer;font-size:13px}
  #gm48-overlay[hidden] {display:none !important}
  #gm48-overlay {position:fixed;inset:0;z-index:100100;background:rgba(0,0,0,.78);display:flex;align-items:center;justify-content:center;padding:12px}
  #gm48-panel {background:#17212e;color:#f8fafc;width:min(940px,100%);max-height:94vh;overflow:auto;border-radius:12px;border:1px solid #64748b;padding:18px;box-sizing:border-box;font:14px/1.6 system-ui,sans-serif}
  #gm48-panel * {box-sizing:border-box}
  #gm48-panel h2 {font-size:19px;font-weight:700;margin:0 0 5px}
  #gm48-panel p {margin:5px 0 10px}
  #gm48-panel button,#gm48-panel select,#gm48-panel input {border-radius:7px;padding:8px;border:1px solid #64748b;color:#f8fafc;background:#283548}
  #gm48-panel button {cursor:pointer}
  #gm48-controls {display:flex;gap:9px;flex-wrap:wrap;margin:13px 0}
  #gm48-controls input {min-width:170px;flex:2}
  #gm48-controls select {flex:1;min-width:140px}
  #gm48-list {border:1px solid #475569;border-radius:8px;max-height:220px;overflow-y:auto;margin:8px 0;padding:5px}
  #gm48-list button {width:100%;display:block;text-align:left;border:none;border-bottom:1px solid #475569;border-radius:0;background:none;padding:10px}
  #gm48-list button:hover,#gm48-list button[aria-pressed=true] {background:#334155}
  #gm48-detail {border:1px solid #64748b;border-radius:8px;padding:12px;margin-top:10px;background:#0f172a}
  #gm48-detail svg {display:block;width:100%;height:auto;max-height:160px}
  #gm48-detail .gm48-reason {white-space:pre-wrap;overflow-wrap:anywhere}
  #gm48-status {color:#a5f3fc}
  `;
  const style = document.createElement('style'); style.textContent = CSS; document.head.append(style);

  const button = document.createElement('button'); button.type='button'; button.id='gm48-open'; button.textContent='Gemma 4 推定関係（未承認）';
  (document.querySelector('header') || document.body).append(button);
  const overlay = document.createElement('div'); overlay.id='gm48-overlay'; overlay.hidden=true;
  overlay.innerHTML = `<section id="gm48-panel" role="dialog" aria-modal="true" aria-labelledby="gm48-heading">
  <div style="display:flex;justify-content:space-between;gap:10px;align-items:center"><h2 id="gm48-heading">Gemma 4の推定関係</h2><button id="gm48-close" type="button">閉じる</button></div>
  <p>AI推定・未承認：研究用の判定です。正式な承認済み関係や検索順位には反映されません。</p>
  <p id="gm48-status" aria-live="polite">読み込み前</p>
  <div id="gm48-controls"><input id="gm48-filter" type="search" aria-label="メモタイトルやIDで絞り込み" placeholder="メモタイトル・IDで絞り込み"><select id="gm48-type" aria-label="関係の種類"><option value="all">すべての関係</option><option value="development">発展</option><option value="revision">修正・置換</option><option value="branch">分岐</option><option value="related">関連</option></select></div>
  <div id="gm48-list" role="group" aria-label="Gemma 4の推定関係一覧"></div>
  <div id="gm48-detail" aria-live="polite">関係を選択すると詳細を表示します。</div>
  </section>`;
  document.body.append(overlay);
  const $=id=>document.getElementById(id);
  let records=[], memoById=new Map(), active=-1, previousFocus=null;
  function elt(tag, txt, cls) { const e=document.createElement(tag); if(txt!==undefined) e.textContent=String(txt); if(cls) e.className=cls; return e; }
  function svgNode(svg, tag, attrs, text) { const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,String(v)); if(text!==undefined)el.textContent=String(text);svg.append(el);return el; }
  function detail(i) {
    active=i; const r=records[i]; const area=$('gm48-detail');area.replaceChildren(); if(!r){area.textContent='関係を選択してください。';return;}
    area.append(elt('p',`${TYPE_LABEL[r.relation]}　|　信頼度 ${Number(r.confidence).toFixed(2)}　|　類似度 ${Number(r.cosine).toFixed(3)}`));
    const root=document.createElementNS('http://www.w3.org/2000/svg','svg');area.append(root);
    // SVG is an explanatory two-node link, not the app's approved relationship graph.
    root.setAttribute('viewBox','0 0 760 118');root.setAttribute('role','img');root.setAttribute('aria-label','旧メモから新メモへのAI推定関係');
    svgNode(root,'rect',{x:10,y:12,width:310,height:90,rx:10,fill:'#26364a',stroke:'#64748b'});
    svgNode(root,'rect',{x:440,y:12,width:310,height:90,rx:10,fill:'#26364a',stroke:'#64748b'});
    svgNode(root,'path',{d:'M325 57 H430 M421 49 L430 57 L421 65',stroke:'#5eead4','stroke-width':2.5,fill:'none'});
    svgNode(root,'text',{x:165,y:39,'text-anchor':'middle',fill:'#f8fafc','font-size':14},'旧メモ '+r.parent_id);
    svgNode(root,'text',{x:165,y:64,'text-anchor':'middle',fill:'#f8fafc','font-size':12},(r.parent_title||'').slice(0,23));
    svgNode(root,'text',{x:595,y:39,'text-anchor':'middle',fill:'#f8fafc','font-size':14},'新メモ '+r.child_id);
    svgNode(root,'text',{x:595,y:64,'text-anchor':'middle',fill:'#f8fafc','font-size':12},(r.child_title||'').slice(0,23));
    const heading=elt('p','Gemma 4の判定理由：');heading.style.fontWeight='700';area.append(heading);
    area.append(elt('p',r.reason||'理由の記録なし','gm48-reason'));
    area.append(elt('p','※分類結果は未承認です。関係の正しさを保証するものではありません。'));
    area.querySelectorAll('p').forEach(x=>x.style.margin='8px 0');
    $('gm48-list').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.index)===i)));
  }
  function drawList() {
    const q=$('gm48-filter').value.trim().toLowerCase(), type=$('gm48-type').value, list=$('gm48-list');list.replaceChildren();
    let shown=0;
    records.forEach((r,i)=>{if(type!=='all' && r.relation!==type)return;
      const hay=[r.parent_id,r.child_id,r.parent_title,r.child_title,r.reason].join(' ').toLowerCase();if(q&&!hay.includes(q))return;
      const b=elt('button',`${TYPE_LABEL[r.relation]} ｜ ${r.parent_title} → ${r.child_title} （${Number(r.confidence).toFixed(2)}）`); b.type='button';b.dataset.index=i;b.setAttribute('aria-pressed',String(i===active));b.onclick=()=>detail(i);list.append(b);shown++;
    });
    if(!shown) list.append(elt('p','該当する推定関係がありません。'));
  }
  async function show(){previousFocus=document.activeElement;overlay.hidden=false;$('gm48-close').focus();$('gm48-status').textContent='判定データと実際のメモを検証しています…';
    try {
      const [p,m]=await Promise.all([fetch(DATA_URL,{cache:'no-store'}),fetch('/api/memos',{cache:'no-store'})]);
      if(!p.ok||!m.ok) throw Error(`取得失敗 (JSON:${p.status}, メモ:${m.status})`);
      const data=await p.json(), memos=await m.json();
      if(!data || !Array.isArray(data.results) || !Array.isArray(memos)) throw Error('不正なJSON形式');
      if(data.dataset_memo_count!==48 || memos.length!==48) throw Error('48件の実験データではありません。サーバーのTHESIS_STUDY_DATA_FILEを確認してください。');
      const actual=new Map(memos.map(x=>[String(x.id),x]));
      if(actual.size!==48 || !Array.isArray(data.memo_identity) || data.memo_identity.length!==48)throw Error('メモIDの重複または不足');
      for(const src of data.memo_identity){const got=actual.get(src.id);if(!got||got.title!==src.title||got.content!==src.content)throw Error(`メモデータが分類時と一致しません: ${src.id}`);}
      if(data.results.length!==29 || data.candidate_total!==30) throw Error('分類件数が一致しません');
      for(const r of data.results){if(!actual.has(r.parent_id)||!actual.has(r.child_id)||!(r.relation in TYPE_LABEL)||r.relation==='none')throw Error('分類結果のIDまたは種類が不正');}
      memoById=actual;records=data.results;$('gm48-status').textContent=`48メモ／Gemma 4推定29関係（30候補中1件は「関係なし」）。AI推定・未承認。`;
      drawList();detail(0);
    }catch(e){records=[];$('gm48-list').replaceChildren();$('gm48-detail').textContent='分類結果を表示できません。';$('gm48-status').textContent=`確認エラー：${e.message}`;}
  }
  function hide(){overlay.hidden=true;previousFocus?.focus?.();}
  button.addEventListener('click',show);$('gm48-close').addEventListener('click',hide);
  $('gm48-filter').addEventListener('input',drawList);$('gm48-type').addEventListener('change',drawList);
  overlay.addEventListener('click',e=>{if(e.target===overlay)hide()});
  document.addEventListener('keydown',e=>{if(!overlay.hidden&&e.key==='Escape')hide()});
})();
