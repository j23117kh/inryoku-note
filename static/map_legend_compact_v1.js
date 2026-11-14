/* THESIS_MAP_LEGEND_COMPACT_V1
 * UI-only fix: move long in-map instructions into a small, optional popover,
 * and give the graph usable height when the memo focus selector is present.
 * Does not change rank, query, memo, graph edges or search engine.
 */
(() => {
  'use strict';
  if (window.__thesisMapLegendCompact) return;
  const map = document.getElementById('view-map');
  if (!map) return;
  const legend = document.getElementById('map-legend');
  const count = document.getElementById('map-count');
  const helpPanel = legend?.parentElement;
  if (!legend || !count || !helpPanel || !map.contains(helpPanel)) {
    console.warn('THESIS_MAP_LEGEND_COMPACT_V1: map legend missing; no layout change');
    return;
  }

  const css = document.createElement('style');
  css.id = 'thesis-map-legend-compact-v1-style';
  css.textContent = `
    /* Preserve a usable graph when the new focus selector consumes panel height. */
    html body > main #result-panel.map-focused {
      overflow-y: auto !important;
      overflow-x: hidden !important;
      overscroll-behavior-y: contain;
    }
    html body > main #result-panel.map-focused #view-map:not(:fullscreen) {
      flex: 0 0 clamp(350px, 50dvh, 480px) !important;
      height: clamp(350px, 50dvh, 480px) !important;
      min-height: 350px !important;
      overflow: hidden !important;
    }
    #view-map #map-compact-help {
      position: absolute;
      z-index: 28;
      right: 10px;
      bottom: 10px;
      padding: 6px 11px;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.4;
      color: #f1f5f9;
      background: rgba(15, 23, 42, .96);
      border: 1px solid #64748b;
      border-radius: 8px;
      box-shadow: 0 5px 16px rgba(0,0,0,.4);
      cursor: pointer;
      pointer-events: auto;
    }
    #view-map #map-compact-help:hover,
    #view-map #map-compact-help:focus-visible {
      background: #1e293b;
      outline: 2px solid #10b981;
      outline-offset: 2px;
    }
    #view-map #map-compact-help-panel {
      display: none !important;
    }
    #view-map #map-compact-help-panel.is-open {
      display: flex !important;
      position: absolute !important;
      top: auto !important;
      bottom: 48px !important;
      left: 8px !important;
      right: 8px !important;
      z-index: 27 !important;
      width: auto !important;
      max-height: min(52%, 220px) !important;
      padding: 10px !important;
      flex-direction: column !important;
      gap: 6px !important;
      overflow: auto !important;
      pointer-events: auto !important;
      background: rgba(2,6,23,.97) !important;
      border: 1px solid #475569 !important;
      border-radius: 10px !important;
      box-shadow: 0 12px 30px rgba(0,0,0,.65) !important;
      color: #f1f5f9 !important;
    }
    #view-map #map-compact-help-panel > span {
      display: block !important;
      width: auto !important;
      max-width: 100% !important;
      padding: 0 !important;
      margin: 0 !important;
      background: transparent !important;
      border: 0 !important;
      font-size: 11px !important;
      line-height: 1.5 !important;
      font-weight: normal !important;
      overflow-wrap: anywhere !important;
      white-space: normal !important;
    }
    #view-map:fullscreen #map-compact-help {bottom: 14px; right: 14px;}
    #view-map:fullscreen #map-compact-help-panel.is-open {bottom: 56px !important;}
  `;
  document.head.appendChild(css);

  helpPanel.id = 'map-compact-help-panel';
  helpPanel.setAttribute('role', 'region');
  helpPanel.setAttribute('aria-label', '関係図の凡例と操作');
  // The old footer remains in the DOM. Its contents continue to update from drawMap.
  const button = document.createElement('button');
  button.type = 'button';
  button.id = 'map-compact-help';
  button.textContent = '？ 凡例';
  button.setAttribute('aria-controls', helpPanel.id);
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-label', '関係図の凡例と操作方法を表示');
  map.appendChild(button);

  const setOpen = (open) => {
    helpPanel.classList.toggle('is-open', !!open);
    button.setAttribute('aria-expanded', String(!!open));
    button.textContent = open ? '× 閉じる' : '？ 凡例';
    button.setAttribute('aria-label', open ? '凡例を閉じる' : '関係図の凡例と操作方法を表示');
  };
  button.addEventListener('click', () => setOpen(!helpPanel.classList.contains('is-open')));
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && helpPanel.classList.contains('is-open')) setOpen(false);
  });
  map.addEventListener('click', e => {
    if (helpPanel.classList.contains('is-open') &&
        !helpPanel.contains(e.target) && !button.contains(e.target)) setOpen(false);
  });

  window.__thesisMapLegendCompact = Object.freeze({version: '1', close: () => setOpen(false)});
})();
