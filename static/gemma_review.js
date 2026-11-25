(() => {
  "use strict";

  const css = document.createElement("style");
  css.textContent = `
    #gemma-review-open {
      margin: 0 10px; padding: 8px 12px;
      border: 1px solid #64748b; border-radius: 8px;
      background: #334155; color: white; cursor: pointer;
    }
    #gemma-review-overlay {
      position: fixed; inset: 0; z-index: 99999;
      background: rgba(0,0,0,.78);
      display: flex; align-items: center; justify-content: center;
      padding: 16px; box-sizing: border-box;
    }
    #gemma-review-overlay[hidden] { display: none !important; }
    #gemma-review-panel {
      width: min(560px, 100%); max-height: 90vh;
      overflow-y: auto; background: #18212d;
      color: #f1f5f9; border: 1px solid #64748b;
      border-radius: 12px; padding: 20px;
      font-family: sans-serif; box-sizing: border-box;
    }
    #gemma-review-panel input,
    #gemma-review-panel select {
      width: 100%; padding: 9px; margin: 5px 0 12px;
      box-sizing: border-box; background: #0f172a;
      color: white; border: 1px solid #64748b;
    }
    #gemma-review-panel button {
      padding: 9px 14px; margin: 4px;
      border: 1px solid #64748b; border-radius: 6px;
      background: #334155; color: white; cursor: pointer;
    }
    #gemma-review-panel button:disabled {
      opacity: .4; cursor: not-allowed;
    }
    #gemma-review-result {
      white-space: pre-wrap; overflow-wrap: anywhere;
      padding: 12px; margin-top: 12px;
      border: 1px solid #475569; border-radius: 8px;
    }
  `;
  document.head.appendChild(css);

  const openButton = document.createElement("button");
  openButton.id = "gemma-review-open";
  openButton.type = "button";
  openButton.textContent = "Gemma関係分類";

  const header = document.querySelector("header");
  (header || document.body).appendChild(openButton);

  const overlay = document.createElement("div");
  overlay.id = "gemma-review-overlay";
  overlay.hidden = true;
  overlay.innerHTML = `
    <section id="gemma-review-panel" role="dialog"
      aria-modal="true" aria-label="Gemma関係分類">
      <h2>Gemma 4による思考関係の分類</h2>
      <p>対象メモを選び、判定後に内容を確認してください。</p>

      <label>管理用トークン</label>
      <input id="gemma-review-token" type="password"
        autocomplete="off" placeholder="管理用トークン">

      <label>対象メモを検索</label>
      <input id="gemma-review-filter" type="search"
        placeholder="タイトルまたはメモID">

      <label>対象メモ</label>
      <select id="gemma-review-memo"></select>

      <div>
        <button id="gemma-review-run" type="button">
          Gemmaで分類
        </button>
        <button id="gemma-review-close" type="button">
          閉じる
        </button>
      </div>

      <p id="gemma-review-status" aria-live="polite"></p>
      <div id="gemma-review-result" hidden></div>

      <button id="gemma-review-approve" type="button" disabled>
        この関係を承認して保存
      </button>
    </section>
  `;
  document.body.appendChild(overlay);

  const get = id => document.getElementById(id);
  const token = get("gemma-review-token");
  const filter = get("gemma-review-filter");
  const select = get("gemma-review-memo");
  const status = get("gemma-review-status");
  const result = get("gemma-review-result");
  const run = get("gemma-review-run");
  const approve = get("gemma-review-approve");

  let memos = [];
  let pending = null;

  function resetResult() {
    pending = null;
    result.hidden = true;
    result.textContent = "";
    approve.disabled = true;
  }

  function renderMemos() {
    const query = filter.value.trim().toLowerCase();
    const matches = memos.filter(m =>
      String(m.title || "").toLowerCase().includes(query) ||
      String(m.id || "").toLowerCase().includes(query)
    ).slice(0, 80);

    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "メモを選択してください";
    select.appendChild(placeholder);

    for (const memo of matches) {
      const opt = document.createElement("option");
      opt.value = String(memo.id);
      opt.textContent =
        String(memo.title || "(無題)").slice(0, 55) +
        " [" + memo.id + "]";
      select.appendChild(opt);
    }
    resetResult();
  }

  async function openPanel() {
    overlay.hidden = false;
    status.textContent = "メモを読み込んでいます";
    resetResult();
    try {
      const response = await fetch("/api/memos", {
        cache: "no-store"
      });
      if (!response.ok) throw new Error("メモ取得に失敗");
      const rows = await response.json();
      if (!Array.isArray(rows)) throw new Error("データ形式が不正");
      memos = rows.filter(x => x && x.id);
      renderMemos();
      status.textContent = `${memos.length}件のメモを読み込みました`;
    } catch (e) {
      status.textContent = e.message;
    }
  }

  openButton.addEventListener("click", openPanel);

  get("gemma-review-close").addEventListener("click", () => {
    overlay.hidden = true;
    token.value = "";
    resetResult();
  });

  filter.addEventListener("input", renderMemos);
  select.addEventListener("change", resetResult);

  run.addEventListener("click", async () => {
    resetResult();
    if (!token.value || !select.value) {
      status.textContent = "トークンと対象メモを指定してください";
      return;
    }

    run.disabled = true;
    status.textContent = "分類中です";

    try {
      const response = await fetch("/api/experiment-relations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-gemma-admin-token": token.value
        },
        body: JSON.stringify({memo_id: select.value})
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status}: ${data.error || "分類失敗"}`
        );
      }

      const item = (data.candidates || [])[0];
      if (!item) {
        status.textContent = "分類できる候補がありません";
        return;
      }

      pending = item;
      result.hidden = false;
      result.textContent =
        "旧メモ: " + (item.parent_title || item.parent_id) +
        "\n新メモID: " + item.child_id +
        "\n関係: " + item.relation +
        "\n信頼度: " + item.confidence +
        "\n理由: " + item.reason +
        "\n状態: 未承認";

      approve.disabled = item.relation === "none";
      status.textContent = "分類完了。内容を確認してください。";
    } catch (e) {
      status.textContent = e.message;
    } finally {
      run.disabled = false;
    }
  });

  approve.addEventListener("click", async () => {
    if (!pending || pending.relation === "none") return;

    const approved = window.confirm(
      "判定内容を確認しましたか？\n" +
      "この操作は保存済みメモの関係情報を変更します。\n" +
      "関係: " + pending.relation + "\n" +
      "承認して保存しますか？"
    );
    if (!approved) return;

    approve.disabled = true;
    status.textContent = "保存中です";

    try {
      const response = await fetch("/api/thought-links", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          parent_id: pending.parent_id,
          child_id: pending.child_id,
          relation: pending.relation
        })
      });
      const data = await response.json();
      if (!response.ok || data.status !== "success") {
        throw new Error(data.message || data.error || "保存失敗");
      }
      status.textContent = "承認して保存しました";
      window.location.reload();
    } catch (e) {
      status.textContent = e.message;
      approve.disabled = false;
    }
  });
})();