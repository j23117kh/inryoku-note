
(() => {
  "use strict";

  const nativeFetch = window.fetch.bind(window);
  const relationNames = {
    revision: "\u4fee\u6b63\u30fb\u7f6e\u63db",
    development: "\u767a\u5c55",
    branch: "\u5206\u5c90",
    related: "\u95a2\u9023"
  };
  const processed = new Set();

  function status(message) {
    let node = document.getElementById("auto-gemma-status");
    if (!node) {
      const saveButton = document.querySelector(
        "button[onclick*='saveMemo']"
      );
      if (!saveButton) return;
      node = document.createElement("div");
      node.id = "auto-gemma-status";
      node.style.cssText =
        "margin-top:6px;font-size:11px;color:#94a3b8;";
      saveButton.parentElement.insertAdjacentElement("afterend", node);
    }
    node.textContent = message;
  }

  function line(parent, text, css) {
    const p = document.createElement("div");
    p.textContent = text;
    if (css) p.style.cssText = css;
    parent.appendChild(p);
    return p;
  }

  function button(text, parent, action) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.style.cssText =
      "padding:8px 13px;border:1px solid #475569;" +
      "border-radius:8px;background:#1e293b;color:#f8fafc;";
    b.addEventListener("click", action);
    parent.appendChild(b);
    return b;
  }

  function awaitOldModal() {
    const old = document.getElementById("proposal-modal");
    if (!old || old.classList.contains("hidden")) {
      return Promise.resolve();
    }
    return new Promise(resolve => {
      const observer = new MutationObserver(() => {
        if (old.classList.contains("hidden")) {
          observer.disconnect();
          resolve();
        }
      });
      observer.observe(old, {
        attributes: true,
        attributeFilter: ["class"]
      });
    });
  }

  async function showCandidates(data) {
    await awaitOldModal();

    // Existing manual approvals always take precedence.
    let approved = new Set();
    try {
      const r = await nativeFetch("/api/memos");
      if (r.ok) {
        const memos = await r.json();
        const child = memos.find(
          m => String(m.id) === String(data.memo_id)
        );
        if (child) {
          for (const id of child.supersedes || []) {
            approved.add(String(id));
          }
          for (const link of child.thought_links || []) {
            approved.add(String(link.parent_id));
          }
        }
      }
    } catch (e) {
      console.warn("Relation status check:", e);
    }

    const candidates = (data.candidates || []).filter(
      c => !approved.has(String(c.parent_id))
    );
    if (!candidates.length) {
      status("\u601d\u8003\u95a2\u4fc2\u306e\u8ffd\u52a0\u5019\u88dc\u306a\u3057");
      return;
    }

    const overlay = document.createElement("div");
    overlay.id = "auto-gemma-proposals";
    overlay.style.cssText =
      "position:fixed;inset:0;z-index:100;" +
      "display:flex;align-items:center;justify-content:center;" +
      "background:rgba(0,0,0,.85);padding:16px;";

    const panel = document.createElement("section");
    panel.style.cssText =
      "width:min(94vw,560px);max-height:85vh;overflow:auto;" +
      "background:#0f172a;border:1px solid #475569;" +
      "border-radius:14px;padding:20px;color:#f8fafc;";
    overlay.appendChild(panel);

    line(
      panel,
      "Gemma 4 \u306e\u601d\u8003\u95a2\u4fc2\u63d0\u6848",
      "font-size:17px;font-weight:600;margin-bottom:8px;"
    );
    line(
      panel,
      "\u5019\u88dc\u306f\u672a\u627f\u8a8d\u3067\u3059\u3002" +
      "\u627f\u8a8d\u3057\u305f\u95a2\u4fc2\u3060\u3051\u4fdd\u5b58\u3057\u307e\u3059\u3002",
      "font-size:12px;color:#94a3b8;margin-bottom:14px;"
    );

    for (const c of candidates) {
      const row = document.createElement("div");
      row.dataset.autoGemmaCandidate = "1";
      row.style.cssText =
        "border:1px solid #334155;border-radius:10px;" +
        "padding:12px;margin-bottom:10px;";
      panel.appendChild(row);
      line(row, c.parent_title || c.parent_id,
        "font-size:14px;font-weight:600;margin-bottom:5px;");
      line(
        row,
        (relationNames[c.relation] || c.relation)
          + " / confidence " + c.confidence.toFixed(2)
          + " / cosine " + c.cosine.toFixed(3),
        "font-size:12px;color:#a7f3d0;margin-bottom:6px;"
      );
      line(row, c.reason, "font-size:12px;margin-bottom:10px;");

      if (c.low_confidence) {
        line(
          row,
          "\u4fe1\u983c\u5ea60.75\u672a\u6e80\uff1a\u614e\u91cd\u306b\u78ba\u8a8d",
          "font-size:11px;color:#fbbf24;margin-bottom:8px;"
        );
      }

      const approve = button(
        "\u3053\u306e\u95a2\u4fc2\u3092\u627f\u8a8d",
        row,
        async () => {
          approve.disabled = true;
          try {
            const response = await nativeFetch("/api/thought-links", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({
                parent_id: c.parent_id,
                child_id: c.child_id,
                relation: c.relation
              })
            });
            const result = await response.json();
            if (!response.ok || result.status !== "success") {
              throw new Error(
                result.message || "\u627f\u8a8d\u5931\u6557"
              );
            }
            row.remove();
            if (!panel.querySelector('[data-auto-gemma-candidate="1"]')) {
              overlay.remove();
            }
            status("\u601d\u8003\u95a2\u4fc2\u3092\u627f\u8a8d\u30fb\u4fdd\u5b58\u3057\u307e\u3057\u305f");
            if (typeof window.loadArchive === "function") {
              await window.loadArchive();
            }
            if (typeof window.refreshApprovedThoughtGraph === "function") {
              await window.refreshApprovedThoughtGraph();
            }
            if (!panel.querySelector("div[style*='padding: 12px']")) {
              // User may still close the dialog manually.
            }
          } catch (e) {
            status(String(e.message || e));
            approve.disabled = false;
          }
        }
      );
      approve.style.background = "#047857";
    }

    button("\u9589\u3058\u308b\uff08\u672a\u627f\u8a8d\u306e\u307e\u307e\uff09",
      panel, () => overlay.remove());

    document.body.appendChild(overlay);
    status(
      "\u601d\u8003\u95a2\u4fc2\u306e\u5019\u88dc " +
      candidates.length + " \u4ef6"
    );
  }

  async function suggest(memoId) {
    if (!memoId || processed.has(memoId)) return;
    processed.add(memoId);
    status("Gemma 4 \u3067\u601d\u8003\u95a2\u4fc2\u3092\u5206\u985e\u4e2d...");
    try {
      const r = await nativeFetch("/api/auto-relation-suggest", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({memo_id: memoId})
      });
      const data = await r.json();
      if (!r.ok) {
        status(
          "\u30e1\u30e2\u306f\u4fdd\u5b58\u6e08\u307f\u3002" +
          "Gemma 4: " + (data.error || r.status)
        );
        return;
      }
      if (!data.candidates || !data.candidates.length) {
        status("\u30e1\u30e2\u4fdd\u5b58\u5b8c\u4e86\u3002\u8ffd\u52a0\u306e\u601d\u8003\u95a2\u4fc2\u5019\u88dc\u306a\u3057");
        return;
      }
      await showCandidates(data);
    } catch (e) {
      status(
        "\u30e1\u30e2\u306f\u4fdd\u5b58\u6e08\u307f\u3002" +
        "Gemma 4: " + String(e.message || e)
      );
    }
  }

  window.fetch = async function(input, init) {
    const response = await nativeFetch(input, init);
    const rawUrl = typeof input === "string"
      ? input : input && input.url;
    const method = String(
      (init && init.method) ||
      (typeof input === "object" && input.method) ||
      "GET"
    ).toUpperCase();

    try {
      const url = new URL(rawUrl, location.href);
      if (url.pathname === "/api/save" &&
          method === "POST" && response.ok) {
        response.clone().json().then(data => {
          if (data.status === "success" && data.id) {
            void suggest(String(data.id));
          }
        }).catch(console.error);
      }
    } catch (e) {
      console.warn("Auto suggest hook:", e);
    }
    return response;
  };
})();
