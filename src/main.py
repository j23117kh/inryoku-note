import datetime
import os
from time import perf_counter
import uuid
import numpy as np
import uvicorn
import networkx as nx
from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles

from src.config import (
    ASK_MIN_SCORE,
    ASK_TOP_K,
    STATIC_DIR,
    UI_FILE,
    SUPERSEDE_CANDIDATES,
    SUPERSEDE_MIN_SIM,
)

from src.data_manager import (
    init_data_file,
    load_memos,
    save_memos,
    live_memos,
    superseded_map,
)

from src.thought_links import set_link, remove_link, relationship_graph

from src.embedding import get_vector

from src.memo_analyzer import (
    analyze_memo,
    gen_json,
)

from src.recommender import search_core
from src.search_policy import (
    APP_DEFAULT_THRESHOLD_PERCENT, APP_DEFAULT_TOP_K,
    GRAVITY_DEFAULT_MODE, resolve_gravity_mode,
)

from src.similarity import parse_date
from src.query_intent import classify_query
from src.intent_search import classify_and_search
from src.integrated_search import SCORING_VERSION, integrated_search, resolve_search_intent


# =========================
# FastAPI
# =========================

app = FastAPI()

# In an isolated participant session, only the predeclared integrated query API
# may rank data. All write routes and alternative query modes are blocked so a
# participant cannot accidentally alter the fixed dataset or scoring method.
_STUDY_BLOCKED_POSTS = frozenset({
    "/api/save", "/api/delete", "/api/confirm", "/api/revert",
    "/api/thought-links", "/api/thought-links/remove",
    "/api/search", "/api/gravity", "/api/ask",
    "/api/intent-search", "/api/intent-classify",
})


@app.middleware("http")
async def freeze_study_requests(request: Request, call_next):
    if (os.getenv("THESIS_STUDY_SEARCH_LOCK") == "1"
            and request.method == "POST"
            and request.url.path != "/api/integrated-search"):
        return JSONResponse(status_code=403, content={
            "status": "error", "message": "参加者実験では固定の統合検索だけ使用できます"
        })
    return await call_next(request)
# ⬇️⬇️⬇️ ここに以下の関数をまるごと追加 ⬇️⬇️⬇️
def is_safe_from_cycles(memos, new_memo_id=None, new_supersedes=None):
    """保存前にA->B->Aのループが発生しないか検証する"""
    graph = nx.DiGraph()
    for memo in memos:
        graph.add_node(memo["id"])
        
    for memo in memos:
        curr_id = memo["id"]
        edges_to_add = new_supersedes if curr_id == new_memo_id else memo.get("supersedes", [])
        for old_id in edges_to_add:
            if old_id in graph.nodes:
                graph.add_edge(old_id, curr_id)
                
    return nx.is_directed_acyclic_graph(graph)
# ⬆️⬆️⬆️ 追加ここまで ⬆️⬆️⬆️


def _same_memo_body(first, second):
    """Reject duplicate bodies as new revisions (preserve source records).

    Ignore whitespace-only differences; this is only a gate for a *new*
    supersedes relation, not an automatic migration of existing relations.
    """
    if not isinstance(first, str) or not isinstance(second, str):
        return False
    first, second = " ".join(first.split()), " ".join(second.split())
    return bool(first) and first == second


# =========================
# データ初期化
# =========================

init_data_file()


# =========================
# Static
# =========================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static",
    )


# =========================
# UI
# =========================

@app.get("/")
async def get_ui():

    if UI_FILE.exists():

        with open(
            UI_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            return HTMLResponse(
                f.read()
            )

    return HTMLResponse(
        "index.html が見つかりません。"
    )


# =========================
# メモ保存
# =========================

@app.post("/api/save")
async def save_memo(
    request: Request
):

    data = await request.json()

    title = (
        data.get(
            "title",
            ""
        ).strip()
        or "無題の思考"
    )

    content = data.get(
        "content",
        ""
    ).strip()

    # Never apply AI-generated revisions without a subsequent /api/confirm.
    # Ignore legacy clients that still send auto_apply=true.
    auto_apply = False

    if not content:

        return {
            "status": "error",
            "message": "内容が空です",
        }

    # =========================
    # ベクトル化
    # =========================

    vector = get_vector(
        content,
        "RETRIEVAL_DOCUMENT",
        title,
    )

    if vector is None:

        return {
            "status": "error",
            "message": "AI計算に失敗しました",
        }

    memos = load_memos()

    now = datetime.datetime.now()

    # =========================
    # 更新候補
    # =========================

    pool = live_memos(memos)

    superseded_by = superseded_map(
        pool
    )

    active = [
        memo
        for memo in pool
        if (
            "vector" in memo
            and memo["id"]
            not in superseded_by
            and not _same_memo_body(memo.get("content", ""), content)
        )
    ]

    candidates = []

    if active:

        similarities = (
            np.array(
                [
                    memo["vector"]
                    for memo in active
                ]
            )
            @ vector
        )

        indices = np.argsort(
            -similarities
        )[
            :SUPERSEDE_CANDIDATES
        ]

        for index in indices:

            if (
                similarities[index]
                >= SUPERSEDE_MIN_SIM
            ):

                candidates.append(
                    active[index]
                )

    # =========================
    # Geminiによる解析
    # =========================

    event_date, suggestions = (
        analyze_memo(
            content,
            now.date().isoformat(),
            candidates,
        )
    )

    # Defense in depth: only non-duplicate candidates may be proposed.
    candidate_ids = {memo["id"] for memo in candidates}
    suggestions = [
        item for item in suggestions
        if isinstance(item, dict) and item.get("id") in candidate_ids
    ]

    # =========================
    # 新メモ
    # =========================

    new_memo = {

        "id": str(
            uuid.uuid4()
        ),

        "date": now.strftime(
            "%Y/%m/%d"
        ),

        "created_at":
            now.isoformat(
                timespec="seconds"
            ),

        "event_date":
            event_date,

        # Suggestions are never authoritative until explicitly approved.
        "supersedes": [],
        "suggested": suggestions,

        "status": "active",

        "thought_links": [],

        "title": title,

        "content": content,

        "vector": vector.tolist(),
    }
    # Cycles and duplicate updates are checked again at confirmation time.
    memos.insert(0,new_memo)

    save_memos(memos)

    # =========================
    # 提案情報
    # =========================

    by_id = {
        memo["id"]: memo
        for memo in memos
    }

    proposals = []

    for suggestion in suggestions:

        memo = by_id[
            suggestion["id"]
        ]

        proposals.append(
            {
                "id":
                    suggestion["id"],

                "reason":
                    suggestion["reason"],

                "date":
                    memo["date"],

                "title":
                    memo["title"],

                "content":
                    memo["content"][:120],
            }
        )

    return {

        "status":
            "success",

        "id":
            new_memo["id"],

        "event_date":
            event_date,

        "applied": False,
        "auto_applied": False,  # legacy client compatibility

        "proposals":
            proposals,
    }


# =========================
# 更新提案の承認
# =========================

@app.post("/api/confirm")
async def confirm(
    request: Request
):

    data = await request.json()

    memo_id = data.get(
        "id"
    )

    # A confirmation request is an explicit user action. Reject malformed
    # inputs rather than treating strings as character sets or silently
    # accepting unknown/deleted/future parents.
    raw_accept = data.get("accept", [])
    if not isinstance(raw_accept, list) or any(
        not isinstance(item, str) or not item for item in raw_accept
    ):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "accept はメモIDの配列で指定してください。"},
        )
    accept = set(raw_accept)

    memos = load_memos()

    for memo in memos:

        if memo["id"] != memo_id:
            continue

        allowed = {
            item["id"]
            for item in (memo.get("suggested") or [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if not accept <= allowed:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "承認対象に未提案のメモが含まれています。"},
            )

        chosen = accept
        by_id = {item["id"]: item for item in memos}
        # validate_link checks existence, deleted state, chronology and the
        # entire combined directed graph (supersedes + thought_links).
        from src.thought_links import validate_link
        try:
            for old_id in sorted(chosen):
                validate_link(memos, memo_id, old_id, "revision")
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
        duplicates = [
            old_id for old_id in chosen
            if old_id in by_id
            and _same_memo_body(memo.get("content", ""), by_id[old_id].get("content", ""))
        ]
        if duplicates:
            return {
                "status": "error",
                "message": "本文が同じメモは修正・更新として承認できません。",
            }

        new_supersedes = sorted(
            list(set(memo.get("supersedes", [])) | chosen)
        )

        if not is_safe_from_cycles(memos, memo_id, new_supersedes):
            return {
                "status": "error",
                "message": "循環参照が発生するため、この更新は承認できません。"
            }

        # An accepted revision must have one authoritative representation in
        # both stored fields. The graph falls back to legacy supersedes-only
        # records, but new approvals always persist the matching thought link.
        # Validation (including combined-graph cycles) ran above for every ID.
        for old_id in sorted(chosen):
            set_link(memos, memo_id, old_id, "revision")

        memo["supersedes"] = new_supersedes
        memo["suggested"] = []

        save_memos(memos)

        return {
            "status":
                "success",

            "supersedes":
                memo["supersedes"],
        }

    return {
        "status":
            "error",

        "message":
            "メモが見つかりません",
    }


# =========================
# 更新関係の取り消し
# =========================

@app.post("/api/revert")
async def revert(
    request: Request
):

    data = await request.json()
    memo_id = data.get("id") if isinstance(data, dict) else None
    old_id = data.get("old_id") if isinstance(data, dict) else None
    if not isinstance(memo_id, str) or not memo_id or not isinstance(old_id, str) or not old_id:
        return JSONResponse(status_code=400, content={
            "status": "error", "message": "id と old_id はメモIDの文字列で指定してください。"
        })

    memos = load_memos()
    for memo in memos:
        if memo.get("id") != memo_id:
            continue
        supersedes = list(memo.get("supersedes") or [])
        links = list(memo.get("thought_links") or [])
        # Revert the *revision* for precisely this pair. Do not erase a
        # development/branch/related link that may coexist with legacy data.
        old_revision = [
            link for link in links
            if isinstance(link, dict)
            and link.get("parent_id") == old_id
            and link.get("relation") == "revision"
        ]
        if old_id not in supersedes and not old_revision:
            break
        memo["supersedes"] = [parent for parent in supersedes if parent != old_id]
        memo["thought_links"] = [
            link for link in links
            if not (
                isinstance(link, dict)
                and link.get("parent_id") == old_id
                and link.get("relation") == "revision"
            )
        ]
        save_memos(memos)
        return {"status": "success"}

    return {"status": "error", "message": "該当の更新関係がありません"}


# =========================
# 全メモ取得
# =========================

@app.get("/api/memos")
async def get_all_memos():

    memos = load_memos()

    pool = live_memos(
        memos
    )

    superseded_by = (
        superseded_map(
            pool
        )
    )

    keys = (
        "id",
        "date",
        "title",
        "content",
        "created_at",
        "event_date",
        "supersedes",
        "suggested",
        "thought_links",
    )

    result = []

    for memo in pool:

        item = {
            key:
                memo.get(key)
            for key in keys
        }

        item["stale"] = (
            memo["id"]
            in superseded_by
        )

        item["superseded_by"] = (
            superseded_by.get(
                memo["id"]
            )
        )

        result.append(item)

    return result


# =========================
# メモ削除
# =========================

@app.post("/api/delete")
async def delete_memo(
    request: Request
):

    data = await request.json()

    memo_id = data.get(
        "id"
    )

    memos = load_memos()

    hit = False

    for memo in memos:

        if memo["id"] != memo_id:
            continue

        memo["status"] = "deleted"

        memo["content"] = ""

        memo["title"] = "(削除済み)"

        memo.pop(
            "vector",
            None
        )

        hit = True

    if hit:
        save_memos(memos)

    return {
        "status":
            "success"
            if hit
            else "error"
    }


# =========================
# 検索
# =========================

@app.post("/api/search")
async def search(
    request: Request
):

    data = await request.json()

    result = search_core(

        data.get(
            "text",
            ""
        ).strip(),

        data.get(
            "mode",
            "semantic_lineage"
        ),

        float(
            data.get(
                "threshold",
                APP_DEFAULT_THRESHOLD_PERCENT
            )
        ) / 100.0,

        int(
            data.get(
                "top_k",
                APP_DEFAULT_TOP_K
            )
        ),

        parse_date(
            data.get(
                "as_of"
            )
        ),

        parse_date(
            data.get(
                "time_from"
            )
        ),

        parse_date(
            data.get(
                "time_to"
            )
        ),
    )

    if result is None:

        return JSONResponse(
            {
                "error":
                    "クエリのベクトル化に失敗"
            },
            status_code=500,
        )

    return result


# =========================
# 質問目的の自動判定＋保存関係の検索（既存方式とは独立）
# =========================

@app.post("/api/intent-classify")
async def api_intent_classify(request: Request):
    """Rule-based intent baseline, with no labels or embedding API calls."""
    data = await request.json()
    try:
        return classify_query(data.get("text", ""), as_of=data.get("as_of"))
    except (ValueError, TypeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/intent-search")
async def api_intent_search(request: Request):
    """Hybrid + bounded 3-hop *saved* lineage, query text drives intent.

    Does not use oracle relationships, hidden labels or artificial metadata.
    Does not mutate the database or alter /api/gravity's legacy behavior.
    """
    data = await request.json()
    try:
        query = data.get("text", "")
        classified = classify_query(query, as_of=data.get("as_of"))
        top_k = data.get("top_k", APP_DEFAULT_TOP_K)
        threshold = data.get("threshold", APP_DEFAULT_THRESHOLD_PERCENT)
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k は整数を指定してください")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold は数値を指定してください")
        threshold = float(threshold) / 100.0
        if not (0 <= threshold <= 1):
            raise ValueError("threshold は0～100を指定してください")
        if not 1 <= top_k <= 30:
            raise ValueError("top_k は1～30の整数を指定してください")
        memos = load_memos()
        # No API call for an empty memo database.
        if not any(m.get('status') != 'deleted' and m.get('vector') is not None for m in memos):
            return {"query": query, "classification": classified,
                    "results": [], "total_candidates": 0,
                    "scoring": "hybrid_saved_3hop_v1"}
        vector = get_vector(query, "RETRIEVAL_QUERY")
        if vector is None:
            return JSONResponse(status_code=503,
                                content={"error": "質問のベクトル化に失敗しました。API利用枠や設定を確認してください。"})
        return classify_and_search(query, memos=memos, query_vector=vector,
                                   as_of=data.get("as_of"), top_k=top_k,
                                   threshold=threshold)
    except (ValueError, TypeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})



# =========================
# Question-aware integrated Hybrid (independent from v4 experiment and legacy modes)
# =========================
@app.post("/api/integrated-search")
async def api_integrated_search(request: Request):
    """Semantic+BM25, saved graph (3 hops), optional event date, current-only stale penalty.

    Nothing is written to the memo database; existing /api/gravity and
    /api/intent-search semantics are preserved for comparisons.
    """
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("JSONのオブジェクト形式で指定してください")
        query = data.get("text", "")
        as_of = data.get("as_of")
        # Only in explicitly isolated study sessions: reject deviations from
        # the frozen question-only protocol. Never inject the private gold date.
        if os.getenv("THESIS_STUDY_SEARCH_LOCK") == "1":
            if as_of not in (None, "") or data.get("top_k", 10) != 10 or data.get("threshold", 30) != 30:
                raise ValueError("参加者実験では日時の手入力・top_k・閾値の変更はできません")
            as_of = None
            top_k = 10
            threshold = 30.0
        else:
            top_k = data.get("top_k", APP_DEFAULT_TOP_K)
            threshold = data.get("threshold", APP_DEFAULT_THRESHOLD_PERCENT)
        classified = resolve_search_intent(query, as_of=as_of)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 30:
            raise ValueError("top_k は1～30の整数で指定してください")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold は0～100の数値で指定してください")
        threshold = float(threshold) / 100.0
        if not 0 <= threshold <= 1:
            raise ValueError("threshold は0～100の数値で指定してください")
        memos = load_memos()
        if not any(m.get("status") != "deleted" and m.get("vector") is not None for m in memos):
            return {"query": query, "classification": classified, "results": [],
                    "total_candidates": 0, "scoring": SCORING_VERSION}
        vector = get_vector(query, "RETRIEVAL_QUERY")
        if vector is None:
            return JSONResponse(status_code=503, content={
                "error": "質問の埋め込みを取得できませんでした。API設定と利用枠を確認してください。"})
        return integrated_search(query, memos=memos, query_vector=vector,
                                 as_of=as_of, top_k=top_k, threshold=threshold)
    except (ValueError, TypeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

# =========================
# 質問応答
# =========================

@app.post("/api/ask")
async def ask(
    request: Request
):

    data = await request.json()

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return {
            "status":
                "error",

            "message":
                "質問が空です",
        }

    search_started = perf_counter()
    hits = search_core(

        question,

        "semantic_lineage",

        ASK_MIN_SCORE,

        ASK_TOP_K,

        parse_date(
            data.get(
                "as_of"
            )
        ),

        parse_date(
            data.get(
                "time_from"
            )
        ),

        parse_date(
            data.get(
                "time_to"
            )
        ),
    )

    search_elapsed = perf_counter() - search_started
    print(f"[api/ask] search_seconds={search_elapsed:.3f}", flush=True)

    if hits is None:

        return JSONResponse(
            {
                "error":
                    "クエリのベクトル化に失敗"
            },
            status_code=500,
        )

    if not hits:

        return {
            "status":
                "success",

            "answer":
                "関連するメモが見つかりませんでした。",

            "evidence": [],
        }

    evidence_lines = []
    for hit in hits:
        status_str = "【古い情報(更新済み)】" if hit["stale"] else "【有効】"
        sup_str = f"\n  ※注意: 過去のメモ(ID: {', '.join(hit.get('supersedes', []))})を上書き・訂正しています。" if hit.get("supersedes") else ""
        rel_str = f"\n  ※関連: ID {', '.join(hit.get('related_ids', []))} のメモと強い関連があります。" if hit.get("related_ids") else ""
        
        line = (
            f'- id={hit["id"]} / 作成={hit["created_at"][:10]} / 出来事日={hit["event_date"]} / {status_str}'
            f'{sup_str}{rel_str}\n'
            f'  内容={hit["content"][:300]}'
        )
        evidence_lines.append(line)
        
    evidence_text = "\n".join(evidence_lines)

    prompt = f"""
あなたは個人メモに基づいて答えるアシスタントです。
JSONだけを返してください。

質問:
{question}

メモ:
{evidence_text}

出力形式:

{{
  "answer": "回答(日本語、簡潔に)",
  "used_ids": [
    "根拠にしたメモid"
  ],
  "outdated_note": "古い情報に触れた場合のみ一言、なければ空文字"
}}

ルール:

- 回答は【有効】なメモだけを根拠にする。
- 【古い情報】は回答の根拠にはしない。
- メモに答えがなければ分からないと答える。
- 推測で補わない。
- used_idsはメモに存在するidだけ。
"""

    generation_started = perf_counter()
    try:
        output = gen_json(prompt)
    except Exception as e:
        generation_elapsed = perf_counter() - generation_started
        provider_code = getattr(e, "code", None)
        if not isinstance(provider_code, int):
            provider_code = None
        # 質問文・APIキー・エラー本文はログへ出さない。
        print(
            f"[api/ask] generation_seconds={generation_elapsed:.3f} "
            f"status=failed provider_code={provider_code} "
            f"exception={type(e).__name__}",
            flush=True,
        )
        if provider_code == 503:
            status, error_code, message = (
                503, "GEMINI_UNAVAILABLE",
                "Geminiの回答生成サービスが一時的に混雑しています。時間をおいて再試行してください。",
            )
        elif provider_code == 429:
            status, error_code, message = (
                429, "GEMINI_RATE_LIMIT",
                "Geminiの利用上限に達しました。時間をおいて再試行してください。",
            )
        elif provider_code in (401, 403):
            status, error_code, message = (
                502, "GEMINI_AUTH_ERROR",
                "Geminiへの接続設定に問題があります。サーバーの設定を確認してください。",
            )
        elif provider_code == 404:
            status, error_code, message = (
                502, "GEMINI_MODEL_NOT_FOUND",
                "指定されたGeminiモデルを利用できません。モデル設定を確認してください。",
            )
        elif provider_code in (500, 502, 504):
            status, error_code, message = (
                502, "GEMINI_PROVIDER_ERROR",
                "Gemini側でエラーが発生しました。時間をおいて再試行してください。",
            )
        elif isinstance(e, (ValueError, TypeError, KeyError, AttributeError)):
            status, error_code, message = (
                502, "GEMINI_INVALID_RESPONSE",
                "Geminiの回答を正しく読み取れませんでした。もう一度お試しください。",
            )
        else:
            status, error_code, message = (
                500, "ANSWER_GENERATION_ERROR",
                "回答の生成に失敗しました。サーバーログを確認してください。",
            )
        return JSONResponse(
            {"status": "error", "error": message, "code": error_code},
            status_code=status,
        )

    generation_elapsed = perf_counter() - generation_started
    print(
        f"[api/ask] generation_seconds={generation_elapsed:.3f} status=success",
        flush=True,
    )
    if not isinstance(output, dict):
        return JSONResponse(
            {
                "status": "error",
                "error": "Geminiの回答形式が不正です。もう一度お試しください。",
                "code": "GEMINI_INVALID_RESPONSE",
            },
            status_code=502,
        )

    valid_ids = {
        hit["id"]
        for hit in hits
    }

    used_ids = [
        item
        for item
        in output.get(
            "used_ids",
            []
        )
        if item in valid_ids
    ]

    return {

        "status":
            "success",

        "answer":
            output.get(
                "answer",
                ""
            ),

        "outdated_note":
            output.get(
                "outdated_note",
                ""
            ),

        "evidence":
            [
                hit
                for hit in hits
                if hit["id"]
                in used_ids
            ],

        "only_stale_found":
            all(
                hit["stale"]
                for hit in hits
            ),
    }


# =========================
# 旧UI互換
# =========================

@app.post("/api/gravity")
async def gravity_compat(
    request: Request
):

    data = await request.json()

    # UNIFIED_V1_GRAVITY: draft memo uses the production integrated ranker.
    if isinstance(data, dict) and data.get("mode") == "integrated_intent_v1":
        try:
            query = data.get("text", "")
            if not isinstance(query, str):
                raise ValueError("text must be a string")
            query = query.strip()
            if not query:
                return []

            percent = data.get("threshold", APP_DEFAULT_THRESHOLD_PERCENT)
            if isinstance(percent, bool) or not isinstance(percent, (int, float)):
                raise ValueError("threshold must be numeric")
            percent = float(percent)
            if not 0 <= percent <= 100:
                raise ValueError("threshold must be between 0 and 100")

            memos = load_memos()
            if not any(
                m.get("status") != "deleted" and m.get("vector") is not None
                for m in memos
            ):
                return []

            vector = get_vector(query, "RETRIEVAL_QUERY")
            if vector is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Embedding unavailable"}
                )

            response = integrated_search(
                query,
                memos=memos,
                query_vector=vector,
                as_of=data.get("as_of"),
                top_k=APP_DEFAULT_TOP_K,
                threshold=percent / 100.0,
            )
            return response["results"]

        except (ValueError, TypeError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)}
            )

    mode = resolve_gravity_mode(data.get("mode", GRAVITY_DEFAULT_MODE))

    result = search_core(

        data.get(
            "text",
            ""
        ).strip(),

        mode,

        float(
            data.get(
                "threshold",
                APP_DEFAULT_THRESHOLD_PERCENT
            )
        ) / 100.0,
        APP_DEFAULT_TOP_K,
    )

    if result is None:

        return JSONResponse(
            {
                "error":
                    "ベクトル化に失敗"
            },
            status_code=500,
        )
    
    return result



# Confirmed thought connections (independent of similarity-based suggestions).
@app.post("/api/thought-links")
async def save_thought_link(request: Request):
    data = await request.json()
    memos = load_memos()
    try:
        if data.get("relation") == "revision":
            by_id = {item["id"]: item for item in memos}
            parent = by_id.get(data.get("parent_id"))
            child = by_id.get(data.get("child_id"))
            if parent and child and _same_memo_body(parent.get("content"), child.get("content")):
                raise ValueError("本文が同じメモは修正・更新として承認できません。")
        links = set_link(memos, data.get("child_id"), data.get("parent_id"), data.get("relation"))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
    save_memos(memos)
    return {"status": "success", "thought_links": links}


@app.post("/api/thought-links/remove")
async def delete_thought_link(request: Request):
    data = await request.json()
    memos = load_memos()
    try:
        links = remove_link(memos, data.get("child_id"), data.get("parent_id"))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
    save_memos(memos)
    return {"status": "success", "thought_links": links}


# BEGIN SEARCH_PATHS_V1
from src.search_paths import build_search_path


@app.get("/api/search-paths/{memo_id}")
async def get_search_path(memo_id: str, hops: int = 2, as_of: str | None = None):
    """Read-only local view of saved memo relations. Does not embed or rerank."""
    try:
        cutoff = None
        if as_of is not None:
            parsed = parse_date(as_of)
            if parsed is None:
                raise ValueError("as_of の日付形式が不正です")
            cutoff = parsed.date()
        return build_search_path(load_memos(), memo_id, hops=hops, as_of=cutoff)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})


# END SEARCH_PATHS_V1

@app.get("/api/thought-graph")
async def get_thought_graph(memo_id: str | None = None, hops: int = 3):
    try:
        graph = relationship_graph(load_memos(), memo_id, hops)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
    return graph

from src.recommender import get_multi_hop_lineage

# =========================
# 思考の系譜グラフAPI (UI連携用)
# =========================
@app.get("/api/lineage/{memo_id}")
async def api_lineage(memo_id: str, hops: int = 3):
    """グラフUI用に思考の系譜データ(nodes/edges)を返す"""
    data = get_multi_hop_lineage(start_memo_id=memo_id, max_hops=hops)
    return JSONResponse(content=data)
# =========================
# 起動
# =========================

if __name__ == "__main__":

    

    uvicorn.run(app,host="127.0.0.1",port=8000,)
# EXPERIMENT_PARITY_ROUTER_V1
from src.experiment_api import router as experiment_router
app.include_router(experiment_router)


# THESIS_PROPOSAL_B_ROUTER_V1
from src.proposal_b_routes import router as proposal_b_router
app.include_router(proposal_b_router)


# AUTO_GEMMA_DIRECT_APP_ROUTE_V1
from src.auto_relation_suggestion import router as auto_gemma_router
app.include_router(auto_gemma_router)
