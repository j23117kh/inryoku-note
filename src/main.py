import datetime
import uuid
import numpy as np
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

from src.embedding import get_vector

from src.memo_analyzer import (
    analyze_memo,
    gen_json,
)

from src.recommender import search_core

from src.similarity import parse_date


# =========================
# FastAPI
# =========================

app = FastAPI()


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

    auto_apply = bool(
        data.get(
            "auto_apply",
            False
        )
    )

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

        "supersedes": (
            [
                item["id"]
                for item in suggestions
            ]
            if auto_apply
            else []
        ),

        "suggested": (
            []
            if auto_apply
            else suggestions
        ),

        "status": "active",

        "title": title,

        "content": content,

        "vector": vector.tolist(),
    }

    memos.insert(
        0,
        new_memo
    )

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

        "applied":
            auto_apply,

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

    accept = set(
        data.get(
            "accept",
            []
        )
    )

    memos = load_memos()

    for memo in memos:

        if memo["id"] != memo_id:
            continue

        allowed = {
            item["id"]
            for item
            in memo.get(
                "suggested",
                []
            )
        }

        memo["supersedes"] = sorted(
            set(
                memo["supersedes"]
            )
            |
            (
                accept
                & allowed
            )
        )

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

    memo_id = data.get(
        "id"
    )

    old_id = data.get(
        "old_id"
    )

    memos = load_memos()

    for memo in memos:

        if (
            memo["id"] == memo_id
            and old_id
            in memo["supersedes"]
        ):

            memo[
                "supersedes"
            ].remove(
                old_id
            )

            save_memos(memos)

            return {
                "status":
                    "success"
            }

    return {
        "status":
            "error",

        "message":
            "該当の関係がありません",
    }


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
            "full"
        ),

        float(
            data.get(
                "threshold",
                30
            )
        ) / 100.0,

        int(
            data.get(
                "top_k",
                10
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

    hits = search_core(

        question,

        "full",

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

    evidence_text = "\n".join(

        (
            f'- id={hit["id"]} / '
            f'作成={hit["created_at"][:10]} / '
            f'出来事日={hit["event_date"]} / '
            f'{"【古い情報(更新済み)】" if hit["stale"] else "【有効】"} / '
            f'内容={hit["content"][:300]}'
        )

        for hit in hits
    )

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

    try:

        output = gen_json(
            prompt
        )

    except Exception as e:

        print(
            f"回答生成エラー: {e}"
        )

        return JSONResponse(
            {
                "error":
                    "回答の生成に失敗"
            },
            status_code=500,
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

    mode = data.get(
        "mode",
        "hybrid"
    )

    mode = {
        "ai": "semantic",
        "hybrid": "full",
    }.get(
        mode,
        mode
    )

    result = search_core(

        data.get(
            "text",
            ""
        ).strip(),

        mode,

        float(
            data.get(
                "threshold",
                30
            )
        ) / 100.0,
    )

    if result is None:

        return JSONResponse(
            {
                "error":
                    "ベクトル化に失敗"
            },
            status_code=500,
        )
    # ⬇️⬇️⬇️ ここから追加：フロントエンドの多階層連鎖のためにデータを結合 ⬇️⬇️⬇️
    all_memos = load_memos()
    memo_db = {m["id"]: m for m in all_memos}

    for item in result:
        original_memo = memo_db.get(item["id"], {})
        item["supersedes"] = original_memo.get("supersedes", [])
        item["related_ids"] = original_memo.get("related_ids", [])
    # ⬆️⬆️⬆️ ここまで追加 ⬆️⬆️⬆️
    return result


# =========================
# 起動
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )