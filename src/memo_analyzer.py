import datetime
import json

from src.config import (
    GEMINI_API_KEY,
    GEN_MODEL,
)

_client = None


def _get_client():
    """Geminiクライアントの遅延初期化（import時にAPIキーを要求しない）。"""

    global _client

    if _client is None:

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY が設定されていません。"
            )

        from google import genai

        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def gen_json(prompt):
    """
    GeminiからJSON形式の回答を取得する。
    """

    response = _get_client().models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    return json.loads(response.text)


def analyze_memo(
    content,
    today,
    candidates,
):
    """
    新しいメモを解析し、

    - event_date
    - supersedes

    を取得する。

    戻り値:
        (
            event_date,
            [
                {
                    "id": "...",
                    "reason": "..."
                }
            ]
        )
    """

    candidate_text = "\n".join(
        (
            f'- id={memo["id"]} / '
            f'作成={memo["created_at"][:10]} / '
            f'出来事日={memo.get("event_date")} / '
            f'内容={memo["content"][:200]}'
        )
        for memo in candidates
    )

    if not candidate_text:
        candidate_text = "(なし)"

    prompt = f"""
あなたは個人メモの整理係です。
新しいメモを読み、JSONだけを返してください。

今日の日付:
{today}

新しいメモ:
{content}

既存メモの候補:
{candidate_text}

出力形式:

{{
  "event_date": "YYYY-MM-DD" または null,
  "supersedes": [
    {{
      "id": "更新・訂正・取り消された既存メモのid",
      "reason": "20字程度の理由"
    }}
  ]
}}

ルール:

- event_dateは、
  メモが指す出来事の日付。
- 「昨日」などの相対表現は
  今日の日付から解決する。
- 不明ならnull。
- supersedesには、
  同じ事柄について新しいメモが
  古い情報を上書きしている場合のみ入れる。
- 単に関連するだけなら入れない。
- 候補にないidは絶対に入れない。
"""

    try:

        result = gen_json(prompt)

        valid_ids = {
            memo["id"]
            for memo in candidates
        }

        supersedes = []

        for item in result.get(
            "supersedes",
            []
        ):

            if isinstance(item, dict):

                memo_id = item.get("id")
                reason = item.get(
                    "reason",
                    ""
                )

            else:

                memo_id = item
                reason = ""

            if memo_id in valid_ids:

                supersedes.append(
                    {
                        "id": memo_id,
                        "reason": reason,
                    }
                )

        # (中略) supersedes の処理が終わった後...
        
        event_date = result.get("event_date")
        if event_date:
            try:
                # バリデーション（フォーマットチェック）
                datetime.date.fromisoformat(event_date)
            except (ValueError, TypeError):
                # フォーマットが不正な場合は日付のみ無効化（supersedesは活かす）
                print(f"警告: 不正な event_date フォーマット: {event_date}")
                event_date = None

        return (event_date, supersedes)

    except Exception as e:
        print(f"メモ解析エラー(全体): {e}")
        return (None, [])

    