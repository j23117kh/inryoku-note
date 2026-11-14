"""Conservative, deterministic Japanese question-intent/date extraction.

No answer labels, question metadata or oracle relationships are accepted. This
is a rule-based baseline, NOT a trained or validated general-purpose model.
"""
from __future__ import annotations

from datetime import date
import re

_DATE_FULL = re.compile(r"(?<!\d)(\d{4})\s*(?:年|[/-])\s*(\d{1,2})\s*(?:月|[/-])\s*(\d{1,2})\s*日?(?!\d)")
_DATE_PARTIAL = re.compile(r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日")
_CHANGE = re.compile(
    r"変遷|経緯|変更履歴|更新履歴|修正履歴|変更経緯|修正経緯|思考の流れ|思考の変化|"
    r"初期案から|最初の案から|未着手から完了|分岐して|やり直した理由|"
    r"(?:どう|どのように|どんなふうに).{0,24}(?:分かれ|枝分かれ|分岐し)|"
    r"(?:分かれた|枝分かれした).{0,20}(?:経緯|理由|過程|流れ)|"
    r"どう(?:いう|のように)?変わ|どう変化|どのように変化|"
    r"(?:変更|更新|修正|訂正|分岐|変化).{0,18}(?:理由|経緯|履歴|過程|流れ|なぜ|どうして)|"
    r"(?:なぜ|どうして|どのように).{0,20}(?:変更|修正|訂正|更新|分岐|変化)|"
    r"(?:当初|以前|最初).{0,32}(?:今|現在|最新).{0,16}(?:違い|比較|変わ)|"
    r"(?:何回|何度).{0,12}(?:更新|修正|変更)|"
    r"(?:修正|更新|変更)を(?:重ね|繰り返し)|"
    r"(?:見直し|取り消し|保留|再開|延長|統合).{0,20}(?:理由|結果|経過)|"
    r"(?:から|以降|以後).{0,36}(?:までの流れ|に至る|まで追い|まで何が)|"
    r"(?:変わった|変えた|追加した|修正した).{0,12}(?:理由|根拠|後)|"
    r"(?:進捗|変化).{0,12}(?:流れ|経過|たど|追い)|"
    r"(?:なぜ|どうして).{0,24}(?:保留|見送|中止|変わ)|"
    r"(?:比較して|比較した).{0,30}(?:決め|見送|保留|組み合わせ)|"
    r"(?:中断|検討|調査|判断|依頼|予約).{0,24}(?:してから|した後|から).{0,30}(?:再開|見送|取消|完了|何が|どう)|"
    r"(?:中断|検討|調査|判断|依頼|予約)してから|"
    r"(?:取り消す|中止する|見送る).{0,15}(?:まで|理由|流れ)|"
    r"(?:どう|どのように).{0,12}(?:直した|更新した|変えた)|"
    r"(?:認識|予定|方針|見積もり).{0,18}(?:なぜ変わ|変わり)|"
    r"(?:ことにした|することにした).{0,12}理由|"
    r"(?:最終|結論).{0,12}(?:に至る|になった)"
)
_PAST = re.compile(
    r"以前|当時|過去|昔|元々|もともと|旧版|変更前|修正前|訂正前|"
    r"前回|当初|最初の|初期の|時点|そのころ|その頃|かつて|"
    r"最初|(?:初め|はじめ)(?:は|、|に|何|どんな)|予定だった|考えていた|"
    r"(?:だった|していた|ていた|された|までに)|"
    r"速報値|時の記録|"
    r"までの(?:記録|メモ|判断)|当初の予定"
)


def classify_query(text: str, *, as_of: str | None = None) -> dict:
    """Classify question text using ONLY its text and optional explicit date.

    Date-only requests without a four-digit year are deliberately not guessed.
    Absolute dates used as part of a change-history question are *not* taken
    as an as-of cutoff unless the user explicitly supplies an as_of override.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("質問文を入力してください")
    if len(text) > 3000:
        raise ValueError("質問文は3000文字以内にしてください")
    text = text.strip()
    is_change = bool(_CHANGE.search(text))
    is_past = bool(_PAST.search(text))
    match = _DATE_FULL.search(text)
    mentioned_date = None
    if match:
        try:
            mentioned_date = date(*(int(x) for x in match.groups())).isoformat()
        except ValueError as exc:
            raise ValueError("質問文の日付が不正です") from exc
    if as_of not in (None, ""):
        if not isinstance(as_of, str):
            raise ValueError("as_of はYYYY-MM-DD形式で指定してください")
        try:
            cutoff = date.fromisoformat(as_of.strip()).isoformat()
        except ValueError as exc:
            raise ValueError("as_of はYYYY-MM-DD形式で指定してください") from exc
        # An explicitly supplied date is a snapshot cutoff, regardless of text.
        date_source = "explicit_field"
    elif is_change:
        cutoff = None  # "2026年4月からどう変わった" is not a cutoff.
        date_source = "not_applied_to_change_question"
    elif mentioned_date:
        cutoff = mentioned_date
        date_source = "question_text"
    else:
        cutoff = None
        date_source = "none"

    if is_change:
        intent = "change"
    elif is_past or cutoff is not None:
        intent = "past"
    else:
        intent = "current"
    ambiguous_date = bool(_DATE_PARTIAL.search(text) and not match and cutoff is None)
    return {
        "intent": intent,
        "as_of": cutoff,
        "mentioned_date": mentioned_date,
        "date_source": date_source,
        "needs_date_clarification": intent == "past" and cutoff is None or ambiguous_date,
        "classifier": "rule_based_v1",
        "automatic": True,
        "validated_accuracy": None,
    }
