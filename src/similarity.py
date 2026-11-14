from datetime import datetime, date
import numpy as np


def cosine_similarity(vector_a, vector_b):
    """2つのベクトルのコサイン類似度を計算する。"""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"ベクトル形状が一致しません: {a.shape} != {b.shape}")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def parse_date(value):
    """日付文字列または日付オブジェクトを datetime に変換する。"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    value = str(value).strip()
    if not value:
        return None

    # ISO 8601 (タイムゾーン付きも含む) を先に処理
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # naive/aware の比較エラーを避け、日付ベースの評価に統一
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d", "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def time_align(memo, time_from=None, time_to=None):
    """
    event_date と期間の近さを0～1で返す。
    event_date がない/不正なら None。指定期間内は1、期間外は日単位で減衰。
    """
    start = parse_date(time_from)
    end = parse_date(time_to)

    if time_from not in (None, "") and start is None:
        raise ValueError(f"不正な time_from: {time_from!r}")
    if time_to not in (None, "") and end is None:
        raise ValueError(f"不正な time_to: {time_to!r}")
    if start is not None and end is not None and start > end:
        raise ValueError("time_from は time_to 以前である必要があります")

    event_value = memo.get("event_date")
    event_date = parse_date(event_value)
    if event_value not in (None, "") and event_date is None:
        return None
    if event_date is None:
        return None
    if start is None and end is None:
        return 1.0

    if start is not None and event_date < start:
        distance = max(0, (start.date() - event_date.date()).days)
        return 1.0 / (1.0 + distance)
    if end is not None and event_date > end:
        distance = max(0, (event_date.date() - end.date()).days)
        return 1.0 / (1.0 + distance)
    return 1.0
