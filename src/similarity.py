from datetime import datetime, date
import numpy as np

def cosine_similarity(vector_a, vector_b):
    """2つのベクトルのコサイン類似度を計算する。"""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))

def parse_date(value):
    """日付文字列または日付オブジェクトを datetime に変換する。"""
    if not value:
        return None
        
    if isinstance(value, datetime):
        return value
        
    # 追加: datetime.date型が渡された場合、文字列変換を介さず直接変換する
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    value = str(value).strip()
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None

def time_align(memo, time_from=None, time_to=None):
    """
    メモの event_date と指定された期間の時間的な近さを0～1で評価する。
    """
    event_date = memo.get("event_date")
    if not event_date:
        return None

    event_date = parse_date(event_date)
    if event_date is None:
        return None

    start = parse_date(time_from)
    end = parse_date(time_to)

    if start is None and end is None:
        return 1.0

    # 修正: 片側指定時の論理矛盾を解消
    if start is not None and end is None:
        if event_date >= start:
            return 1.0
        distance = (start - event_date).days
        return 1.0 / (1.0 + distance)

    # 修正: 片側指定時の論理矛盾を解消
    if start is None and end is not None:
        if event_date <= end:
            return 1.0
        distance = (event_date - end).days
        return 1.0 / (1.0 + distance)

    if start <= event_date <= end:
        return 1.0

    if event_date < start:
        distance = (start - event_date).days
    else:
        distance = (event_date - end).days

    return 1.0 / (1.0 + distance)