import datetime
import json
import os
import shutil

from src.config import DATA_FILE, LEGACY_DATA_FILE


def init_data_file():
    """
    初回起動時に旧データを新しいデータファイルへコピーする。
    元ファイルは変更しない。
    """

    if not DATA_FILE.exists() and LEGACY_DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(LEGACY_DATA_FILE, DATA_FILE)


def normalize_memo(memo):
    """
    旧データとの互換処理。足りないキーを補う（memoを直接書き換えて返す）。
    """
    raw_date = memo.get("date", "")
    created_at_val = "1970-01-01"

    if raw_date:
        try:
            # "2023/1/5" などのスラッシュ区切りを "2023-01-05" (ゼロ埋めISO形式) に変換
            dt = datetime.datetime.strptime(raw_date, "%Y/%m/%d")
            created_at_val = dt.strftime("%Y-%m-%d")
        except ValueError:
            # 既に "2023-01-05" のようにハイフン区切りになっている、
            # あるいは想定外のフォーマットの場合は単純置換でフォールバック
            created_at_val = raw_date.replace("/", "-")

    memo.setdefault(
        "created_at",
        created_at_val,
    )

    memo.setdefault("event_date", None)

    memo.setdefault("supersedes", [])

    memo.setdefault("suggested", [])

    # 旧仕様のsupersededをactiveへ戻す
    if memo.get("status") == "superseded":
        memo["status"] = "active"

    memo.setdefault("status", "active")

    return memo


def load_memos():
    """
    メモデータをJSONから読み込む。
    """

    if not DATA_FILE.exists():
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            memos = json.load(f)

    except json.JSONDecodeError:
        backup = (
            str(DATA_FILE)
            + ".corrupt."
            + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )

        shutil.copy(DATA_FILE, backup)

        raise RuntimeError(
            f"{DATA_FILE} が壊れています。"
            f"{backup} に退避しました。"
        )

    for memo in memos:
        normalize_memo(memo)

    return memos


def save_memos(memos):
    """
    メモを安全に保存する。
    """

    tmp_file = str(DATA_FILE) + ".tmp"

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(
            memos,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 一時ファイルから正式ファイルへ原子的に置換
    os.replace(tmp_file, DATA_FILE)


def live_memos(memos, as_of=None):
    """
    削除されていないメモだけを取得する。

    as_ofが指定された場合は、
    その日までに作成されたメモだけを対象にする。
    """

    result = [
        memo
        for memo in memos
        if memo.get("status") != "deleted"
    ]

    if as_of:
        result = [
            memo
            for memo in result
            if (
                parse_created_date(memo.get("created_at"))
                or datetime.date.max
            ) <= as_of
        ]

    return result


def parse_created_date(value):
    """
    created_atから日付だけ取り出す。
    """

    try:
        return datetime.date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None


def superseded_map(memos):
    """
    {古いメモID: 最終的な最新メモID} を作る。
    更新の連鎖 (A -> B -> C) を解決し、循環 (A -> B -> A) を防ぐ。
    """
    # 1. まず直接の更新関係を作る {A: B, B: C}
    direct_map = {}
    sorted_memos = sorted(memos, key=lambda x: x.get("created_at", ""))
    for memo in sorted_memos:
        for old_id in memo.get("supersedes", []):
            direct_map[old_id] = memo["id"]

    # 2. 連鎖をたどって最終的な最新IDを導出する
    latest = {}
    for old_id in direct_map:
        curr = old_id
        visited = set()
        
        # 循環を検知しつつ最後までたどる
        while curr in direct_map and curr not in visited:
            visited.add(curr)
            curr = direct_map[curr]
            
        # 無限ループでなければ、最終到達地点をセット
        if curr not in visited:
            latest[old_id] = curr
            
    return latest

def update_memo(memo_id, new_content=None, new_title=None, new_event_date=None):
    """
    既存のメモを指定したIDで探し、内容を部分的に更新して保存する。
    """
    memos = load_memos()
    updated = False

    for memo in memos:
        if memo["id"] == memo_id:
            if new_content is not None:
                memo["content"] = new_content
            if new_title is not None:
                memo["title"] = new_title
            if new_event_date is not None:
                memo["event_date"] = new_event_date
            
            updated = True
            break

    if updated:
        save_memos(memos)
        return True
    
    return False