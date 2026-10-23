"""評価パイプラインの共通処理。"""

import json
import random
import re

import numpy as np

from src import config
from src.config import (
    DEFAULT_EMBED_MODEL,
    EVAL_DIR,
    LEGACY_DATA_FILE,
    resolve_embed_dim,
    resolve_embed_style,
)
from src.data_manager import normalize_memo


def variant_tag(model=None, style=None):
    """
    埋め込みのバリアント（モデル×スタイル）を表すファイル名の接尾辞。
    既定（gemini-embedding-2 + legacy）は空文字: 既存のファイル名と互換。
    """

    model = (config.EMBED_MODEL if model is None else str(model).strip())
    style = resolve_embed_style(config.EMBED_STYLE if style is None else style)

    if model == DEFAULT_EMBED_MODEL and style == "legacy":
        return ""

    slug = re.sub(r"[^A-Za-z0-9.\-]+", "-", model)

    return f"_{slug}_{style}"


def eval_paths(fake=False, dim=None, model=None, style=None):
    """
    評価データのパス。ベクトル付きファイルは、次元・モデル・スタイルごとに分ける
    （別の条件のベクトルが混ざって比較が壊れるのを防ぐため）。
    """

    dim = resolve_embed_dim(config.EMBED_DIM if dim is None else dim)

    suffix = f"_d{dim}" + variant_tag(model, style) + ("_fake" if fake else "")

    return {
        "memos_raw": EVAL_DIR / "memos_eval.json",
        "queries_raw": EVAL_DIR / "queries_eval.json",
        "memos_vec": EVAL_DIR / f"memos_eval_vec{suffix}.json",
        "queries_vec": EVAL_DIR / f"queries_eval_vec{suffix}.json",
    }


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False,indent=2)


def check_no_leak(memos, queries, tol=1e-6):
    """
    データリークの検査。

    クエリのベクトルが、どれかのメモのベクトルと（ほぼ）同一なら例外を出す。
    旧評価では「クエリのベクトル = 正解メモ自身のベクトル」だったため、
    その再発を防ぐ。
    """

    ids = [m["id"] for m in memos if "vector" in m]
    mat = np.array([m["vector"] for m in memos if "vector" in m])

    if mat.size == 0:
        return

    for q in queries:

        qv = np.asarray(q["query_vector"], dtype=float)

        dist = np.linalg.norm(mat - qv, axis=1)

        j = int(dist.argmin())

        if dist[j] < tol:
            raise ValueError(
                f"データリーク: クエリ {q['qid']} のベクトルが "
                f"メモ {ids[j]} のベクトルと同一です。"
            )


def load_distractors(n, seed, dim=None):
    """
    既存のWikipediaメモ（ベクトル付き）から n 件を背景ノイズとして取り出す。
    created_at は十分古い日付にして、as_of 検索でも常に対象に含める。
    """

    if n <= 0 or not LEGACY_DATA_FILE.exists():
        return []

    dim = resolve_embed_dim(config.EMBED_DIM if dim is None else dim)

    allm = [m for m in read_json(LEGACY_DATA_FILE) if "vector" in m]

    # 次元が違うベクトルは混ぜられない（内積が計算できない）
    pool = [m for m in allm if len(m["vector"]) == dim]

    if not pool:
        have = sorted({len(m["vector"]) for m in allm})
        print(
            f"警告: 背景ノイズ(Wikipedia)のベクトルは {have} 次元で、"
            f"評価の次元 {dim} と一致しないため使いません。"
            "使うには tools.reembed_memos で再埋め込みしてください。"
        )
        return []

    rng = random.Random(seed)
    rng.shuffle(pool)

    out = []

    for i, m in enumerate(pool[:n]):
        out.append(normalize_memo({
            "id": f"wiki-{i:04d}",
            "title": m.get("title", ""),
            "content": m.get("content", ""),
            "date": "2000/01/01",
            "created_at": "2000-01-01T00:00:00",
            "vector": m["vector"],
            "kind": "distractor",
        }))

    return out
