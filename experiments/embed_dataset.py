"""
評価用データセットをベクトル化する（Gemini API を使用・完全整合性対応版）。

  memos_eval.json    -> memos_eval_vec_d<次元>.json     （メモ: タイトル付きの RETRIEVAL_DOCUMENT）
  queries_eval.json -> queries_eval_vec_d<次元>.json    （クエリ: クエリ文から新規に計算 = RETRIEVAL_QUERY）

次元は --dim で指定する（min=128 / recommended=768 / max=3072 / 任意の整数）。
モデルは --model、入力形式は --style（legacy / documented）で指定する。
既定（gemini-embedding-2 + legacy）以外は、ファイル名に条件が付く。

重要: クエリのベクトルは「クエリ文そのもの」から計算する。
正解メモのベクトルは絶対に流用しない。完了後に check_no_leak() で確認する。

途中で止まっても、再実行すれば計算済みの分はスキップして続きから進む。

使い方:  python -m experiments.embed_dataset --dim recommended
         python -m experiments.embed_dataset --fake   # APIなしの動作確認用
"""

import argparse
import time
from pathlib import Path

from src import config
from src.config import resolve_embed_dim, resolve_embed_style
from src.embedding import build_request
from experiments.common import (
    check_no_leak,
    eval_paths,
    read_json,
    variant_tag,
    write_json,
)


def _embed_with_retry(fn, retries=5, wait=2.0):
    """指数バックオフ（Exponential Backoff）を伴う安全な埋め込み取得"""
    for attempt in range(retries):
        vec = fn()
        if vec is not None:
            return vec
        # 指数バックオフによる待機（2のべき乗）
        sleep_time = wait * (2 ** attempt)
        time.sleep(sleep_time)

    raise RuntimeError("埋め込みに失敗しました（リトライ上限）。APIキーと通信を確認してください。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true",
                        help="ハッシュ埋め込みで代用（動作確認専用）")
    parser.add_argument("--sleep", type=float, default=0.1,
                        help="API呼び出し間隔（秒）")
    parser.add_argument("--dim", default=None,
                        help="出力次元: min / recommended / max / 128〜3072の整数"
                             "（省略時は .env の EMBED_DIM）")
    parser.add_argument("--model", default=None,
                        help="埋め込みモデル名。省略時は .env の EMBED_MODEL")
    parser.add_argument("--style", default=None,
                        help="入力形式: legacy / documented。省略時は .env の EMBED_STYLE")
    args = parser.parse_args()

    dim = resolve_embed_dim(config.EMBED_DIM if args.dim is None else args.dim)
    model = config.EMBED_MODEL if args.model is None else args.model.strip()
    style = resolve_embed_style(config.EMBED_STYLE if args.style is None else style)

    print(f"埋め込み条件: 次元={dim} / モデル={model} / 形式={style}")

    paths = eval_paths(fake=args.fake, dim=dim, model=model, style=style)

    if args.fake:
        from experiments.fake_embed import fake_vector

        def embed_doc(m):
            text, _ = build_request(m["content"], "RETRIEVAL_DOCUMENT", m["title"], model, style)
            return fake_vector(text, dim)

        def embed_query(q):
            text, _ = build_request(q["text"], "RETRIEVAL_QUERY", None, model, style)
            return fake_vector(text, dim)
    else:
        from src.embedding import get_vector

        def embed_doc(m):
            return _embed_with_retry(
                lambda: get_vector(m["content"], "RETRIEVAL_DOCUMENT", m["title"],
                           dim=dim, model=model, style=style)
            )

        def embed_query(q):
            return _embed_with_retry(
                lambda: get_vector(q["text"], "RETRIEVAL_QUERY",
                           dim=dim, model=model, style=style)
            )

    memos = read_json(paths["memos_raw"])
    queries = read_json(paths["queries_raw"])

    # 再開用キャッシュ（本文・タイトルが変わっていたら再計算）
    cache_m = {}
    if paths["memos_vec"].exists():
        cache_m = {
            m["id"]: m for m in read_json(paths["memos_vec"])
            if "vector" in m
        }

    cache_q = {}
    if paths["queries_vec"].exists():
        cache_q = {
            q["qid"]: q for q in read_json(paths["queries_vec"])
            if "query_vector" in q
        }

    for i, m in enumerate(memos, 1):
        old = cache_m.get(m["id"])

        if old and old["content"] == m["content"] and old["title"] == m["title"]:
            m["vector"] = old["vector"]
            continue

        vec = embed_doc(m)
        m["vector"] = [float(x) for x in vec]

        if not args.fake:
            time.sleep(args.sleep)

        if i % 50 == 0:
            print(f"メモ {i}/{len(memos)}")
            write_json(paths["memos_vec"], memos)

    for i, q in enumerate(queries, 1):
        old = cache_q.get(q["qid"])

        if old and old["text"] == q["text"]:
            q["query_vector"] = old["query_vector"]
            continue

        vec = embed_query(q)
        q["query_vector"] = [float(x) for x in vec]

        if not args.fake:
            time.sleep(args.sleep)

        # クエリ側でも定期的に途中保存を行うよう修正
        if i % 50 == 0:
            print(f"クエリ {i}/{len(queries)}")
            write_json(paths["queries_vec"], queries)

    check_no_leak(memos, queries)

    write_json(paths["memos_vec"], memos)
    write_json(paths["queries_vec"], queries)

    print(f"完了: メモ{len(memos)}件 / クエリ{len(queries)}件（リーク検査OK）")
    print(f"保存先: {paths['memos_vec'].parent}")


if __name__ == "__main__":
    main()