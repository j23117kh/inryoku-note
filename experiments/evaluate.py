"""
卒研用：thought evaluation dataset × src.recommender.search_core 総合評価スクリプト
（時間的リーク防止・リトライ機構・シード固定対応版）

実行:
    プロジェクトルートで
    python experiments/evaluate.py

入力:
    data/eval/memos_eval.json
    data/eval/queries_eval.json

出力:
    results/thought_eval_results.csv        (全クエリ×手法ごとの詳細スコア)
    results/thought_eval_summary.csv        (手法ごとの平均スコア一覧)
    results/thought_eval_details.json       (検索結果オブジェクトの全詳細)
    results/thought_eval_vectors.json       (計算済みベクトルのキャッシュ)
"""

import csv
import json
import sys
import random
import time
from datetime import date
from pathlib import Path
from collections import defaultdict
import numpy as np


# ============================================================
# 乱数シードの固定（再現性の担保）
# ============================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# プロジェクトルートを Python の import path に追加
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# パス設定
# ============================================================

EVAL_DIR = ROOT / "data" / "eval"
RESULT_DIR = ROOT / "results"

MEMO_FILE = EVAL_DIR / "memos_eval.json"
QUERY_FILE = EVAL_DIR / "queries_eval.json"

RESULT_FILE = RESULT_DIR / "thought_eval_results.csv"
SUMMARY_FILE = RESULT_DIR / "thought_eval_summary.csv"
DETAIL_FILE = RESULT_DIR / "thought_eval_details.json"
VECTOR_FILE = RESULT_DIR / "thought_eval_vectors.json"


# ============================================================
# 評価設定
# ============================================================

TOP_K = 10
THRESHOLD = 0.0

MODES = (
    "keyword",
    "semantic",
    "graph",
    "full_no_penalty",
    "full",
    "thought_evolution",
)


# ============================================================
# JSON 入出力
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Query 関連ユーティリティ
# ============================================================

def get_query_text(query):
    if "text" in query:
        return query["text"]
    if "query" in query:
        return query["query"]
    raise KeyError(f"質問文のキーが見つかりません。使用可能なキー: {list(query.keys())}")


def get_query_id(query):
    if "qid" in query:
        return query["qid"]
    if "id" in query:
        return query["id"]
    raise KeyError(f"質問IDのキーが見つかりません。使用可能なキー: {list(query.keys())}")


def get_relevant_ids(query):
    if query.get("relevant_ids"):
        return list(query["relevant_ids"])
    if query.get("relevant_id"):
        return [query["relevant_id"]]
    return []


def get_stale_ids(query):
    """古い情報の混入率計算用の古いメモIDリストを取得"""
    if query.get("stale_ids"):
        return list(query["stale_ids"])
    return []


# ============================================================
# Memo 関連ユーティリティ
# ============================================================

def get_memo_content(memo):
    if "content" in memo:
        return memo["content"]
    if "text" in memo:
        return memo["text"]
    raise KeyError(f"メモ本文のキーが見つかりません。使用可能なキー: {list(memo.keys())}")


def get_memo_title(memo):
    if memo.get("title"):
        return memo["title"]
    return memo.get("theme", "")


def get_memo_date(memo):
    return memo.get("date", "")


def get_created_at(memo):
    return memo.get("created_at", memo.get("date", ""))


def get_event_date(memo):
    return memo.get("event_date")


# ============================================================
# 日付ユーティリティ
# ============================================================

def parse_date(date_text):
    if not date_text:
        return None
    return str(date_text)[:10]


def parse_date_object(date_text):
    if not date_text:
        return None
    try:
        return date.fromisoformat(str(date_text)[:10])
    except ValueError as e:
        raise ValueError(f"日付を解釈できません: {date_text!r}") from e


def make_time_args(query):
    time_from = query.get("time_from")
    time_to = query.get("time_to")

    if time_from is not None or time_to is not None:
        return parse_date(time_from), parse_date(time_to)

    as_of = query.get("as_of")
    if as_of:
        day = parse_date(as_of)
        return day, day

    return None, None


# ============================================================
# ベクトル生成・キャッシュ（リトライ機構つき）
# ============================================================

def load_or_create_vectors(raw_memos, queries):
    if VECTOR_FILE.exists():
        print(f"既存ベクトルを読み込みます: {VECTOR_FILE}")
        data = load_json(VECTOR_FILE)
        memo_vectors = data.get("memos", {})
        query_vectors = data.get("queries", {})

        missing_memos = [m["id"] for m in raw_memos if m["id"] not in memo_vectors]
        missing_queries = [get_query_id(q) for q in queries if get_query_id(q) not in query_vectors]

        if not missing_memos and not missing_queries:
            return memo_vectors, query_vectors

        print(f"不足ベクトルを追加生成します: memos={len(missing_memos)}, queries={len(missing_queries)}")
    else:
        print("評価用ベクトルを新規作成します。")
        memo_vectors = {}
        query_vectors = {}

    from src.embedding import get_vector

    def get_vector_safe(text, task_type, title=None, max_retries=5):
        """レート制限(429)・一時エラー対策付きの安全なベクトル取得"""
        for attempt in range(max_retries):
            try:
                vec = get_vector(text, task_type=task_type, title=title)
                if vec is not None:
                    time.sleep(0.7)  # API負荷軽減・レート制限回避のためのウェイト
                    return vec
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = (2 ** attempt) + 3.0
                    print(f"  [レート制限 (429)] {wait_time:.1f}秒待機して再試行します... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
            time.sleep(1.0)
        return None

    for index, memo in enumerate(raw_memos, start=1):
        memo_id = memo["id"]
        if memo_id in memo_vectors:
            continue
        text = get_memo_content(memo)
        print(f"[memo {index}/{len(raw_memos)}] vectorizing {memo_id}")
        vector = get_vector_safe(text, task_type="RETRIEVAL_DOCUMENT", title=get_memo_title(memo))
        if vector is None:
            raise RuntimeError(f"メモ {memo_id} のベクトル取得に失敗しました。")
        memo_vectors[memo_id] = [float(x) for x in vector]

        if index % 10 == 0 or index == len(raw_memos):
            save_json(VECTOR_FILE, {"memos": memo_vectors, "queries": query_vectors})

    for index, query in enumerate(queries, start=1):
        qid = get_query_id(query)
        if qid in query_vectors:
            continue
        text = get_query_text(query)
        print(f"[query {index}/{len(queries)}] vectorizing {qid}")
        vector = get_vector_safe(text, task_type="RETRIEVAL_QUERY")
        if vector is None:
            raise RuntimeError(f"質問 {qid} のベクトル取得に失敗しました。")
        query_vectors[qid] = [float(x) for x in vector]
        if index % 10 == 0 or index == len(queries):
            save_json(VECTOR_FILE, {"memos": memo_vectors, "queries": query_vectors})

    print(f"評価用ベクトルを保存しました: {VECTOR_FILE}")
    return memo_vectors, query_vectors


# ============================================================
# 評価指標（総合版）
# ============================================================

def precision_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for memo_id in top if memo_id in relevant) / k


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, memo_id in enumerate(ranked_ids, start=1):
        if memo_id in relevant:
            return 1.0 / rank
    return 0.0


def stale_rate_at_k(ranked_ids, stale_ids, k=5):
    stale = set(stale_ids or [])
    if not stale or not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    stale_count = sum(1 for memo_id in top_k if memo_id in stale)
    return stale_count / len(top_k)


def dcg(values):
    total = 0.0
    for rank, value in enumerate(values, start=1):
        total += value / np.log2(rank + 1)
    return float(total)


def ndcg_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    actual = [1 if memo_id in relevant else 0 for memo_id in ranked_ids[:k]]
    ideal_count = min(len(relevant), k)
    if ideal_count == 0:
        return 0.0
    ideal = [1 for _ in range(ideal_count)]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(actual) / ideal_dcg


def turning_at_k(ranked_ids, turning_ids, k=5):
    if not turning_ids:
        return None
    top_k = set(ranked_ids[:k])
    found = sum(1 for tid in turning_ids if tid in top_k)
    return found / len(turning_ids)


def root_at_k(ranked_ids, root_id, k=5):
    if not root_id:
        return None
    top_k = set(ranked_ids[:k])
    return 1.0 if root_id in top_k else 0.0


# ============================================================
# search_core 呼び出し（時間的リーク防止・動的グラフ構築対応）
# ============================================================

def prepare_memos(raw_memos, vectors):
    prepared = []
    for memo in raw_memos:
        memo_id = memo["id"]
        prepared.append({
            "id": memo_id,
            "date": get_memo_date(memo),
            "created_at": get_created_at(memo),
            "title": get_memo_title(memo),
            "content": get_memo_content(memo),
            "event_date": get_event_date(memo),
            "supersedes": list(memo.get("supersedes", []) or []),
            "chain_id": memo.get("chain_id"),
            "version": memo.get("version"),
            "status": memo.get("status", "active"),
            "vector": vectors[memo_id],
        })
    return prepared


def get_temporal_filtered_graph_and_memos(all_prepared_memos, query, graph_cache, build_graph_func):
    """
    【時間的リーク防止対応】
    クエリの as_of（指定時点）または time_to に基づき、その時点で存在していた
    メモのみを抽出し、そのサブセットに対してグラフを構築・キャッシュする。
    """
    as_of_str = query.get("as_of")
    if not as_of_str:
        # as_of が指定されていない場合は全メモ・全グラフを返す
        cache_key = "all"
        if cache_key in graph_cache:
            return graph_cache[cache_key], all_prepared_memos
        
        vectors = np.array([m["vector"] for m in all_prepared_memos], dtype=float)
        g = build_graph_func(all_prepared_memos, vectors)
        graph_cache[cache_key] = g
        return g, all_prepared_memos

    as_of_date = parse_date_object(as_of_str)
    
    # as_of 時点以前に作成されたメモのみにフィルタリング
    filtered_memos = []
    for m in all_prepared_memos:
        created_str = m.get("created_at") or m.get("date")
        if not created_str:
            filtered_memos.append(m)  # 日付不明なものは安全のため含める
            continue
        try:
            m_date = parse_date_object(created_str)
            if m_date <= as_of_date:
                filtered_memos.append(m)
        except Exception:
            filtered_memos.append(m)

    cache_key = f"as_of_{as_of_str}"
    if cache_key in graph_cache:
        return graph_cache[cache_key], filtered_memos

    # フィルタリングされたメモ群のみでグラフを再構築（未来情報の混入を完全に遮断）
    vectors = np.array([m["vector"] for m in filtered_memos], dtype=float) if filtered_memos else np.zeros((0, 1536))
    g = build_graph_func(filtered_memos, vectors)
    graph_cache[cache_key] = g
    return g, filtered_memos


def call_search_core(
    search_core,
    query,
    mode,
    all_prepared_memos,
    query_vector,
    graph_cache,
    build_graph_func,
):
    query_text = get_query_text(query)
    as_of = parse_date_object(query.get("as_of"))
    time_from, time_to = make_time_args(query)

    actual_mode = mode
    use_stale_penalty = True

    if mode == "full_no_penalty":
        actual_mode = "full"
        use_stale_penalty = False

    # 時間的リークを防止するため、as_of時点での有効メモ・グラフを取得
    graph, filtered_memos = get_temporal_filtered_graph_and_memos(
        all_prepared_memos, query, graph_cache, build_graph_func
    )

    active_graph = graph if actual_mode in ("graph", "full", "thought_evolution") else None

    kwargs = {
        "query": query_text,
        "mode": actual_mode,
        "threshold": THRESHOLD,
        "top_k": TOP_K,
        "as_of": as_of,
        "time_from": time_from,
        "time_to": time_to,
        "memos": filtered_memos,  # 時間フィルター済みのメモ群を渡す
        "query_vector": query_vector,
        "use_stale_penalty": use_stale_penalty,
    }

    if active_graph is not None:
        kwargs["graph"] = active_graph

    return search_core(**kwargs)


# ============================================================
# 評価メイン処理
# ============================================================

def evaluate():
    if not MEMO_FILE.exists():
        raise FileNotFoundError(f"評価メモがありません: {MEMO_FILE}")

    if not QUERY_FILE.exists():
        raise FileNotFoundError(f"評価質問がありません: {QUERY_FILE}")

    raw_memos = load_json(MEMO_FILE)
    queries = load_json(QUERY_FILE)

    print(f"memos   : {len(raw_memos)}")
    print(f"queries : {len(queries)}")

    from src.recommender import search_core
    from src.graph import build_graph

    memo_vectors, query_vectors = load_or_create_vectors(raw_memos, queries)
    all_prepared_memos = prepare_memos(raw_memos, memo_vectors)

    # 動的グラフキャッシュ（as_of ごとに生成・保持）
    graph_cache = {}

    all_rows = []
    all_details = []

    for mode in MODES:
        print(f"\n--- 評価モード: {mode} ---")

        for query in queries:
            qid = get_query_id(query)
            query_text = get_query_text(query)
            relevant_ids = get_relevant_ids(query)
            stale_ids = get_stale_ids(query)
            turning_ids = query.get("turning_ids", [])
            root_id = query.get("root_id")
            qvec = query_vectors[qid]

            try:
                results = call_search_core(
                    search_core,
                    query,
                    mode,
                    all_prepared_memos,
                    qvec,
                    graph_cache,
                    build_graph,
                )
            except Exception as e:
                print(f"[ERROR] mode={mode}, qid={qid}: {e}")
                raise

            if results is None:
                results = []

            ranked_ids = [result["id"] for result in results]

            row = {
                "mode": mode,
                "query_id": qid,
                "query_type": query.get("type", "default"),
                "mrr": reciprocal_rank(ranked_ids, relevant_ids),
                "p@1": precision_at_k(ranked_ids, relevant_ids, 1),
                "p@5": precision_at_k(ranked_ids, relevant_ids, 5),
                "p@10": precision_at_k(ranked_ids, relevant_ids, 10),
                "recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
                "recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
                "stale_rate@5": stale_rate_at_k(ranked_ids, stale_ids, k=5),
                "ndcg@10": ndcg_at_k(ranked_ids, relevant_ids, 10),
                "turning@5": turning_at_k(ranked_ids, turning_ids, k=5),
                "root@5": root_at_k(ranked_ids, root_id, k=5),
            }

            all_rows.append(row)

            all_details.append({
                "mode": mode,
                "query_id": qid,
                "query_type": query.get("type", "default"),
                "query": query_text,
                "as_of": query.get("as_of"),
                "relevant_ids": relevant_ids,
                "stale_ids": stale_ids,
                "results": results,
            })

    save_results(all_rows, all_details)
    print_summary(all_rows)


# ============================================================
# 結果保存
# ============================================================

def save_results(rows, details):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "mode",
        "query_id",
        "query_type",
        "mrr",
        "p@1",
        "p@5",
        "p@10",
        "recall@5",
        "recall@10",
        "stale_rate@5",
        "ndcg@10",
        "turning@5",
        "root@5",
    ]

    # 1. 各クエリ単位の詳細結果CSV
    with open(RESULT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 2. 手法ごとの平均サマリーCSV
    summary_rows = []
    for mode in MODES:
        m_rows = [r for r in rows if r["mode"] == mode]
        if not m_rows:
            continue
        
        s_row = {"mode": mode, "n": len(m_rows)}
        for key in fields[3:]:
            vals = [r[key] for r in m_rows if r[key] is not None]
            s_row[key] = float(np.mean(vals)) if vals else float("nan")
        summary_rows.append(s_row)

    with open(SUMMARY_FILE, "w", encoding="utf-8-sig", newline="") as f:
        s_fields = ["mode", "n"] + fields[3:]
        writer = csv.DictWriter(f, fieldnames=s_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    save_json(DETAIL_FILE, details)

    print("\n出力ファイル:")
    print(f"  - クエリ別詳細CSV: {RESULT_FILE}")
    print(f"  - 手法別平均CSV  : {SUMMARY_FILE}")
    print(f"  - 検索結果JSON   : {DETAIL_FILE}")


# ============================================================
# 評価結果コンソール表示
# ============================================================

def print_summary(rows):
    print("\n" + "=" * 95)
    print(" === 1. 全体平均 評価結果 (スライド対応 ＋ 拡張指標) ===")
    print("=" * 95)

    main_metrics = ["mrr", "p@1", "stale_rate@5", "recall@10", "ndcg@10"]
    headers = {
        "mrr": "MRR (平均逆順位)",
        "p@1": "P@1 (1位的中率)",
        "stale_rate@5": "Stale Rate (混入率)",
        "recall@10": "Recall@10 (網羅率)",
        "ndcg@10": "nDCG@10 (総合順位)",
    }

    print(
        f"{'検索手法 (Mode)':<25}"
        + "".join(f"{headers[m]:>18}" for m in main_metrics)
    )
    print("-" * 115)

    for mode in MODES:
        m_rows = [row for row in rows if row["mode"] == mode]
        if not m_rows:
            continue

        averages = {
            m: sum(r[m] for r in m_rows if r[m] is not None) / len(m_rows)
            for m in main_metrics
        }

        print(
            f"{mode:<25}"
            + "".join(f"{averages[m]:>18.4f}" for m in main_metrics)
        )

    # 思考系譜指標
    has_turning = any(r["turning@5"] is not None for r in rows)
    has_root = any(r["root@5"] is not None for r in rows)

    if has_turning or has_root:
        print("\n" + "=" * 95)
        print(" === 2. 思考系譜指標 (Turning Point & Root Note) ===")
        print("=" * 95)

        extra_metrics = []
        if has_turning: extra_metrics.append("turning@5")
        if has_root: extra_metrics.append("root@5")

        extra_headers = {
            "turning@5": "Turning Point@5 (転換点)",
            "root@5": "Root Note@5 (起点)",
        }

        print(
            f"{'検索手法 (Mode)':<25}"
            + "".join(f"{extra_headers[m]:>25}" for m in extra_metrics)
        )
        print("-" * 75)

        for mode in MODES:
            m_rows = [row for row in rows if row["mode"] == mode]
            if not m_rows:
                continue

            averages = {}
            for m in extra_metrics:
                valid_vals = [r[m] for r in m_rows if r[m] is not None]
                averages[m] = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

            print(
                f"{mode:<25}"
                + "".join(f"{averages[m]:>25.4f}" for m in extra_metrics)
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print("\n評価中にエラーが発生しました。")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)