"""
ラベルなしの「思考の系譜」推薦の評価スクリプト
（devでのtau選択、paired_ciによる信頼区間算出、バグ修正・整合性対応版）

実行（プロジェクトのルートで）:
    python -m experiments.evaluate_lineage                     # v3 (既定) / dim256
    python -m experiments.evaluate_lineage --dataset v2        # v2 データセット
    python -m experiments.evaluate_lineage --dim 768           # 768次元ベクトルで実行

入力: data/eval/thought_{memos,queries}_eval[_v2].json
出力: results/lineage_eval_per_query[_v3]_dim256.csv, results/lineage_eval_summary[_v3]_dim256.csv
"""

import argparse
import copy
import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import evaluate as ev
from src.graph import build_graph
from src.recommender import search_core

TAU_GRID = [round(x, 2) for x in np.arange(0.56, 0.861, 0.02)]
BOOTSTRAP = 5000
SEED = 42


def strip_labels(memos):
    """検索側に渡すメモから、評価用の正解情報を取り除く。"""
    out = copy.deepcopy(memos)
    for m in out:
        m["chain_id"] = None
        m["version"] = None
    return out


class Runner:
    def __init__(self, labeled, queries, query_vectors):
        self.labeled = labeled
        self.nolabel = strip_labels(labeled)
        self.queries = queries
        self.qv = query_vectors
        vectors = np.array([m["vector"] for m in self.nolabel], dtype=float)
        self.graph = build_graph(self.nolabel, vectors)

    def search(self, q, mode, memos=None, top_k=100, **kw):
        memos = self.nolabel if memos is None else memos
        if mode != "thought_evolution_oracle":
            assert all(m["chain_id"] is None and m["version"] is None for m in memos), \
                "ラベルありのメモを、oracle 以外の手法に渡してはいけません"
        as_of = ev.parse_date_object(q.get("as_of"))
        time_from, time_to = ev.make_time_args(q)
        return search_core(
            query=ev.get_query_text(q), mode=mode, threshold=0.0, top_k=top_k,
            as_of=as_of, time_from=time_from, time_to=time_to, memos=memos,
            query_vector=self.qv[ev.get_query_id(q)], graph=self.graph, **kw,
        ) or []

    # ---- 各種手法 ----
    def semantic(self, q):
        return self.search(q, "semantic")

    def graph_(self, q):
        return self.search(q, "graph")

    def full(self, q):
        return self.search(q, "full")

    def recency_demote(self, q):
        """上位5件のうち最も新しいメモを最下位へ配置するベースライン。"""
        res = self.search(q, "semantic")
        top = res[:5]
        newest = max(top, key=lambda r: r["created_at"]) if top else None
        return [r for r in res if r is not newest] + ([newest] if newest else [])

    def old_first(self, q):
        """上位5件を作成日の古い順に並べ替えるベースライン。"""
        res = self.search(q, "semantic")
        return sorted(res[:5], key=lambda r: r["created_at"]) + res[5:]

    def lineage(self, tau):
        return lambda q: self.search(q, "thought_evolution", lineage_tau=tau,
                                     lineage_include_anchors=True)

    def oracle(self, q):
        return self.search(q, "thought_evolution_oracle", memos=self.labeled)


METRICS = ("mrr", "p@1", "stale_rate@5", "recall@10", "ndcg@10", "turning@5", "root@5")


def per_query_metrics(results, q):
    """各クエリごとの評価指標計算"""
    ranked = [r["id"] for r in results][:ev.TOP_K]
    rel = ev.get_relevant_ids(q)
    stale_ids = ev.get_stale_ids(q)
    top5 = set(ranked[:5])
    turning = [t for t in q.get("turning_ids", []) if t in rel]
    root = q.get("root_id")
    
    return {
        "mrr": ev.reciprocal_rank(ranked, rel),
        "p@1": ev.precision_at_k(ranked, rel, 1),
        "stale_rate@5": ev.stale_rate_at_k(ranked, stale_ids, k=5),
        "recall@10": ev.recall_at_k(ranked, rel, 10),
        "ndcg@10": ev.ndcg_at_k(ranked, rel, 10),
        "turning@5": (sum(t in top5 for t in turning) / len(turning)) if turning else None,
        "root@5": (1.0 if root in top5 else 0.0) if (root and root in rel) else None,
    }


def evaluate_method(fn, queries):
    return {ev.get_query_id(q): per_query_metrics(fn(q), q) for q in queries}


def mean_of(rows, metric):
    vals = [r[metric] for r in rows.values() if r[metric] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def paired_ci(a_rows, b_rows, metric, cluster_of, seed=SEED):
    """
    a - b の平均差と、95% 信頼区間をブートストラップ法で算出。
    """
    rng = random.Random(seed)
    clusters = {}
    for q in sorted(a_rows):
        if a_rows[q][metric] is None or b_rows[q][metric] is None:
            continue
        clusters.setdefault(cluster_of[q], []).append(a_rows[q][metric] - b_rows[q][metric])
    keys = sorted(clusters)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    allv = [v for k in keys for v in clusters[k]]
    boots = []
    for _ in range(BOOTSTRAP):
        draw = [v for k in (rng.choice(keys) for _ in keys) for v in clusters[k]]
        boots.append(float(np.mean(draw)))
    boots.sort()
    return float(np.mean(allv)), boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP) - 1]


def split_dev_test(queries):
    """query に split があればそれを使う。無ければ系列を交互に分ける。"""
    if all("split" in q for q in queries):
        return ([q for q in queries if q["split"] == "dev"],
                [q for q in queries if q["split"] == "test"])
    chains = sorted({q.get("chain_id") for q in queries if q.get("chain_id")})
    if not chains:
        return queries, []
    dev_chains = set(chains[0::2])
    return ([q for q in queries if q.get("chain_id") in dev_chains],
            [q for q in queries if q.get("chain_id") not in dev_chains])


def configure_dataset(name):
    """データセット名に応じて evaluate.py の読み込みファイルパスを切り替える。"""
    if name == "v2":
        ev.MEMO_FILE = ev.EVAL_DIR / "thought_memos_eval_v2.json"
        ev.QUERY_FILE = ev.EVAL_DIR / "thought_queries_eval_v2.json"
        ev.VECTOR_FILE = ev.RESULT_DIR / "thought_eval_vectors_v2.json"
        return "_v2"
    elif name == "v1":
        ev.MEMO_FILE = ev.EVAL_DIR / "thought_memos_eval_v1.json"
        ev.QUERY_FILE = ev.EVAL_DIR / "thought_queries_eval_v1.json"
        ev.VECTOR_FILE = ev.RESULT_DIR / "thought_eval_vectors_v1.json"
        return "_v1"
    else:  # v3 (デフォルト)
        ev.MEMO_FILE = ev.EVAL_DIR / "thought_memos_eval.json"
        ev.QUERY_FILE = ev.EVAL_DIR / "thought_queries_eval.json"
        ev.VECTOR_FILE = ev.RESULT_DIR / "thought_eval_vectors.json"
        return "_v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["v1", "v2", "v3"], default="v3")
    ap.add_argument("--dim", type=int, choices=[128, 256, 768, 3072], default=256)
    args = ap.parse_args()

    dataset_suffix = configure_dataset(args.dataset)
    suffix = f"{dataset_suffix}_dim{args.dim}"

    raw = ev.load_json(ev.MEMO_FILE)
    # バグ修正: or True を削除し、意図した条件に限定
    queries = [q for q in ev.load_json(ev.QUERY_FILE) if q.get("type", "").startswith("thought")]
    if not queries:
        queries = ev.load_json(ev.QUERY_FILE)  # フォールバックとして全件取得

    memo_vectors = {}
    for m in raw:
        if "vector" in m and m["vector"]:
            memo_vectors[m["id"]] = m["vector"]

    v_file = ROOT / "results" / f"memos_eval_vec_d{args.dim}.json"
    if v_file.exists():
        try:
            memo_vectors.update(ev.load_json(v_file))
        except Exception:
            pass

    query_vectors = {}
    q_vec_file = ROOT / "results" / f"queries_eval_vec_d{args.dim}.json"
    if q_vec_file.exists():
        try:
            query_vectors.update(ev.load_json(q_vec_file))
        except Exception:
            pass

    for q in queries:
        qid = ev.get_query_id(q)
        if qid not in query_vectors:
            # 整合性担保のため、ベクトルが存在しない場合は明確にエラーとするかevaluate側のキャッシュを利用
            raise RuntimeError(f"質問 {qid} のベクトルが見つかりません。事前にベクトル生成を行ってください。")

    labeled = ev.prepare_memos(raw, memo_vectors)
    run = Runner(labeled, queries, query_vectors)
    dev, test = split_dev_test(queries)
    eval_target_queries = test if test else dev
    print(f"[{args.dataset}] メモ {len(labeled)}件 / クエリ dev {len(dev)}問・test {len(test)}問 (評価対象: {'test' if test else 'dev (全件)'})")

    cluster_of = {ev.get_query_id(q): q.get("chain_id", ev.get_query_id(q)) for q in queries}

    # ==========================================
    # 1. Dev データセットによる最適ハイパーパラメータ (tau) の選択
    # ==========================================
    best_tau = 0.78  # デフォルトフォールバック
    if dev:
        best_tau_score = -1.0
        print("\n--- Dev データセットによる tau チューニング ---")
        for tau in TAU_GRID:
            dev_res = evaluate_method(run.lineage(tau), dev)
            score = mean_of(dev_res, "mrr")
            print(f"  tau={tau:.2f} -> Dev MRR={score:.4f}")
            if score > best_tau_score:
                best_tau_score = score
                best_tau = tau
        print(f"-> 最適 tau を決定しました: {best_tau} (Dev MRR: {best_tau_score:.4f})\n")

    # ==========================================
    # 2. 手法群の定義 (選択された best_tau の lineage を含む)
    # ==========================================
    methods = {
        "semantic": run.semantic,
        "graph": run.graph_,
        "full": run.full,
        "recency_demote": run.recency_demote,
        "old_first": run.old_first,
        f"lineage_tau{best_tau}": run.lineage(best_tau),
        "oracle": run.oracle,
    }

    summary_rows = []
    per_query_rows = []
    results_cache = {}

    for name, fn in methods.items():
        res = evaluate_method(fn, eval_target_queries)
        results_cache[name] = res

        row = {"method": name, "n": len(eval_target_queries)}
        for m in METRICS:
            row[m] = mean_of(res, m)
        summary_rows.append(row)

        for qid, metrics in res.items():
            per_query_rows.append({"method": name, "qid": qid, **metrics})

    # ==========================================
    # 3. 統計的検定 (paired_ci) による信頼区間の計算・出力
    # ==========================================
    baseline_name = "full"
    if baseline_name in results_cache:
        print(f"\n--- 統計的評価 (基準モデル: {baseline_name} との差分 / 95% 信頼区間) ---")
        base_res = results_cache[baseline_name]
        for name in methods:
            if name == baseline_name:
                continue
            target_res = results_cache[name]
            diff_mrr, ci_low, ci_high = paired_ci(target_res, base_res, "mrr", cluster_of)
            print(f"  {name} vs {baseline_name} (MRR差): {diff_mrr:+.4f} [95% CI: {ci_low:+.4f}, {ci_high:+.4f}]")

    # ==========================================
    # 4. 結果の保存
    # ==========================================
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_csv = out_dir / f"lineage_eval_per_query{suffix}.csv"
    summary_csv = out_dir / f"lineage_eval_summary{suffix}.csv"

    ev.save_json(out_dir / f"lineage_eval_details{suffix}.json", results_cache)

    with open(per_query_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_rows)

    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        fields = ["method", "n"] + list(METRICS)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n評価完了。結果を保存しました:\n  - {summary_csv}\n  - {per_query_csv}")


if __name__ == "__main__":
    main()