"""
検索手法の比較評価（完全整合性・統計的検定網羅版）。

比較する手法（すべて src/ の本番コードを使う。評価用の別実装は持たない）:
  bm25              キーワード検索（文字バイグラムBM25）
  semantic          埋め込みのコサイン類似度のみ
  hybrid_bm25_sem   0.7*semantic + 0.3*BM25
  graph             semantic + Personalized PageRank
  full_no_stale     full から「更新済みメモの減点」だけ外したもの
  full              semantic + graph + 更新済み減点（本システム）
  thought_evolution 思考の系譜推薦（ラベルなし・実力値）
  thought_evolution_oracle 思考の系譜推薦（ラベルあり・上限値）

指標: MRR / Hit@1 / Hit@3 / nDCG@5 / StaleAbove（古い版が最新版より上に来る確率。小さいほど良い）
信頼区間: クエリに対するブートストラップ (95%)
"""

import argparse
import copy
import csv
import datetime
import json
import platform
import random

import numpy as np

from experiments.common import (
    check_no_leak,
    eval_paths,
    load_distractors,
    read_json,
    variant_tag,
)
from src import config
from src.data_manager import live_memos
from src.graph import build_graph
from src.metrics import (
    bootstrap_ci,
    hit_at_k,
    ndcg_at_k,
    paired_bootstrap_diff,
    reciprocal_rank,
    stale_above,
)
from src.recommender import search_core
def load_optimized_params():
    """results/ 以下の最適な best_params.json があれば読み込み、なければ config のデフォルト値を返す"""
    # 例として、共通のルート（または最新のパラメータファイル）を指定
    # ※もし実験ごとのフォルダから読みたい場合はパスを調整してください
    best_params_path = config.RESULTS_DIR / "best_params.json"
    params = {
        "weights": (0.6, 0.2, 0.2), 
        "stale_penalty": config.STALE_PENALTY,
        "ppr_alpha": config.PPR_ALPHA,
        "knn_k": config.KNN_K,
        "edge_min_sim": config.EDGE_MIN_SIM,
    }
    if best_params_path.exists():
        try:
            with open(best_params_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if "weights" in saved:
                    saved["weights"] = tuple(saved["weights"])
                params.update(saved)
                print(f"[情報] チューニング済みの最適パラメータを適用します: {best_params_path}")
        except Exception as e:
            print(f"[警告] best_params.json の読み込みに失敗しました（デフォルト値を使用します）: {e}")
    return params


class Evaluator:

    def __init__(self, memos):
        self.memos = memos
        self._graphs = {}
        # 【追加】インスタンス化されたときに最適パラメータをロードして保持する
        self.opt_params = load_optimized_params()

    def graph_for(self, as_of):
        """as_of ごとの候補集合に対するグラフ（ロードした最適パラメータの knn_k と edge_min_sim を使う）。"""
        if as_of not in self._graphs:
            cands = [
                m for m in live_memos(self.memos, as_of) if "vector" in m
            ]
            vecs = np.array([m["vector"] for m in cands])
            # config の値ではなく、ロードした最適パラメータを渡す！
            self._graphs[as_of] = build_graph(
                cands, vecs, 
                k=self.opt_params["knn_k"], 
                min_sim=self.opt_params["edge_min_sim"]
            )

        return self._graphs[as_of]

    def _core(self, q, mode, as_of, stale_penalty=None, **kw):
        time_from = (
            datetime.date.fromisoformat(q["time_from"]) if q.get("time_from") else None
        )
        time_to = (
            datetime.date.fromisoformat(q["time_to"]) if q.get("time_to") else None
        )

        # stale_penalty が明示されていなければ、ロードした最適パラメータを使う
        effective_penalty = stale_penalty if stale_penalty is not None else self.opt_params["stale_penalty"]

        res = search_core(
            q["text"],
            mode=mode,
            threshold=-1.0,
            top_k=BIG,
            as_of=as_of,
            time_from=time_from,
            time_to=time_to,
            memos=self.memos,
            query_vector=q.get("query_vector"),
            stale_penalty=effective_penalty,
            weights=self.opt_params["weights"],      # ← 最適化された重みを反映
            ppr_alpha=self.opt_params["ppr_alpha"],  # ← 最適化されたalphaを反映
            graph=self.graph_for(as_of) if mode in ("graph", "full", "thought_evolution", "thought_evolution_oracle") else None,
            **kw,
        )
        return {r["id"]: r["raw_score"] for r in res}
METHODS = [
    "bm25",
    "semantic",
    "hybrid_bm25_sem",
    "graph",
    "full_no_stale",
    "full",
    "thought_evolution",         
    "thought_evolution_oracle",
]

METRICS = ["mrr", "hit1", "hit3", "ndcg5", "stale_above"]

BIG = 10 ** 9


class Evaluator:

    def __init__(self, memos):
        self.memos = memos
        self._graphs = {}

    def graph_for(self, as_of):
        """as_of ごとの候補集合に対するグラフ（クエリに依存しないのでキャッシュ）。"""
        if as_of not in self._graphs:
            cands = [
                m for m in live_memos(self.memos, as_of) if "vector" in m
            ]
            vecs = np.array([m["vector"] for m in cands])
            self._graphs[as_of] = build_graph(cands, vecs)

        return self._graphs[as_of]

    def _core(self, q, mode, as_of, stale_penalty=None, **kw):
        # 期間指定つきクエリ(windowed)は、期間を検索に渡す
        time_from = (
            datetime.date.fromisoformat(q["time_from"]) if q.get("time_from") else None
        )
        time_to = (
            datetime.date.fromisoformat(q["time_to"]) if q.get("time_to") else None
        )

        res = search_core(
            q["text"],
            mode=mode,
            threshold=-1.0,
            top_k=BIG,
            as_of=as_of,
            time_from=time_from,
            time_to=time_to,
            memos=self.memos,
            query_vector=q.get("query_vector"),
            stale_penalty=stale_penalty,
            graph=self.graph_for(as_of) if mode in ("graph", "full", "thought_evolution", "thought_evolution_oracle") else None,
            **kw,
        )
        return {r["id"]: r["raw_score"] for r in res}

    def scores(self, method, q, stale_penalty=None):
        """全候補の {id: score}。"""
        as_of = (
            datetime.date.fromisoformat(q["as_of"]) if q.get("as_of") else None
        )

        if method == "bm25":
            return self._core(q, "keyword", as_of)

        if method == "semantic":
            return self._core(q, "semantic", as_of)

        if method == "graph":
            return self._core(q, "graph", as_of)

        if method == "full_no_stale":
            return self._core(q, "full", as_of, stale_penalty=1.0)

        if method == "full":
            return self._core(q, "full", as_of, stale_penalty=stale_penalty)

        if method == "thought_evolution":
            return self._core(q, "thought_evolution", as_of, lineage_include_anchors=True)

        if method == "thought_evolution_oracle":
            return self._core(q, "thought_evolution_oracle", as_of)

        if method == "hybrid_bm25_sem":
            sem = self._core(q, "semantic", as_of)
            kw = self._core(q, "keyword", as_of)
            return {k: 0.7 * sem[k] + 0.3 * kw.get(k, 0.0) for k in sem}

        raise ValueError(f"未知の検索手法: {method}")


def per_query_metrics(scores, q):
    rel = q["relevant_id"]
    return {
        "mrr": reciprocal_rank(scores, rel),
        "hit1": hit_at_k(scores, rel, 1),
        "hit3": hit_at_k(scores, rel, 3),
        "ndcg5": ndcg_at_k(scores, rel, 5),
        "stale_above": stale_above(scores, rel, q.get("stale_ids", [])),
    }


def evaluate(ev, queries, method, stale_penalty=None):
    """クエリごとの指標リストを返す。"""
    return [
        per_query_metrics(ev.scores(method, q, stale_penalty), q)
        for q in queries
    ]


def summarize(rows, seed):
    """指標ごとの平均と、MRR / StaleAbove の95%信頼区間。"""
    out = {"n": len(rows)}

    for m in METRICS:
        vals = [r[m] for r in rows]
        out[m] = float(np.mean(vals)) if vals else float("nan")

    for m in ("mrr", "stale_above"):
        lo, hi = bootstrap_ci([r[m] for r in rows], seed=seed)
        out[f"{m}_ci_low"] = lo
        out[f"{m}_ci_high"] = hi

    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def drop_links(memos, rate, seed):
    """
    AIの更新判定の取りこぼしを模擬する: 正解の supersedes リンクを rate の割合で消す。
    副作用（元のデータの書き換え）を防ぐため deepcopy を使用。
    """
    rng = random.Random(seed)
    out = []

    for m in memos:
        c = copy.deepcopy(m)
        c["supersedes"] = [s for s in m.get("supersedes", []) if rng.random() >= rate]
        out.append(c)

    return out


def make_figure(path, summary_current):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["method"] for r in summary_current]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, key, title in (
        (axes[0], "mrr", "MRR (higher is better)"),
        (axes[1], "stale_above", "StaleAbove (lower is better)"),
    ):
        vals = np.array([r[key] for r in summary_current])
        lo = np.array([r[f"{key}_ci_low"] for r in summary_current])
        hi = np.array([r[f"{key}_ci_high"] for r in summary_current])

        ax.bar(x, vals, yerr=[vals - lo, hi - vals], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Current-state queries, 95% bootstrap CI")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true",
                        help="偽の埋め込みで動作確認（結果は報告に使わない）")
    parser.add_argument("--distractors", type=int, default=0,
                        help="既存のWikipediaメモを背景ノイズとして加える件数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None, help="結果の保存フォルダ名")
    parser.add_argument("--dim", default=None,
                        help="評価する次元: min / recommended / max / 128〜3072の整数")
    parser.add_argument("--model", default=None,
                        help="埋め込みモデル名（省略時は .env）")
    parser.add_argument("--style", default=None,
                        help="入力形式 legacy / documented（省略時は .env）")
    args = parser.parse_args()

    dim = config.resolve_embed_dim(
        config.EMBED_DIM if args.dim is None else args.dim
    )
    model = config.EMBED_MODEL if args.model is None else args.model.strip()
    style = config.resolve_embed_style(
        config.EMBED_STYLE if args.style is None else args.style
    )
    tag = variant_tag(model, style)

    name = args.name or (
        f"smoke_fake_d{dim}{tag}" if args.fake else f"main_d{dim}{tag}"
    )
    out_dir = config.RESULTS_DIR / name

    paths = eval_paths(fake=args.fake, dim=dim, model=model, style=style)

    if not paths["memos_vec"].exists() or not paths["queries_vec"].exists():
        raise SystemExit(
            f"次元 {dim} のベクトル化済みデータがありません。先に実行してください:\n"
            "  python -m experiments.build_eval_dataset\n"
            f"  python -m experiments.embed_dataset --dim {dim} --model {model} --style {style}"
        )

    memos = read_json(paths["memos_vec"])
    queries = read_json(paths["queries_vec"])

    check_no_leak(memos, queries)

    if args.distractors and tag and not args.fake:
        print("注意: 背景ノイズ(Wikipedia)のベクトルは既定のモデル・形式で作られているため、"
              "別の条件では使いません（空間が違うと比較が壊れる）。")

    distractors = (
        [] if (args.fake or tag)
        else load_distractors(args.distractors, args.seed, dim)
    )
    all_memos = memos + distractors

    print(f"メモ {len(memos)}件 + 背景ノイズ {len(distractors)}件 / クエリ {len(queries)}件")

    ev = Evaluator(all_memos)

    by_type = {
        t: [q for q in queries if q["type"] == t]
        for t in sorted({q["type"] for q in queries})
    }

    # ---------- メイン比較 ----------

    main_rows = []
    per_query_rows = []
    cache = {}

    for method in METHODS:
        for qtype, qs in by_type.items():
            rows = evaluate(ev, qs, method)
            cache[(method, qtype)] = rows

            main_rows.append({
                "method": method, "query_type": qtype, **summarize(rows, args.seed),
            })

            for q, r in zip(qs, rows):
                per_query_rows.append({
                    "method": method, "qid": q["qid"], "query_type": qtype,
                    "kind": q["kind"], **r,
                })

        print(f"  done: {method}")

    write_csv(out_dir / "main_results.csv", main_rows)
    write_csv(out_dir / "per_query.csv", per_query_rows)

    # ---------- 種類別（current のみ） ----------

    kind_rows = []

    for method in METHODS:
        for kind in sorted({q["kind"] for q in by_type["current"]}):
            idx = [i for i, q in enumerate(by_type["current"]) if q["kind"] == kind]
            rows = [cache[(method, "current")][i] for i in idx]
            kind_rows.append({"method": method, "kind": kind, **summarize(rows, args.seed)})

    write_csv(out_dir / "by_kind_current.csv", kind_rows)

    # ---------- 対応のある差の検定（current：提案手法を含む全手法と full との比較） ----------

    paired = []
    comparison_targets = [m for m in METHODS if m != "full"]

    for other in comparison_targets:
        for metric in ("mrr", "stale_above"):
            a = [r[metric] for r in cache[("full", "current")]]
            b = [r[metric] for r in cache[(other, "current")]]
            d, lo, hi = paired_bootstrap_diff(a, b, seed=args.seed)
            paired.append({
                "comparison": f"full - {other}", "metric": metric,
                "mean_diff": d, "ci_low": lo, "ci_high": hi,
                "significant": not (lo <= 0 <= hi),
            })

    write_csv(out_dir / "paired_diff_current.csv", paired)

    # ---------- 減点係数のスイープ ----------

    sweep = []

    for sp in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
        for qtype, qs in by_type.items():
            sweep.append({
                "stale_penalty": sp, "query_type": qtype,
                **summarize(evaluate(ev, qs, "full", stale_penalty=sp), args.seed),
            })

    write_csv(out_dir / "sweep_stale_penalty.csv", sweep)

    # ---------- 更新判定の取りこぼしへの頑健性 ----------

    robust = []

    for rate in (0.0, 0.2, 0.4, 0.6, 1.0):
        noisy = Evaluator(drop_links(all_memos, rate, args.seed))
        rows = evaluate(noisy, by_type["current"], "full")
        robust.append({"link_drop_rate": rate, **summarize(rows, args.seed)})

    write_csv(out_dir / "robustness_supersedes.csv", robust)

    # ---------- 図・実験条件 ----------

    make_figure(
        out_dir / "main_current.png",
        [r for r in main_rows if r["query_type"] == "current"],
    )

    run_config = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "embedding": (
            "FAKE hash embedding (smoke test only; do not report)"
            if args.fake else
            f"{model} dim={dim} style={style}"
        ),
        "embed_dim": dim,
        "embed_model": model,
        "embed_style": style,
        "n_eval_memos": len(memos),
        "n_distractors": len(distractors),
        "n_queries": {t: len(qs) for t, qs in by_type.items()},
        "params": {
            "KNN_K": config.KNN_K,
            "EDGE_MIN_SIM": config.EDGE_MIN_SIM,
            "PPR_ALPHA": config.PPR_ALPHA,
            "STALE_PENALTY": config.STALE_PENALTY,
            "THINKING_EVOLUTION_ANCHORS": True,  # 思考の系譜のアンカー設定を明文化
        },
        "supersedes_labels": "oracle (ground truth)",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # ---------- コンソール表示 ----------

    print("\n=== current（今の答えを探すクエリ） ===")
    print(f"{'method':28s} {'MRR':>6s} {'Hit@1':>6s} {'nDCG@5':>7s} {'StaleAbove':>11s}")

    for r in main_rows:
        if r["query_type"] == "current":
            print(f"{r['method']:28s} {r['mrr']:6.3f} {r['hit1']:6.3f} "
                  f"{r['ndcg5']:7.3f} {r['stale_above']:11.3f}")

    print(f"\n結果の保存先: {out_dir}")


if __name__ == "__main__":
    main()