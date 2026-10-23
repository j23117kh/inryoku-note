"""
数値（ハイパーパラメータ）と手法を、複数の候補で比べる（API不要。ベクトル化済みデータを使う）。

  python -m experiments.sweep_params --dim recommended

何を比べるか
  1. 感度分析（1つずつ動かす）
       重み (semantic, graph, time)、PPRの減衰係数、kNNグラフの k、エッジの類似度閾値、
       古い版の減点係数、BM25 の k1 / b、BM25 と semantic の混ぜ方の重み
  2. 手法の比較（値ではなく「やり方」の違い）
       古い版の扱い: 減点なし / 減点 / 後回し(demote) / 除外(exclude)
       融合方式    : 重み付き和 / RRF(Reciprocal Rank Fusion, k=60)
  3. クロスバリデーションによる「調整した値 vs 既定値」の比較
       調整に使っていないクエリで評価する（調整した値を同じデータで評価すると楽観的になるため）。
       ベースライン(BM25, hybrid)も同じ条件で調整し、公平に比べる。
  4. 更新判定の候補選択の値（SUPERSEDE_MIN_SIM / SUPERSEDE_CANDIDATES）
       正解の更新元が候補に入る割合（API不要）。

結果は results/params_d<次元>[_<モデル>_<形式>]/ に保存される。

注意（読み方）
  - 感度分析の表は、全クエリで計算した記述的な数字。「この値が最適」の根拠には使わない
    （同じデータで選ぶと楽観的になる）。選ぶ根拠は、3 のクロスバリデーションの結果。
  - 人工データなので、結果は「この設定での傾向」。実メモでの最適値を示すものではない。
  - time の重みは windowed クエリ（期間指定つき）でだけ効く。
"""

import argparse
import datetime
import itertools
import json
import platform
import random

import numpy as np

from experiments.analyze_accuracy import select_candidates
from experiments.common import check_no_leak, eval_paths, read_json, variant_tag
from experiments.run_eval import METRICS, per_query_metrics, summarize, write_csv
from src import config
from src.data_manager import live_memos
from src.graph import build_graph
from src.metrics import paired_bootstrap_diff
from src.recommender import search_core

BIG = 10 ** 9

# ---------------------------------------------------------------- 既定値と候補

DEFAULT_PARAMS = {
    "weights": (0.6, 0.2, 0.2),
    "stale_penalty": config.STALE_PENALTY,
    "ppr_alpha": config.PPR_ALPHA,
    "knn_k": config.KNN_K,
    "edge_min_sim": config.EDGE_MIN_SIM,
}

# 1つずつ動かす候補（既定値を必ず含める）
GRIDS = {
    "stale_penalty": [1.0, 0.9, 0.75, 0.5, 0.25, 0.1],
    "ppr_alpha": [0.5, 0.7, 0.85, 0.95],
    "knn_k": [3, 5, 10, 20],
    "edge_min_sim": [0.3, 0.4, 0.5, 0.6, 0.7],
}

# BM25: Elasticsearch/Lucene の既定 (k1=1.2, b=0.75) と、このシステムの既定 (1.5, 0.75) を含む
DEFAULT_BM25 = (1.5, 0.75)
BM25_GRID = [(k1, b) for k1 in (0.9, 1.2, 1.5, 2.0) for b in (0.25, 0.5, 0.75, 1.0)]

# semantic と BM25 の混ぜ方（semantic 側の重み）。run_eval の hybrid は 0.7
DEFAULT_HYBRID_W = 0.7
HYBRID_GRID = [round(0.1 * i, 1) for i in range(11)]

RRF_K = 60  # Cormack, Clarke, Buettcher (2009) が使った値

# 小さいグリッド（テスト・動作確認用）
QUICK_GRIDS = {
    "stale_penalty": [1.0, 0.5, 0.1],
    "ppr_alpha": [0.5, 0.85],
    "knn_k": [3, 5],
    "edge_min_sim": [0.5, 0.7],
}


def weight_grid(step_tenths=1, min_sem_tenths=3):
    """
    (semantic, graph, time) の重みの候補。合計は1。
    整数（1/10 単位）で数えて、浮動小数点の誤差を避ける。
    semantic は min_sem_tenths/10 以上（検索語との意味的な近さを主役に保つための制約）。
    """

    out = []

    for s in range(min_sem_tenths, 11, step_tenths):
        for g in range(0, 11 - s, step_tenths):
            t = 10 - s - g
            if t % step_tenths == 0 and g % step_tenths == 0:
                out.append((s / 10, g / 10, t / 10))

    return out


def cfg_key(cfg):
    return (tuple(cfg["weights"]), cfg["stale_penalty"], cfg["ppr_alpha"],
            cfg["knn_k"], cfg["edge_min_sim"])


def with_param(param, value):
    cfg = dict(DEFAULT_PARAMS)
    cfg[param] = value
    return cfg


# ---------------------------------------------------------------- 順位の計算（RRF 用）

def average_ranks(scores):
    """
    {id: score} -> {id: 順位}（1始まり、スコアが高いほど上位）。同点は平均順位。
    並べ替えで数える（O(n log n)）。
    """

    items = sorted(scores.items(), key=lambda kv: -kv[1])
    ranks = {}
    i = 0

    while i < len(items):
        j = i
        while j + 1 < len(items) and items[j + 1][1] == items[i][1]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg
        i = j + 1

    return ranks


def rrf_fuse(semantic, bm25, k=RRF_K):
    """
    Reciprocal Rank Fusion。score = Σ 1 / (k + rank)。
    BM25 は「一致した文書だけ」を順位付きリストに入れる（一致なしは寄与0）。
    semantic は全文書を順位付きリストに入れる。
    """

    sem_rank = average_ranks(semantic)
    bm_rank = average_ranks({d: s for d, s in bm25.items() if s > 0})

    out = {}

    for d in semantic:
        v = 1.0 / (k + sem_rank[d])
        if d in bm_rank:
            v += 1.0 / (k + bm_rank[d])
        out[d] = v

    return out


# ---------------------------------------------------------------- 古い版の扱い方

STALE_DEMOTE_BONUS = 10.0  # 非stale を、スコアの取りうる範囲より常に上にする


def apply_stale_handling(scores, stale_ids, how, penalty=None):
    """
    scores は「減点なし」のスコア。how:
      none     何もしない
      penalty  stale に penalty を掛ける（本システムの方式）
      demote   stale を、すべての非stale の後ろに回す（順序は保つ）
      exclude  stale を結果から除く（履歴は見えなくなる）
    """

    if how == "none":
        return dict(scores)

    if how == "penalty":
        return {d: s * (penalty if d in stale_ids else 1.0) for d, s in scores.items()}

    if how == "demote":
        return {d: s + (0.0 if d in stale_ids else STALE_DEMOTE_BONUS)
                for d, s in scores.items()}

    if how == "exclude":
        return {d: s for d, s in scores.items() if d not in stale_ids}

    raise ValueError(how)


# ---------------------------------------------------------------- 評価器

class ParamEvaluator:
    """search_core（本番コード）に、値を注入して呼ぶ。グラフ・単純スコアはキャッシュする。"""

    def __init__(self, memos):
        self.memos = memos
        self._graphs = {}
        self._cache = {}

    @staticmethod
    def _dates(q):
        d = datetime.date.fromisoformat
        return (
            d(q["as_of"]) if q.get("as_of") else None,
            d(q["time_from"]) if q.get("time_from") else None,
            d(q["time_to"]) if q.get("time_to") else None,
        )

    def graph_for(self, as_of, k, min_sim):
        key = (as_of, k, min_sim)

        if key not in self._graphs:
            cands = [m for m in live_memos(self.memos, as_of) if "vector" in m]
            vecs = np.array([m["vector"] for m in cands])
            self._graphs[key] = build_graph(cands, vecs, k=k, min_sim=min_sim)

        return self._graphs[key]

    def _search(self, q, mode, **kw):
        as_of, tf, tt = self._dates(q)

        return search_core(
            q["text"], mode=mode, threshold=-1.0, top_k=BIG, as_of=as_of,
            time_from=tf, time_to=tt, memos=self.memos,
            query_vector=q.get("query_vector"), **kw,
        )

    def full(self, q, cfg):
        """full モード。戻り値: ({id: score}, stale な id の集合)"""

        key = ("full", q["qid"], cfg_key(cfg))

        if key not in self._cache:
            as_of, _, _ = self._dates(q)
            res = self._search(
                q, "full", weights=cfg["weights"], stale_penalty=cfg["stale_penalty"],
                ppr_alpha=cfg["ppr_alpha"],
                graph=self.graph_for(as_of, cfg["knn_k"], cfg["edge_min_sim"]),
            )
            self._cache[key] = (
                {r["id"]: r["raw_score"] for r in res},
                {r["id"] for r in res if r["stale"]},
            )

        return self._cache[key]

    def semantic(self, q):
        key = ("sem", q["qid"])

        if key not in self._cache:
            self._cache[key] = {r["id"]: r["raw_score"]
                                for r in self._search(q, "semantic")}

        return self._cache[key]

    def bm25(self, q, k1, b):
        key = ("bm25", q["qid"], k1, b)

        if key not in self._cache:
            self._cache[key] = {r["id"]: r["raw_score"]
                                for r in self._search(q, "keyword", bm25_k1=k1, bm25_b=b)}

        return self._cache[key]

    def hybrid(self, q, w, k1=DEFAULT_BM25[0], b=DEFAULT_BM25[1]):
        sem, bm = self.semantic(q), self.bm25(q, k1, b)
        return {d: w * sem[d] + (1.0 - w) * bm.get(d, 0.0) for d in sem}


def rows_for(queries, scores_fn):
    return [per_query_metrics(scores_fn(q), q) for q in queries]


def mean_mrr(rows, idx):
    return float(np.mean([rows[i]["mrr"] for i in idx])) if len(idx) else float("nan")


# ---------------------------------------------------------------- クロスバリデーション

def make_folds(queries, n_folds, seed):
    """
    チェーン単位で分割する（同じチェーンの current / as_of / windowed は必ず同じ分割に入る）。
    戻り値: 各クエリの分割番号のリスト。
    """

    chains = sorted({q["chain_id"] for q in queries})
    rng = random.Random(seed)
    rng.shuffle(chains)

    fold_of = {c: i % n_folds for i, c in enumerate(chains)}

    return [fold_of[q["chain_id"]] for q in queries]


def pick_best(values, train_scores, default):
    """
    train_scores が最大の値を選ぶ。最大が複数あれば（差が1e-12未満）、既定値を優先する
    （同点なら、動かさない）。既定値が含まれなければ先頭を選ぶ。
    """

    best = max(train_scores)
    tied = [v for v, s in zip(values, train_scores) if s >= best - 1e-12]

    return default if default in tied else tied[0]


def run_cv(ev, queries, grids, weights, seed, n_folds, bm25_grid=BM25_GRID,
           hybrid_grid=HYBRID_GRID):
    """
    各分割で「訓練クエリだけ」で値を選び、「テストクエリ」で評価する。全クエリがちょうど1回テストされる。

    戻り値: (pooled, selected)
      pooled   : {手法名: クエリごとの指標リスト（queries と同じ順）}
      selected : 選ばれた値の記録
    """

    n = len(queries)
    fold = make_folds(queries, n_folds, seed)

    # --- 全クエリでの指標を、候補ごとに1回だけ計算する（分割に依存しないので使い回せる）
    full_rows = {}

    def full_rows_for(cfg):
        k = cfg_key(cfg)
        if k not in full_rows:
            full_rows[k] = rows_for(queries, lambda q: ev.full(q, cfg)[0])
        return full_rows[k]

    bm_rows = {kb: rows_for(queries, lambda q, kb=kb: ev.bm25(q, *kb)) for kb in bm25_grid}
    hy_rows = {w: rows_for(queries, lambda q, w=w: ev.hybrid(q, w)) for w in hybrid_grid}
    sem_rows = rows_for(queries, ev.semantic)

    # --- 結果の入れ物
    pooled = {name: [None] * n for name in (
        "semantic", "bm25_default", "bm25_tuned", "hybrid_default", "hybrid_tuned",
        "full_default", "full_tuned")}
    selected = []

    for f in range(n_folds):

        train = [i for i in range(n) if fold[i] != f]
        test = [i for i in range(n) if fold[i] == f]

        if not test or not train:
            continue

        # BM25: (k1, b)
        kb = pick_best(bm25_grid, [mean_mrr(bm_rows[x], train) for x in bm25_grid], DEFAULT_BM25)
        selected.append({"fold": f, "family": "bm25", "param": "k1,b",
                         "value": f"{kb[0]}/{kb[1]}", "default": f"{DEFAULT_BM25[0]}/{DEFAULT_BM25[1]}",
                         "train_mrr": mean_mrr(bm_rows[kb], train)})

        # hybrid: semantic 側の重み
        w = pick_best(hybrid_grid, [mean_mrr(hy_rows[x], train) for x in hybrid_grid], DEFAULT_HYBRID_W)
        selected.append({"fold": f, "family": "hybrid", "param": "w_semantic",
                         "value": w, "default": DEFAULT_HYBRID_W,
                         "train_mrr": mean_mrr(hy_rows[w], train)})

        # full: 値ごとに、1つずつ最良を選んでから組み合わせる
        tuned = dict(DEFAULT_PARAMS)

        wtuple = pick_best(
            weights,
            [mean_mrr(full_rows_for(with_param("weights", x)), train) for x in weights],
            DEFAULT_PARAMS["weights"])
        tuned["weights"] = wtuple
        selected.append({"fold": f, "family": "full", "param": "weights",
                         "value": "/".join(map(str, wtuple)),
                         "default": "/".join(map(str, DEFAULT_PARAMS["weights"])),
                         "train_mrr": mean_mrr(full_rows_for(with_param("weights", wtuple)), train)})

        for param, values in grids.items():
            v = pick_best(
                values,
                [mean_mrr(full_rows_for(with_param(param, x)), train) for x in values],
                DEFAULT_PARAMS[param])
            tuned[param] = v
            selected.append({"fold": f, "family": "full", "param": param, "value": v,
                             "default": DEFAULT_PARAMS[param],
                             "train_mrr": mean_mrr(full_rows_for(with_param(param, v)), train)})

        # 組み合わせた設定は、テストクエリだけで新たに評価する
        tuned_rows = {i: per_query_metrics(ev.full(queries[i], tuned)[0], queries[i]) for i in test}
        default_rows = full_rows_for(DEFAULT_PARAMS)

        for i in test:
            pooled["semantic"][i] = sem_rows[i]
            pooled["bm25_default"][i] = bm_rows[DEFAULT_BM25][i]
            pooled["bm25_tuned"][i] = bm_rows[kb][i]
            pooled["hybrid_default"][i] = hy_rows[DEFAULT_HYBRID_W][i]
            pooled["hybrid_tuned"][i] = hy_rows[w][i]
            pooled["full_default"][i] = default_rows[i]
            pooled["full_tuned"][i] = tuned_rows[i]

    return pooled, selected


# ---------------------------------------------------------------- 更新判定の候補（API不要）

def candidate_stats(memos, top_k, min_sim):
    """
    /api/save と同じ手順で、時系列に候補を選び、正解の更新元が候補に入る割合を数える。
      recall     : 更新元(正解)を持つメモのうち、全部の更新元が候補に入った割合
      mean_cands : 1メモあたりの候補数（LLMに渡す量の目安）
      precision  : 候補のうち、実際に更新元だった割合
    """

    ordered = sorted((m for m in memos if "vector" in m), key=lambda m: m["created_at"])

    seen, superseded = [], set()
    hit, with_gold, n_cands, n_true, n_memos = 0, 0, 0, 0, 0

    for m in ordered:

        earlier = [x for x in seen if x["id"] not in superseded]
        cands = select_candidates(m, earlier, top_k, min_sim)
        gold = set(m.get("supersedes", []))
        cand_ids = {c["id"] for c in cands}

        n_memos += 1
        n_cands += len(cands)
        n_true += len(cand_ids & gold)

        if gold:
            with_gold += 1
            hit += gold <= cand_ids

        superseded |= gold
        seen.append(m)

    return {
        "recall": hit / with_gold if with_gold else float("nan"),
        "mean_cands": n_cands / n_memos if n_memos else float("nan"),
        "precision": n_true / n_cands if n_cands else float("nan"),
        "n_with_gold": with_gold,
    }


# ---------------------------------------------------------------- 図

def make_figures(out_dir, sens_rows, weights_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = []
    for r in sens_rows:
        if r["param"] != "weights" and r["param"] not in params:
            params.append(r["param"])

    if params:
        cols = 3
        nrows = (len(params) + cols - 1) // cols
        fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.4 * nrows), squeeze=False)

        for ax, param in zip(axes.ravel(), params):
            vals = []
            for r in sens_rows:
                if r["param"] == param and r["value"] not in vals:
                    vals.append(r["value"])
            for qt in ("current", "windowed", "as_of"):
                ys = [next((r["mrr"] for r in sens_rows if r["param"] == param
                            and r["value"] == v and r["query_type"] == qt), float("nan"))
                      for v in vals]
                ax.plot(range(len(vals)), ys, marker="o", label=qt)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([str(v) for v in vals], fontsize=7, rotation=30)
            ax.set_ylim(0, 1.05)
            ax.set_title(param, fontsize=9)

        for ax in axes.ravel()[len(params):]:
            ax.axis("off")

        axes[0][0].legend(fontsize=7)
        fig.suptitle("MRR vs each value (one at a time, all queries; descriptive)")
        fig.tight_layout()
        fig.savefig(out_dir / "sensitivity.png", dpi=150)
        plt.close(fig)

    if weights_rows:
        gs = sorted({r["w_graph"] for r in weights_rows})
        ts = sorted({r["w_time"] for r in weights_rows})

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, qt in zip(axes, ("current", "windowed")):
            grid = np.full((len(ts), len(gs)), np.nan)
            for r in weights_rows:
                if r["query_type"] == qt:
                    grid[ts.index(r["w_time"]), gs.index(r["w_graph"])] = r["mrr"]
            im = ax.imshow(grid, origin="lower", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(gs)))
            ax.set_xticklabels(gs, fontsize=7)
            ax.set_yticks(range(len(ts)))
            ax.set_yticklabels(ts, fontsize=7)
            ax.set_xlabel("graph weight")
            ax.set_ylabel("time weight")
            ax.set_title(f"MRR ({qt}); semantic = 1 - graph - time", fontsize=9)
            fig.colorbar(im, ax=ax)

        fig.tight_layout()
        fig.savefig(out_dir / "weights_heatmap.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- 本体

def by_type(queries):
    return {t: [i for i, q in enumerate(queries) if q["type"] == t]
            for t in sorted({q["type"] for q in queries})}


def summarize_idx(rows, idx, seed):
    return summarize([rows[i] for i in idx], seed)


def run_sweep(memos, queries, out_dir, seed=42, n_folds=5, grids=None, weights=None,
              bm25_grid=BM25_GRID, hybrid_grid=HYBRID_GRID, extra_config=None):
    """評価データ（ベクトル付き）を受け取り、結果を out_dir に書く。"""

    grids = GRIDS if grids is None else grids
    weights = weight_grid() if weights is None else weights

    if DEFAULT_PARAMS["weights"] not in weights:
        weights = [DEFAULT_PARAMS["weights"]] + list(weights)

    for p, vals in grids.items():
        if DEFAULT_PARAMS[p] not in vals:
            raise ValueError(f"{p} の候補に既定値 {DEFAULT_PARAMS[p]} が含まれていません。")

    ev = ParamEvaluator(memos)
    groups = by_type(queries)
    groups_all = {**groups, "all": list(range(len(queries)))}

    # ============ 1. 感度分析（1つずつ動かす。全クエリ。記述的）
    sens, weights_rows = [], []

    def sens_rows(family, param, value, rows, is_default):
        for qt, idx in groups_all.items():
            sens.append({"family": family, "param": param, "value": value,
                         "is_default": is_default, "query_type": qt,
                         **summarize_idx(rows, idx, seed)})

    for param, values in grids.items():
        for v in values:
            rows = rows_for(queries, lambda q, v=v: ev.full(q, with_param(param, v))[0])
            sens_rows("full", param, v, rows, v == DEFAULT_PARAMS[param])

    for wt in weights:
        rows = rows_for(queries, lambda q, wt=wt: ev.full(q, with_param("weights", wt))[0])
        label = "/".join(map(str, wt))
        sens_rows("full", "weights", label, rows, wt == DEFAULT_PARAMS["weights"])
        for qt, idx in groups_all.items():
            weights_rows.append({"w_semantic": wt[0], "w_graph": wt[1], "w_time": wt[2],
                                 "query_type": qt, **summarize_idx(rows, idx, seed)})

    for kb in bm25_grid:
        rows = rows_for(queries, lambda q, kb=kb: ev.bm25(q, *kb))
        sens_rows("bm25", "k1,b", f"{kb[0]}/{kb[1]}", rows, kb == DEFAULT_BM25)

    for w in hybrid_grid:
        rows = rows_for(queries, lambda q, w=w: ev.hybrid(q, w))
        sens_rows("hybrid", "w_semantic", w, rows, w == DEFAULT_HYBRID_W)

    write_csv(out_dir / "sensitivity.csv", sens)
    write_csv(out_dir / "weights_grid.csv", weights_rows)

    # ============ 2. 手法の比較
    base_cfg = dict(DEFAULT_PARAMS)
    nopen = dict(base_cfg, stale_penalty=1.0)

    def stale_fn(how, penalty=None):
        def fn(q):
            scores, stale = ev.full(q, nopen)
            return apply_stale_handling(scores, stale, how, penalty)
        return fn

    techniques = []

    variants = [
        ("stale_handling", "none（減点なし）", stale_fn("none")),
        ("stale_handling", f"penalty x{DEFAULT_PARAMS['stale_penalty']}（本システム）",
         stale_fn("penalty", DEFAULT_PARAMS["stale_penalty"])),
        ("stale_handling", "demote（非staleの後ろに回す）", stale_fn("demote")),
        ("stale_handling", "exclude（除外。履歴は見えなくなる）", stale_fn("exclude")),
        ("fusion", f"weighted（semantic {DEFAULT_HYBRID_W} + BM25 {round(1 - DEFAULT_HYBRID_W, 1)}）",
         lambda q: ev.hybrid(q, DEFAULT_HYBRID_W)),
        ("fusion", f"RRF（k={RRF_K}）", lambda q: rrf_fuse(ev.semantic(q), ev.bm25(q, *DEFAULT_BM25))),
        ("fusion", "semantic のみ（参考）", ev.semantic),
        ("fusion", "BM25 のみ（参考）", lambda q: ev.bm25(q, *DEFAULT_BM25)),
    ]

    for group, name, fn in variants:
        rows = rows_for(queries, fn)
        for qt, idx in groups_all.items():
            techniques.append({"group": group, "variant": name, "query_type": qt,
                               **summarize_idx(rows, idx, seed)})

    write_csv(out_dir / "techniques.csv", techniques)

    # ============ 3. クロスバリデーション
    pooled, selected = run_cv(ev, queries, grids, weights, seed, n_folds, bm25_grid, hybrid_grid)

    missing = [i for i, r in enumerate(pooled["full_tuned"]) if r is None]
    if missing:
        raise RuntimeError(f"{len(missing)}件のクエリがテストされませんでした（分割のバグ）。")

    cv_rows = []
    for name, rows in pooled.items():
        for qt, idx in groups_all.items():
            cv_rows.append({"method": name, "query_type": qt, **summarize_idx(rows, idx, seed)})

    write_csv(out_dir / "cv_summary.csv", cv_rows)
    write_csv(out_dir / "cv_selected.csv", selected)

    comparisons = [
        ("full_tuned", "full_default"),
        ("full_tuned", "bm25_tuned"),
        ("full_tuned", "hybrid_tuned"),
        ("full_tuned", "semantic"),
        ("full_default", "bm25_default"),
        ("full_default", "semantic"),
        ("hybrid_tuned", "hybrid_default"),
        ("bm25_tuned", "bm25_default"),
    ]

    paired = []
    for a, b in comparisons:
        for qt, idx in groups_all.items():
            for metric in ("mrr", "stale_above"):
                d, lo, hi = paired_bootstrap_diff(
                    [pooled[a][i][metric] for i in idx], [pooled[b][i][metric] for i in idx],
                    seed=seed)
                paired.append({"comparison": f"{a} - {b}", "query_type": qt, "metric": metric,
                               "mean_diff": d, "ci_low": lo, "ci_high": hi,
                               "significant": not (lo <= 0 <= hi)})

    write_csv(out_dir / "cv_paired.csv", paired)

    # ============ 4. 更新判定の候補選択（API不要）
    cand_rows = []
    for min_sim, top_k in itertools.product((0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (1, 3, 5, 10)):
        cand_rows.append({"min_sim": min_sim, "top_k": top_k,
                          **candidate_stats(memos, top_k, min_sim)})

    write_csv(out_dir / "supersede_candidates.csv", cand_rows)

    make_figures(out_dir, sens, weights_rows)

    run_config = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": seed, "folds": n_folds,
        "n_memos": len(memos),
        "n_queries": {t: len(i) for t, i in groups.items()},
        "defaults": {"weights": list(DEFAULT_PARAMS["weights"]),
                     "stale_penalty": DEFAULT_PARAMS["stale_penalty"],
                     "ppr_alpha": DEFAULT_PARAMS["ppr_alpha"],
                     "knn_k": DEFAULT_PARAMS["knn_k"],
                     "edge_min_sim": DEFAULT_PARAMS["edge_min_sim"],
                     "bm25": list(DEFAULT_BM25), "hybrid_w_semantic": DEFAULT_HYBRID_W,
                     "rrf_k": RRF_K},
        "grids": {**{k: list(v) for k, v in grids.items()},
                  "weights": [list(w) for w in weights],
                  "bm25": [list(x) for x in bm25_grid], "hybrid": list(hybrid_grid)},
        "selection_objective": "訓練クエリ全体の MRR の平均（同点なら既定値を優先）",
        "supersedes_labels": "oracle (ground truth)",
        "python": platform.python_version(), "numpy": np.__version__,
        **(extra_config or {}),
    }

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    from collections import Counter

    best_params = dict(DEFAULT_PARAMS)
    
    # 1. 各グリッドパラメータについて、CVの fold ごとの選択結果（selected）から最頻値をとる
    for param in GRIDS.keys():
        p_vals = [s["value"] for s in selected if s["family"] == "full" and s["param"] == param]
        if p_vals:
            most_common_val = Counter(p_vals).most_common(1)[0][0]
            if param in ("stale_penalty", "ppr_alpha", "edge_min_sim"):
                best_params[param] = float(most_common_val)
            elif param == "knn_k":
                best_params[param] = int(most_common_val)
            else:
                best_params[param] = most_common_val

    # 2. weights についても同様に CV の選択結果から最頻値をとる
    w_vals = [s["value"] for s in selected if s["family"] == "full" and s["param"] == "weights"]
    if w_vals:
        best_w_str = Counter(w_vals).most_common(1)[0][0]  # 例: "0.6/0.2/0.2"
        best_params["weights"] = tuple(map(float, best_w_str.split("/")))

    # 3. 実験ごとの出力フォルダ、および共通 results/ 直下の両方に保存
    best_params_path = out_dir / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    shared_path = config.RESULTS_DIR / "best_params.json"
    with open(shared_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print(f"最適パラメータ（CVベース）を保存しました: {best_params_path} および {shared_path}")
    return {
        "pooled": pooled,
        "selected": selected,
        "cv_rows": cv_rows,
        "paired": paired,
        "techniques": techniques,
        "sens": sens,
    }
    


def print_report(result):
    print("\n=== クロスバリデーション（調整に使っていないクエリでの結果 / 全クエリ）===")
    print(f"{'method':16s} {'MRR':>6s} {'Hit@1':>6s} {'StaleAbove':>11s}")
    for r in result["cv_rows"]:
        if r["query_type"] == "all":
            print(f"{r['method']:16s} {r['mrr']:6.3f} {r['hit1']:6.3f} {r['stale_above']:11.3f}")

    print("\n=== 主な差（全クエリ, MRR）: 平均差 [95%CI] ===")
    for r in result["paired"]:
        if r["query_type"] == "all" and r["metric"] == "mrr":
            mark = "有意" if r["significant"] else "差なし(CIが0を含む)"
            print(f"  {r['comparison']:34s} {r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  {mark}")

    print("\n=== 手法の比較（current）===")
    for r in result["techniques"]:
        if r["query_type"] == "current":
            print(f"  {r['group']:15s} {r['variant']:42s} MRR {r['mrr']:.3f}  StaleAbove {r['stale_above']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=None, help="min / recommended / max / 整数（省略時は .env の EMBED_DIM）")
    parser.add_argument("--model", default=None, help="埋め込みモデル名（省略時は .env）")
    parser.add_argument("--style", default=None, help="legacy / documented（省略時は .env）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="クロスバリデーションの分割数")
    parser.add_argument("--fake", action="store_true", help="偽の埋め込みで動作確認（結果は報告に使わない）")
    parser.add_argument("--quick", action="store_true", help="小さい候補で、速く動作確認する")
    parser.add_argument("--name", default=None, help="結果の保存フォルダ名")
    args = parser.parse_args()

    dim = config.resolve_embed_dim(config.EMBED_DIM if args.dim is None else args.dim)
    model = config.EMBED_MODEL if args.model is None else args.model.strip()
    style = config.resolve_embed_style(config.EMBED_STYLE if args.style is None else args.style)
    tag = variant_tag(model, style)

    if args.folds < 2:
        raise SystemExit("--folds は2以上にしてください。")

    paths = eval_paths(fake=args.fake, dim=dim, model=model, style=style)

    if not paths["memos_vec"].exists() or not paths["queries_vec"].exists():
        raise SystemExit(
            f"次元 {dim} / {model} / {style} のベクトル化済みデータがありません。先に実行してください:\n"
            "  python -m experiments.build_eval_dataset\n"
            f"  python -m experiments.embed_dataset --dim {dim} --model {model} --style {style}"
        )

    memos = read_json(paths["memos_vec"])
    queries = read_json(paths["queries_vec"])
    check_no_leak(memos, queries)

    name = args.name or (f"params_smoke_fake_d{dim}{tag}" if args.fake else f"params_d{dim}{tag}")
    out_dir = config.RESULTS_DIR / name

    quick = {}
    if args.quick:
        quick = dict(grids=QUICK_GRIDS, weights=weight_grid(step_tenths=2, min_sem_tenths=4),
                     bm25_grid=[(1.2, 0.75), (1.5, 0.75)], hybrid_grid=[0.5, 0.7, 1.0])

    print(f"メモ {len(memos)}件 / クエリ {len(queries)}件 / 条件: 次元={dim}, モデル={model}, 形式={style}")

    result = run_sweep(
        memos, queries, out_dir, seed=args.seed, n_folds=args.folds, **quick,
        extra_config={
            "embedding": ("FAKE hash embedding (smoke test only; do not report)" if args.fake
                          else f"{model} dim={dim} style={style}"),
            "embed_dim": dim, "embed_model": model, "embed_style": style,
        })

    print_report(result)
    print(f"\n結果の保存先: {out_dir}")


if __name__ == "__main__":
    main()
