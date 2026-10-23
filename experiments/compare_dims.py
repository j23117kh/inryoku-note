"""
次元数ごとの評価結果を、1つの表にまとめる（API不要・完全整合性対応版）。

実行:
    python -m experiments.compare_dims
    python -m experiments.compare_dims --prefix lineage_eval_summary

results/lineage_eval_summary*_dim*.csv を読み、
results/dim_comparison_lineage.csv（表）と dim_comparison_lineage.png（図）にまとめる。
"""

import argparse
import csv
import re
from pathlib import Path

from experiments.run_eval import write_csv
from src.config import EMBED_DIM_RECOMMENDED_ALL, RESULTS_DIR

# evaluate_lineage.py が出力する手法名ベース（順序制御用）
METHODS = [
    "semantic",
    "graph",
    "full",
    "recency_demote",
    "old_first",
    "oracle",
    # lineage_tau* は動的に追加されます
]

KEYS = ["mrr", "p@1", "recall@10", "ndcg@10", "turning@5", "root@5"]


def collect(results_dir=RESULTS_DIR, prefix="lineage_eval_summary"):
    rows = []

    # lineage_eval_summary_v3_dim768.csv のようなファイルを探す
    for f in results_dir.glob(f"{prefix}*_dim*.csv"):
        # ファイル名から次元数を抽出 (例: _dim768)
        m = re.search(r"_dim(\d+)", f.name)
        if not m:
            continue
        dim = int(m.group(1))

        with open(f, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                # 数値フィールドの変換（空文字や不正値は NaN に）
                metrics = {}
                for k in KEYS:
                    val = r.get(k, "")
                    try:
                        metrics[k] = float(val) if val != "" else float("nan")
                    except ValueError:
                        metrics[k] = float("nan")

                rows.append({
                    "dim": dim,
                    "recommended": dim in EMBED_DIM_RECOMMENDED_ALL,
                    "method": r["method"],
                    **metrics,
                })

    # 動的に手法リストを構築（定義済み METHODS を優先し、系譜 lineage 手法等を後ろに統合）
    found_methods = {r["method"] for r in rows}
    
    # 既存の METHODS に含まれないもの（lineage_tau* 等）を抽出して追加
    dynamic_methods = sorted([m for m in found_methods if m not in METHODS])
    ordered_methods = [m for m in METHODS if m in found_methods] + dynamic_methods
          
    order = {m: i for i, m in enumerate(ordered_methods)}
    rows.sort(key=lambda r: (r["dim"], order.get(r["method"], len(order)), r["method"]))

    return rows


def make_figure(path, rows):
    """次元ごとの MRR / nDCG@10（手法別の折れ線グラフ）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = sorted({r["dim"] for r in rows})
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, key, title in (
        (axes[0], "mrr", "MRR (higher is better)"),
        (axes[1], "ndcg@10", "nDCG@10 (higher is better)"),
    ):
        for method in methods:
            ys = []
            for d in dims:
                v = [r[key] for r in rows if r["dim"] == d and r["method"] == method]
                ys.append(v[0] if v else float("nan"))
            ax.plot(range(len(dims)), ys, marker="o", label=method)

        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(
            [f"{d}*" if d in EMBED_DIM_RECOMMENDED_ALL else str(d) for d in dims]
        )
        ax.set_xlabel("Embedding Dimension (* = Recommended)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="lineage_eval_summary",
                        help="読む結果ファイルの接頭辞")
    args = parser.parse_args()

    rows = collect(RESULTS_DIR, args.prefix)

    if not rows:
        raise SystemExit(
            f"results/{args.prefix}*_dim*.csv が見つかりません。\n"
            "先に evaluate_lineage を各次元で実行してください。\n"
            "例: python -m experiments.evaluate_lineage --dim 256"
        )

    stem = "dim_comparison_lineage"

    write_csv(RESULTS_DIR / f"{stem}.csv", rows)
    make_figure(RESULTS_DIR / f"{stem}.png", rows)

    dims = sorted({r["dim"] for r in rows})
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    print("\n--- MRR 比較表（行 = 手法 / 列 = 次元数 [* = 公式推奨値]） ---")
    header_width = 32
    col_width = 10
    print(f"{'method':{header_width}s}" + "".join(
        f"{(str(d) + ('*' if d in EMBED_DIM_RECOMMENDED_ALL else '')):>{col_width}s}" for d in dims))
    print("-" * (header_width + col_width * len(dims)))

    for method in methods:
        cells = []
        for d in dims:
            v = [r["mrr"] for r in rows if r["dim"] == d and r["method"] == method]
            val = v[0] if v else float("nan")
            if val != val:  # NaN の判定
                cells.append(f"{'-':>{col_width}s}")
            else:
                cells.append(f"{val:{col_width}.3f}")
        print(f"{method:{header_width}s}" + "".join(cells))

    print(f"\n保存完了:\n  - CSV: {RESULTS_DIR / (stem + '.csv')}\n  - PNG: {RESULTS_DIR / (stem + '.png')}")


if __name__ == "__main__":
    main()