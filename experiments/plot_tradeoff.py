"""
sweep_stale_penalty.csv のデータから、
ペナルティ係数に応じた「MRR（精度）」と「StaleAbove（古さ）」のトレードオフを2軸グラフで描画する。

使い方:
    python -m experiments.plot_tradeoff --dir results/main_d768
"""

import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_sweep_csv(csv_path):
    """sweep_stale_penalty.csv を読み込んでデータを整理する（run_eval.py の出力）"""
    data = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "stale_penalty": float(row["stale_penalty"]),
                "query_type": row["query_type"],
                "mrr": float(row["mrr"]),
                "stale_above": float(row["stale_above"]),
            })
    return data


def plot_tradeoff(csv_path, out_path):
    data = load_sweep_csv(csv_path)

    # 実用クエリである "current" を抽出
    current_data = [d for d in data if d["query_type"] == "current"]
    if not current_data:
        raise ValueError(f"エラー: 指定されたCSVに query_type='current' のデータが見つかりません: {csv_path}")
    
    # 昇順（0.1 -> 1.0）にソート
    current_data.sort(key=lambda x: x["stale_penalty"])

    penalties = [d["stale_penalty"] for d in current_data]
    mrrs = [d["mrr"] for d in current_data]
    stale_aboves = [d["stale_above"] for d in current_data]

    # プロットの作成（2軸グラフ）
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

    # 左軸: MRR (検索精度 - 高いほうが良い)
    color = "#1f77b4"
    ax1.set_xlabel("Stale Penalty Coefficient ($\alpha$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MRR (Higher is better)", color=color, fontsize=11, fontweight="bold")
    line1 = ax1.plot(penalties, mrrs, color=color, marker="o", linewidth=2.2, label="MRR (Search Accuracy)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1.05)
    
    # X軸を反転: 左側にペナルティなし（1.0）、右側に向かって強いペナルティ（0.1）とする
    ax1.invert_xaxis()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 右軸: StaleAbove (古い情報の混入度 - 低いほうが良い)
    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.set_ylabel("StaleAbove (Lower is better)", color=color, fontsize=11, fontweight="bold")
    line2 = ax2.plot(penalties, stale_aboves, color=color, marker="s", linestyle="--", linewidth=2.2, label="StaleAbove (Stale Rate)")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(-0.02, 1.02)
    # 右軸のグリッドは左軸と重なると見づらくなるため非表示にするか控えめにする

    # スイートスポット（最適動作点）の動的ハイライト（例: 0.5 付近）
    target_sweet_spot = 0.5
    if penalties:
        best_x = min(penalties, key=lambda x: abs(x - target_sweet_spot))
        idx = penalties.index(best_x)
        
        ax1.axvline(x=best_x, color="gray", linestyle=":", alpha=0.8, linewidth=1.5)
        
        # テキスト位置がグラフの上下限からはみ出さないように安全にオフセット計算
        y_pos = mrrs[idx]
        text_y = max(0.2, min(0.85, y_pos - 0.2))  # 常に適切な高さに収める
        
        ax1.annotate(
            f"Sweet Spot ($\alpha$={best_x})\n(Optimal Balance)",
            xy=(best_x, y_pos),
            xytext=(best_x + 0.12, text_y),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

    # 2つの軸の凡例をまとめて綺麗に配置する処理
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=9, frameon=True)

    plt.title("Trade-off between Search Accuracy (MRR) and Stale Information Penalty", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[完了] 論文品質のトレードオフグラフを保存しました: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Stale Penaltyの感度分析からトレードオフグラフを描画する")
    parser.add_argument("--dir", default="results/main_d768", help="run_eval.py の結果が格納されているフォルダパス")
    args = parser.parse_args()

    csv_path = Path(args.dir) / "sweep_stale_penalty.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"エラー: 指定されたパスに 'sweep_stale_penalty.csv' が見つかりません: {csv_path}\n"
            "先に run_eval.py を実行して該当CSVを出力してください。"
        )

    out_path = Path(args.dir) / "tradeoff_stale_penalty.png"
    plot_tradeoff(csv_path, out_path)


if __name__ == "__main__":
    main()