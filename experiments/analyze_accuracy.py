"""
AIによる更新判定 (memo_analyzer.analyze_memo) の精度評価スクリプト
（リトライ機構・エラーハンドリング・予測値ベースの状態伝搬選択対応版）

実行（プロジェクトのルートで）:
    python -m experiments.analyze_accuracy --dim 256
    python -m experiments.analyze_accuracy --dim 256 --propagate pred   # 予測値に基づく状態伝搬で評価

出力 (results/<name>/):
    supersedes_accuracy.csv        精度のまとめ（Precision / Recall / F1 / 候補再現率 / 日付一致率）
    supersedes_predictions.csv     メモごとの予測と正解
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# プロジェクトルートのパス追加
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import eval_paths, read_json, variant_tag
from experiments.run_eval import write_csv
from src import config


def score_supersedes(rows):
    """
    rows: [{"pred": set, "gold": set}, ...]
    戻り値: precision / recall / f1（リンク単位。全メモ合算）
    """
    tp = sum(len(r["pred"] & r["gold"]) for r in rows)
    fp = sum(len(r["pred"] - r["gold"]) for r in rows)
    fn = sum(len(r["gold"] - r["pred"]) for r in rows)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    if tp == 0:
        f1 = 0.0 if (fp + fn) else float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def select_candidates(memo, earlier, top_k, min_sim):
    """main.py の /api/save と同じ候補選択（コサイン類似度の上位、閾値以上）。"""
    if not earlier:
        return []

    sims = np.array([m["vector"] for m in earlier]) @ np.array(memo["vector"])
    order = np.argsort(-sims)[:top_k]

    return [earlier[i] for i in order if sims[i] >= min_sim]


def run(memos, analyze_memo_func, top_k, min_sim, limit=None, sleep=0.0, propagate_mode="gold"):
    """
    propagate_mode:
      - "gold": 正解ラベルで状態を進める（上限値・バイアスあり）
      - "pred": AIの予測結果で状態を進める（実運用に近いエラー波及評価）
    """
    ordered = sorted(memos, key=lambda m: m["created_at"])

    seen = []
    superseded = set()
    results = []

    def analyze_memo_safe(content, date_str, cands, max_retries=5):
        """レート制限や一時エラー対策付きの安全なLLM解析"""
        for attempt in range(max_retries):
            try:
                event_date, sup = analyze_memo_func(content, date_str, cands)
                if event_date is not None or sup is not None:
                    time.sleep(0.5)  # API負荷軽減ウェイト
                    return event_date, sup
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = (2 ** attempt) + 3.0
                    print(f"  [レート制限 (429)] {wait_time:.1f}秒待機して再試行します... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"  [WARN] LLM解析エラー: {e} (再試行 {attempt+1}/{max_retries})")
                    time.sleep(2.0)
            time.sleep(1.0)
        # すべて失敗した場合は空を返す
        return None, []

    for index, memo in enumerate(ordered, start=1):
        earlier = [m for m in seen if m["id"] not in superseded]
        cands = select_candidates(memo, earlier, top_k, min_sim)
        gold = set(memo.get("supersedes", []))

        if limit is None or len(results) < limit:
            print(f"[memo {index}/{len(ordered)} analyzing {memo['id']}")
            event_date, sup = analyze_memo_safe(
                memo["content"], memo["created_at"][:10], cands
            )

            pred_set = {s["id"] for s in (sup or [])}

            results.append({
                "id": memo["id"],
                "kind": memo.get("kind"),
                "pred": pred_set,
                "gold": gold,
                "gold_in_candidates": (
                    gold <= {c["id"] for c in cands} if gold else None
                ),
                "pred_event_date": event_date,
                "gold_event_date": memo.get("event_date"),
            })

            if sleep:
                time.sleep(sleep)
        else:
            pred_set = set()

        # 状態伝搬のモード切り替え
        if propagate_mode == "pred":
            superseded |= pred_set
        else:
            superseded |= gold

        seen.append(memo)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="評価するメモ数の上限")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--dim", default=256, type=int, help="使うベクトルの次元数")
    parser.add_argument("--model", default=None, help="埋め込みモデル名（省略時は .env）")
    parser.add_argument("--style", default=None, help="legacy / documented（省略時は .env）")
    parser.add_argument("--name", default=None)
    parser.add_argument("--propagate", choices=["gold", "pred"], default="gold",
                        help="状態伝搬の方法: gold (正解ラベルベース) または pred (予測ベース)")
    args = parser.parse_args()

    dim = config.resolve_embed_dim(
        config.EMBED_DIM if args.dim is None else args.dim
    )
    model = config.EMBED_MODEL if args.model is None else args.model.strip()
    style = config.resolve_embed_style(
        config.EMBED_STYLE if args.style is None else args.style
    )

    from src.memo_analyzer import analyze_memo

    paths = eval_paths(dim=dim, model=model, style=style)
    if not Path(paths["memos_vec"]).exists():
        raise FileNotFoundError(f"ベクトル済みメモファイルが見つかりません: {paths['memos_vec']}。事前に埋め込みを実行してください。")

    memos = read_json(paths["memos_vec"])

    print(f"評価開始: 伝搬モード={args.propagate}, 対象メモ数={len(memos) if args.limit is None else min(len(memos), args.limit)}")

    results = run(
        memos, analyze_memo,
        config.SUPERSEDE_CANDIDATES, config.SUPERSEDE_MIN_SIM,
        args.limit, args.sleep, args.propagate,
    )

    link = score_supersedes(results)

    with_gold = [r for r in results if r["gold"]]
    cand_recall = (
        float(np.mean([r["gold_in_candidates"] for r in with_gold]))
        if with_gold else float("nan")
    )

    dated = [r for r in results if r["gold_event_date"]]
    date_acc = (
        float(np.mean([r["pred_event_date"] == r["gold_event_date"] for r in dated]))
        if dated else float("nan")
    )

    # 出力先ディレクトリ名を一貫させる
    suffix = f"_prop_{args.propagate}" if args.propagate != "gold" else ""
    out_dir = config.RESULTS_DIR / (args.name or f"main_d{dim}{variant_tag(model, style)}{suffix}")

    write_csv(out_dir / "supersedes_accuracy.csv", [{
        "n_memos": len(results),
        "n_gold_links": link["tp"] + link["fn"],
        **link,
        "candidate_recall": cand_recall,
        "event_date_accuracy": date_acc,
        "analyzer_model": config.GEN_MODEL,
        "propagate_mode": args.propagate,
    }])

    write_csv(out_dir / "supersedes_predictions.csv", [{
        "id": r["id"], "kind": r["kind"],
        "pred": ";".join(sorted(r["pred"])),
        "gold": ";".join(sorted(r["gold"])),
        "pred_event_date": r["pred_event_date"],
        "gold_event_date": r["gold_event_date"],
    } for r in results])

    print(f"\n--- 評価結果 ({args.propagate} 伝搬) ---")
    print(f"links: P={link['precision']:.3f} R={link['recall']:.3f} F1={link['f1']:.3f}")
    print(f"候補再現率={cand_recall:.3f} / event_date一致率={date_acc:.3f}")
    print(f"保存先: {out_dir}")


if __name__ == "__main__":
    main()