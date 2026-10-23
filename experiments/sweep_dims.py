"""
複数の次元を、まとめて「ベクトル化 → 評価 → 比較表作成」まで一括実行するオーケストレーションスクリプト。

使用例:
    # 思考の系譜・ラベルなし評価（系譜特化モード / 既定）
    python -m experiments.sweep_dims --dims recommended_all --mode lineage

    # 総合検索性能の比較モード
    python -m experiments.sweep_dims --dims recommended_all --mode standard

    # 最小 + 推奨すべてを対象にする場合
    python -m experiments.sweep_dims --dims all --mode lineage

    # APIなしで流れだけ確認（動作確認用）
    python -m experiments.sweep_dims --dims recommended_all --fake --mode lineage
"""

import argparse
import json
import subprocess
import sys

from src.config import EVAL_DIR, resolve_embed_dims

# 評価データがまだ無いときの目安（メモ330 + クエリ188）
FALLBACK_CALLS_PER_DIM = 518


def calls_per_dim():
    """
    1次元あたりのAPI呼び出し回数（メモ数 + クエリ数）。
    評価データがあればその件数から数え、無ければ既定の目安を返す。
    """
    try:
        with open(EVAL_DIR / "memos_eval.json", "r", encoding="utf-8") as f:
            n_memos = len(json.load(f))
        with open(EVAL_DIR / "queries_eval.json", "r", encoding="utf-8") as f:
            n_queries = len(json.load(f))
    except (OSError, ValueError):
        return FALLBACK_CALLS_PER_DIM

    return n_memos + n_queries


def build_commands(dims, mode="lineage", fake=False, distractors=0, seed=42, skip_embed=False, sleep=None):
    """実行するコマンドのリスト [(説明, [引数...]), ...] を作る。"""
    py = sys.executable
    cmds = []

    for d in dims:
        # 1. ベクトル化ステップ（共通）
        if not skip_embed:
            c = [py, "-m", "experiments.embed_dataset", "--dim", str(d)]
            if fake:
                c.append("--fake")
            if sleep is not None:
                c += ["--sleep", str(sleep)]
            cmds.append((f"[{d}次元] ベクトル化", c))

        # 2. 評価ステップ（モードに応じて切り替え）
        if mode == "lineage":
            c = [py, "-m", "experiments.evaluate_lineage", "--dim", str(d), "--seed", str(seed)]
            if fake:
                c.append("--fake")
            cmds.append((f"[{d}次元] 系譜特化評価 (evaluate_lineage)", c))
        elif mode == "standard":
            c = [py, "-m", "experiments.run_eval", "--dim", str(d), "--seed", str(seed)]
            if fake:
                c.append("--fake")
            elif distractors:
                c += ["--distractors", str(distractors)]
            cmds.append((f"[{d}次元] 総合検索評価 (run_eval)", c))
        else:
            raise ValueError(f"未知の実行モードです: {mode}")

    # 3. 比較表・グラフ作成ステップ（系譜モードの場合のみcompare_dimsに対応）
    if mode == "lineage":
        c = [py, "-m", "experiments.compare_dims"]
        if fake:
            c += ["--prefix", "smoke_fake_lineage"]
        else:
            c += ["--prefix", "lineage_eval_summary"]
        cmds.append(("比較表・グラフの作成 (compare_dims)", c))

    return cmds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="recommended_all",
                        help="recommended_all / all / min / recommended / max / 整数"
                             "（カンマ区切りで組み合わせ可）。既定: recommended_all")
    parser.add_argument("--mode", choices=["lineage", "standard"], default="lineage",
                        help="評価モード: lineage (系譜特化・ラベルなし評価) または standard (総合検索比較)")
    parser.add_argument("--fake", action="store_true",
                        help="偽の埋め込みで動作確認（結果は報告に使わない）")
    parser.add_argument("--distractors", type=int, default=0,
                        help="背景ノイズ(Wikipedia, 256次元)の件数（standardモード専用）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-embed", action="store_true",
                        help="ベクトル化済みのとき、評価ステップだけやり直す")
    parser.add_argument("--sleep", type=float, default=None, help="API呼び出し間隔（秒）")
    parser.add_argument("--dry-run", action="store_true",
                        help="実行せず、コマンドとAPI呼び出し回数の目安だけ表示")
    args = parser.parse_args()

    dims = resolve_embed_dims(args.dims)

    cmds = build_commands(
        dims=dims,
        mode=args.mode,
        fake=args.fake,
        distractors=args.distractors,
        seed=args.seed,
        skip_embed=args.skip_embed,
        sleep=args.sleep,
    )

    print(f"評価モード: {args.mode}")
    print(f"対象の次元: {dims}")

    if args.mode == "standard":
        print("情報: standard モードのため、一括の compare_dims はスキップされます（各結果は results/ に個別保存されます）。")

    if not args.fake and not args.skip_embed:
        print(f"API呼び出しの目安: 約{calls_per_dim() * len(dims):,}回"
              f"（1次元あたり {calls_per_dim()}回）")

    if args.distractors and any(d != 256 for d in dims) and not args.fake:
        print("注意: 背景ノイズ(Wikipedia)は256次元のため、256以外では使われません。")

    for label, c in cmds:
        print(f"\n=== {label} ===")
        print(" ".join(c))

        if args.dry_run:
            continue

        r = subprocess.run(c)

        if r.returncode != 0:
            raise SystemExit(
                f"失敗しました: {label}（終了コード {r.returncode}）。\n"
                "同じコマンドを再実行すると、キャッシュにより続きから再開できます。"
            )

    if args.dry_run:
        print("\n（--dry-run のため、何も実行していません）")
    else:
        print("\n すべてのパイプラインが正常に完了しました！")


if __name__ == "__main__":
    main()