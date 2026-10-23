"""
検索評価の指標。

入力はすべて「文書ID -> スコア」の辞書（全候補分）。
スコアが同点の文書がある場合は、順位を「期待順位」で扱う
（BM25のようにスコア0が大量に並ぶ手法が、並び順の偶然で
有利・不利にならないようにするため）。
"""

import math

import numpy as np


def expected_rank(scores, target_id):
    """
    target_id の順位（1始まり）。同点の他文書があれば、その半数ぶんを加算する。
    target_id が scores に無ければ None。
    """

    if target_id not in scores:
        return None

    target = scores[target_id]

    greater = sum(
        1 for k, v in scores.items()
        if k != target_id and v > target
    )

    ties = sum(
        1 for k, v in scores.items()
        if k != target_id and v == target
    )

    return 1.0 + greater + ties / 2.0


def reciprocal_rank(scores, relevant_id):
    rank = expected_rank(scores, relevant_id)
    return 0.0 if rank is None else 1.0 / rank


def hit_at_k(scores, relevant_id, k):
    rank = expected_rank(scores, relevant_id)
    return 0.0 if rank is None else float(rank <= k)


def ndcg_at_k(scores, relevant_id, k):
    """正解が1件のときの nDCG@k（理想DCG=1）。"""

    rank = expected_rank(scores, relevant_id)

    if rank is None or rank > k:
        return 0.0

    return 1.0 / math.log2(rank + 1.0)


def stale_above(scores, relevant_id, stale_ids):
    """
    「更新済み（古い版）のメモが、正解（最新版）より上位に来る確率」。
    古い版が複数あるときは、どれか1つでも上位に来る確率。
    同点は 0.5 として数える。これが小さいほど良い。
    """

    if relevant_id not in scores:
        return 0.0

    target = scores[relevant_id]

    not_above = 1.0

    for sid in stale_ids:

        if sid not in scores:
            continue

        if scores[sid] > target:
            p = 1.0
        elif scores[sid] == target:
            p = 0.5
        else:
            p = 0.0

        not_above *= 1.0 - p

    return 1.0 - not_above


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """平均のブートストラップ信頼区間 (low, high)。"""

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)

    idx = rng.integers(0, values.size, size=(n_boot, values.size))

    means = values[idx].mean(axis=1)

    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_bootstrap_diff(a, b, n_boot=2000, seed=0, alpha=0.05):
    """
    同じクエリ集合での (a - b) の平均差と、その信頼区間。
    戻り値: (mean_diff, low, high)
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # 比較する配列の形状が一致することを確認する（対応関係の崩れを防ぐ）
    if a.shape != b.shape:
        raise ValueError(
            f"比較する配列の形状が一致しません: a={a.shape}, b={b.shape}"
        )

    diff = a - b

    low, high = bootstrap_ci(diff, n_boot, seed, alpha)

    return (float(diff.mean()) if diff.size else float("nan"), low, high)