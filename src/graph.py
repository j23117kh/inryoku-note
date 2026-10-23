import numbers
import networkx as nx
import numpy as np

from src.config import (
    KNN_K,
    EDGE_MIN_SIM,
    SUPERSEDES_EDGE_WEIGHT,
)


def build_graph(
    memos,
    vectors,
    k=None,
    min_sim=None,
    supersedes_weight=None,
):
    """
    kNN類似エッジ + 整合性検証付きsupersedesエッジ から有向グラフを作成する。
    
    【仕様・設計方針】
    - ベクトルは必須入力とし、指定されたメモとの件数・形状・有限値を厳格に検証する。
      ※ 呼び出し元でメモID順にベクトルが正しく対応付けられていることが前提となります。
    - メモIDの重複や欠損は例外（ValueError）として検出し、誤ったグラフ構築を防止する。
    - パラメータ（knn_k, edge_min_sim, supersedes_weight）は型および有効範囲を厳格に検証する。
    - 更新関係（supersedes）に循環（サイクル）が含まれる場合は DAG 判定により検出し、例外を送出する。
    - 同じ2ノード間に knn と supersedes が競合する場合、更新関係（supersedes）の重みおよび関係属性が優先（上書き）される。
    """
    # 1. パラメータの妥当性および範囲検証（numpy整数型などの柔軟な許容とboolの除外）
    knn_k = KNN_K if k is None else k
    if not isinstance(knn_k, numbers.Integral) or isinstance(knn_k, bool) or knn_k <= 0:
        raise ValueError(f"knn_k は正の整数である必要があります: {knn_k}")

    edge_min_sim = EDGE_MIN_SIM if min_sim is None else min_sim
    if (
        not isinstance(edge_min_sim, (int, float, np.integer, np.floating))
        or isinstance(edge_min_sim, bool)
        or not np.isfinite(edge_min_sim)
        or not -1 <= edge_min_sim <= 1
    ):
        raise ValueError(
            f"edge_min_sim は -1 以上 1 以下の有限値である必要があります: {edge_min_sim}"
        )

    sup_weight = SUPERSEDES_EDGE_WEIGHT if supersedes_weight is None else supersedes_weight
    if (
        not isinstance(sup_weight, (int, float, np.integer, np.floating))
        or isinstance(sup_weight, bool)
        or not np.isfinite(sup_weight)
        or sup_weight <= 0
    ):
        raise ValueError(f"supersedes_weight は正の有限値である必要があります: {sup_weight}")

    # 2. メモの存在およびIDの厳格な検証（欠損や重複を許容しない）
    if not memos:
        return nx.DiGraph()

    seen_ids = set()
    valid_memos = []
    for memo in memos:
        memo_id = memo.get("id")
        if not memo_id:
            raise ValueError("IDが空のメモが検出されました。データ不備を修正してください。")
        if memo_id in seen_ids:
            raise ValueError(f"重複したメモIDが検出されました: {memo_id}")
        seen_ids.add(memo_id)
        valid_memos.append(memo)

    ids = [memo["id"] for memo in valid_memos]
    id_set = set(ids)

    graph = nx.DiGraph()
    graph.add_nodes_from(ids)

    # 3. ベクトルの必須検証（設定ミスや未指定を即座に検知）
    if vectors is None:
        raise ValueError("グラフ構築には埋め込みベクトル（vectors）が必須です。")

    vectors_arr = np.asarray(vectors, dtype=float)

    # 形状・次元数の検証（件数参照より前に実行して IndexError を防ぐ）
    if vectors_arr.ndim != 2 or vectors_arr.shape[1] == 0:
        raise ValueError(f"不正なベクトルの形状です: {vectors_arr.shape}")

    # 件数一致の検証（※呼び出し元で valid_memos と同じ順序でベクトルが並んでいることが前提）
    if vectors_arr.shape[0] != len(valid_memos):
        raise ValueError(
            f"メモの件数({len(valid_memos)})とベクトルの件数({vectors_arr.shape[0]})が一致しません。"
        )

    # NaN / 無限大の検出
    if not np.isfinite(vectors_arr).all():
        raise ValueError("ベクトルに NaN または無限大（Inf）が含まれています。")

    # 4. 類似エッジ（kNN）の構築
    if len(valid_memos) > 1:
        norms = np.linalg.norm(vectors_arr, axis=1, keepdims=True)
        valid_vector_mask = (norms > 0).flatten()

        normalized_vectors = np.divide(
            vectors_arr,
            norms,
            out=np.zeros_like(vectors_arr, dtype=float),
            where=valid_vector_mask[:, np.newaxis],
        )

        similarity = normalized_vectors @ normalized_vectors.T
        np.fill_diagonal(similarity, -1.0)

        actual_k = min(knn_k, len(valid_memos) - 1)

        for i in range(len(valid_memos)):
            if not valid_vector_mask[i]:
                continue

            nearest = np.argsort(-similarity[i])[:actual_k]
            for j in nearest:
                if not valid_vector_mask[j]:
                    continue

                score = similarity[i, j]
                if score >= edge_min_sim:
                    weight = float(score)
                    graph.add_edge(ids[i], ids[j], weight=weight, relation="knn")
                    graph.add_edge(ids[j], ids[i], weight=weight, relation="knn")

    # 5. 更新関係（supersedes）の検証とエッジ追加
    supersede_digraph = nx.DiGraph()
    supersede_digraph.add_nodes_from(ids)

    for memo in valid_memos:
        new_id = memo["id"]

        for old_id in memo.get("supersedes", []):
            if old_id not in id_set or old_id == new_id:
                continue
            if old_id not in id_set:
                continue

            supersede_digraph.add_edge(old_id, new_id)

            # メイングラフへエッジ追加
            # ※ 同一ペアにknnエッジが存在する場合、上書きにより supersedes が優先されます。
            graph.add_edge(
                old_id,
                new_id,
                weight=float(sup_weight),
                relation="supersedes",
            )

    # 6. DAG判定による効率的な更新履歴の循環検出
    if not nx.is_directed_acyclic_graph(supersede_digraph):
        raise ValueError(
            "メモの更新履歴に循環（サイクル）が検出されました。データ（supersedes関係）を修正してください。"
        )

    return graph