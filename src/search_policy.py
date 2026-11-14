"""Webアプリとオフライン評価が共有する検索UI/APIの既定条件。

検索スコア計算は src.recommender.search_core が唯一の実装。
検索画面(/api/gravity)の旧 ``hybrid`` は仕様上 Full の別名。
BM25+Semantic の実験方式 ``hybrid_bm25_sem`` とは異なる。
"""

APP_DEFAULT_THRESHOLD_PERCENT = 30.0
APP_DEFAULT_THRESHOLD = APP_DEFAULT_THRESHOLD_PERCENT / 100.0
APP_DEFAULT_TOP_K = 10
GRAVITY_DEFAULT_MODE = "semantic_lineage"
GRAVITY_MODE_ALIASES = {"ai": "semantic", "hybrid": "full"}


def resolve_gravity_mode(mode):
    """既存のWebクライアントのモード名を変更せず実際の検索方式へ変換。"""
    return GRAVITY_MODE_ALIASES.get(mode, mode)
