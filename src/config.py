import os
from pathlib import Path

from dotenv import load_dotenv


# =========================
# プロジェクトルート
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# =========================
# API
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# =========================
# Geminiモデル
# =========================

# 既定の埋め込みモデル。.env の EMBED_MODEL で切り替えられる（比較実験用）
DEFAULT_EMBED_MODEL = "gemini-embedding-2"
EMBED_MODEL = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL
GEN_MODEL = "gemini-3.8-flash"


# =========================
# 埋め込みの入力形式（スタイル）
# =========================
# legacy      従来の方式: task_type を指定し、文書だけ "title: ... | text: ..." にする。
#             クエリは素の文字列のまま。（既定。既存の保存済みデータと互換）
# documented  Google公式ドキュメントの推奨形式。
#             gemini-embedding-2: task_type は使わず、プロンプトに形式を付ける
#               クエリ: "task: search result | query: {本文}"
#               文書  : "title: {タイトル or none} | text: {本文}"
#             gemini-embedding-001 など: task_type を指定する（本文は素のまま）
# 出典: https://ai.google.dev/gemini-api/docs/embeddings

EMBED_STYLES = ("legacy", "documented")


def resolve_embed_style(value):
    """スタイル名を検証して返す。不正な値はエラー（黙って既定に戻さない）。"""

    key = str(value).strip().lower()

    if key not in EMBED_STYLES:
        raise ValueError(
            f"埋め込みスタイル '{value}' は不正です。{list(EMBED_STYLES)} のどれかを指定してください。"
        )

    return key


EMBED_STYLE = resolve_embed_style(os.getenv("EMBED_STYLE", "legacy"))


# =========================
# 埋め込みの次元数
# =========================
# Gemini API（gemini-embedding-2）の仕様:
#   出力次元: 128〜3072 の範囲で指定可能（デフォルトは3072）
#   推奨値  : 768 / 1536 / 3072
# 出典: https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2-preview
# 仕様が変わった場合は、ここの3つの値だけ直せばよい。

EMBED_DIM_MIN = 128
EMBED_DIM_RECOMMENDED = 768          # 推奨値のうち最小。他の推奨値は 1536, 3072
EMBED_DIM_RECOMMENDED_ALL = (768, 1536, 3072)
EMBED_DIM_MAX = 3072

EMBED_DIM_PRESETS = {
    "min": EMBED_DIM_MIN,
    "recommended": EMBED_DIM_RECOMMENDED,
    "max": EMBED_DIM_MAX,
}


# 複数の次元をまとめて指す名前（評価の一括実行 sweep_dims 用）
EMBED_DIM_MULTI_PRESETS = {
    # 公式の推奨値すべて: 768, 1536, 3072
    "recommended_all": EMBED_DIM_RECOMMENDED_ALL,
    # 最小 + 推奨すべて（最大3072は推奨値に含まれる）: 128, 768, 1536, 3072
    "all": tuple(sorted({EMBED_DIM_MIN, *EMBED_DIM_RECOMMENDED_ALL, EMBED_DIM_MAX})),
}


def resolve_embed_dim(value):
    """
    次元数を整数に解決する。

    value には次のどれかを指定できる:
      "min" / "recommended" / "max"   プリセット名（大文字小文字は問わない）
      128〜3072 の整数、またはその文字列（例: 256, "1536"）

    範囲外・不正な値は ValueError にする（黙って丸めない）。
    """

    if isinstance(value, str):

        key = value.strip().lower()

        if key in EMBED_DIM_PRESETS:
            return EMBED_DIM_PRESETS[key]

        if key in EMBED_DIM_MULTI_PRESETS or "," in key:
            raise ValueError(
                f"'{value}' は複数の次元を指します。ここでは1つの次元だけ指定できます。"
                "複数をまとめて評価するには experiments.sweep_dims の --dims を使ってください。"
            )

        try:
            value = int(key)
        except ValueError:
            raise ValueError(
                f"次元数 '{value}' を解釈できません。"
                f"{sorted(EMBED_DIM_PRESETS)} または "
                f"{EMBED_DIM_MIN}〜{EMBED_DIM_MAX} の整数を指定してください。"
            )

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"次元数は整数かプリセット名で指定してください: {value!r}")

    if not EMBED_DIM_MIN <= value <= EMBED_DIM_MAX:
        raise ValueError(
            f"次元数 {value} は範囲外です"
            f"（{EMBED_DIM_MIN}〜{EMBED_DIM_MAX}）。"
        )

    return value


def resolve_embed_dims(value):
    """
    複数の次元を、重複なしの昇順リストに解決する。

    value には次のものを指定できる（カンマ区切りで組み合わせ可）:
      "recommended_all"       公式の推奨値すべて (768, 1536, 3072)
      "all"                   最小 + 推奨すべて (128, 768, 1536, 3072)
      "min" / "recommended" / "max" / 128〜3072 の整数
      例: "min,recommended_all"  ->  [128, 768, 1536, 3072]
          "256,768"              ->  [256, 768]
    リスト・タプルも可。1つでも不正ならエラー。
    """

    if isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        tokens = [t for t in str(value).split(",") if t.strip()]

    dims = set()

    for tok in tokens:

        key = tok.strip().lower() if isinstance(tok, str) else None

        if key in EMBED_DIM_MULTI_PRESETS:
            dims.update(EMBED_DIM_MULTI_PRESETS[key])
        else:
            dims.add(resolve_embed_dim(tok))

    if not dims:
        raise ValueError("次元が1つも指定されていません。")

    return sorted(dims)


# .env の EMBED_DIM で切り替える（未指定なら 256: 既存データと同じ）
#   EMBED_DIM=min | recommended | max | 256 | 1536 ...
# 注意: 次元を変えたら、保存済みメモを再埋め込みすること
#   python -m tools.reembed_memos --dim recommended
EMBED_DIM = resolve_embed_dim(os.getenv("EMBED_DIM", "256"))


# =========================
# データファイル
# =========================

DATA_DIR = BASE_DIR / "data" / "processed"

RAW_DIR = BASE_DIR / "data" / "raw"

# 初回起動時に DATA_FILE が無ければ、このファイルからコピーされる
LEGACY_DATA_FILE = RAW_DIR / "memos_wikipedia_951.json"
DATA_FILE = DATA_DIR / "memos_data_v3.json"

# 評価データ（experiments/ のパイプラインが生成する）
EVAL_DIR = BASE_DIR / "data" / "eval"
RESULTS_DIR = BASE_DIR / "results"


# =========================
# UI
# =========================

STATIC_DIR = BASE_DIR / "static"

UI_FILE = STATIC_DIR / "index.html"


# =========================
# グラフ・推薦パラメータ
# =========================

KNN_K = 5

EDGE_MIN_SIM = 0.5

PPR_ALPHA = 0.85


# =========================
# 更新判定
# =========================

SUPERSEDE_CANDIDATES = 5

SUPERSEDE_MIN_SIM = 0.5

SUPERSEDES_EDGE_WEIGHT = 1.0

STALE_PENALTY = 0.5


# =========================
# ASK
# =========================

ASK_TOP_K = 5

ASK_MIN_SCORE = 0.25