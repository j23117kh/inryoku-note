import numpy as np

from src import config as app_config
from src.config import (
    GEMINI_API_KEY,
    resolve_embed_dim,
    resolve_embed_style,
)

_client = None


def _get_client():
    """
    Geminiクライアントを初回利用時に作る（遅延初期化）。

    import時にAPIキーを要求すると、オフライン評価やテストが
    できなくなるため、実際にAPIを呼ぶ時点でチェックする。
    """

    global _client

    if _client is None:

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY が設定されていません。"
                ".env を確認してください（.env.example 参照）。"
            )

        from google import genai

        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def build_request(text, task_type, title, model, style):
    """
    API に渡す本文と、task_type を指定するかどうかを決める（純粋関数）。

    戻り値: (本文, task_type を使うか)

    legacy      : 従来どおり。task_type を使い、文書にタイトルがあれば "title: | text:" 形式にする。
    documented  : 公式ドキュメントの推奨形式（モデルごとに異なる）。
    """

    is_doc = task_type == "RETRIEVAL_DOCUMENT"

    if style == "legacy":

        if title and is_doc:
            return f"title: {title} | text: {text}", True

        return text, True

    if style == "documented":

        if str(model).startswith("gemini-embedding-2"):

            if is_doc:
                return f"title: {title or 'none'} | text: {text}", False

            if task_type == "RETRIEVAL_QUERY":
                return f"task: search result | query: {text}", False

            return text, False

        # gemini-embedding-001 など: task_type パラメータを使う
        return text, True

    raise ValueError(f"unknown style: {style}")


def get_vector(
    text,
    task_type="RETRIEVAL_DOCUMENT",
    title=None,
    dim=None,
    model=None,
    style=None,
):
    """
    テキストをGemini Embeddingでベクトル化する（L2正規化済み）。
    失敗時は None を返す。

    dim  : 出力次元（数値 or "min"/"recommended"/"max"）。
           省略時は config.EMBED_DIM。範囲外は ValueError（APIを呼ぶ前に検出）。
    model: 埋め込みモデル名。省略時は config.EMBED_MODEL。
    style: 入力形式 "legacy" / "documented"。省略時は config.EMBED_STYLE。
    """

    dim = resolve_embed_dim(
        app_config.EMBED_DIM if dim is None else dim
    )

    model = app_config.EMBED_MODEL if model is None else str(model).strip()

    style = resolve_embed_style(
        app_config.EMBED_STYLE if style is None else style
    )

    from google.genai import types

    final_text, use_task_type = build_request(
        text, task_type, title, model, style
    )

    try:
        client = _get_client()

        try:
            if use_task_type:
                config = types.EmbedContentConfig(
                    output_dimensionality=dim,
                    task_type=task_type,
                )
            else:
                config = types.EmbedContentConfig(
                    output_dimensionality=dim,
                )
            response = client.models.embed_content(
                model=model,
                contents=final_text,
                config=config,
            )

        except Exception:
            # task_type指定が失敗した場合は、指定なしで再試行
            config = types.EmbedContentConfig(
                output_dimensionality=dim
            )
            response = client.models.embed_content(
                model=model,
                contents=final_text,
                config=config,
            )

        vector = np.array(
            response.embeddings[0].values,
            dtype=float,
        )

        if len(vector) != dim:
            print(
                f"ベクトル化エラー: 要求した次元 {dim} に対し、"
                f"{len(vector)} 次元が返りました。"
            )
            return None

        # 3072未満に切り詰めたベクトルは正規化されていないため、L2正規化する
        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return vector

    except Exception as e:
        print(f"ベクトル化エラー: {e}")
        return None
