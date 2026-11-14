import os
import numpy as np

from src import config as app_config
from src.config import (
    GEMINI_API_KEY,
    resolve_embed_dim,
    resolve_embed_style,
)
from src import embedding_cache

_client = None


def _get_client():
    """Gemini client is created only if a cache miss permits live API calls."""
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
    """Same request formatting as the original implementation."""
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
        return text, True
    raise ValueError(f"unknown style: {style}")


def _is_task_type_unsupported(error):
    """Only fall back without task_type for a 400 explicitly about task_type.

    Retry is NEVER performed for 429 quota exhaustion, 503, timeouts, etc.
    """
    try:
        code = int(getattr(error, "code", 0))
    except (TypeError, ValueError):
        return False
    message = str(error).lower()
    return code == 400 and (
        "task_type" in message or "task type" in message or "tasktype" in message
    )


def get_vector(
    text,
    task_type="RETRIEVAL_DOCUMENT",
    title=None,
    dim=None,
    model=None,
    style=None,
):
    """L2-normalized embedding, using an exact persistent cache before API.

    EMBED_CACHE_ONLY=1: cache hit allowed; a miss returns None without any API.
    EMBED_CACHE_ONLY absent/0: cache miss calls the existing Gemini model.

    Failed API calls or wrong-dimensional/nonfinite responses are never cached.
    """
    dim = resolve_embed_dim(app_config.EMBED_DIM if dim is None else dim)
    model = app_config.EMBED_MODEL if model is None else str(model).strip()
    style = resolve_embed_style(app_config.EMBED_STYLE if style is None else style)
    final_text, use_task_type = build_request(text, task_type, title, model, style)
    key = embedding_cache.request_key(
        text=final_text, model=model, dim=dim, style=style,
        task_type=task_type, use_task_type=use_task_type,
    )
    # Web searches use RETRIEVAL_QUERY. Document embedding keeps its original
    # behavior by default (important for existing evaluation and test tools).
    # Set EMBED_CACHE_DOCUMENTS=1 to opt in to document caching separately.
    cache_active = (task_type == "RETRIEVAL_QUERY") or (
        os.getenv("EMBED_CACHE_DOCUMENTS", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    if cache_active:
        cached = embedding_cache.read(key, dim)
        if cached is not None:
            return cached

    if os.getenv("EMBED_CACHE_ONLY", "0").strip().lower() in ("1", "true", "yes", "on"):
        print("ベクトル化を停止: オフラインモードでキャッシュがありません。")
        return None

    try:
        from google.genai import types
        client = _get_client()
        config_kwargs = {"output_dimensionality": dim}
        if use_task_type:
            config_kwargs["task_type"] = task_type
        effective_key = key
        try:
            response = client.models.embed_content(
                model=model,
                contents=final_text,
                config=types.EmbedContentConfig(**config_kwargs),
            )
        except Exception as error:
            if not (use_task_type and _is_task_type_unsupported(error)):
                raise
            # Preserve the existing fallback ONLY when the API explicitly
            # rejects task_type, not on quota, auth, or server errors.
            response = client.models.embed_content(
                model=model,
                contents=final_text,
                config=types.EmbedContentConfig(output_dimensionality=dim),
            )
            # Fallback used different request conditions; do not cache the
            # response under the original task_type-bearing request key.
            effective_key = embedding_cache.request_key(
                text=final_text, model=model, dim=dim, style=style,
                task_type=task_type, use_task_type=False,
            )

        vector = np.asarray(response.embeddings[0].values, dtype=np.float64)
        if vector.ndim != 1 or len(vector) != dim or not np.isfinite(vector).all():
            print("ベクトル化エラー: 不正な次元数または非有限値が返されました。")
            return None
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        if cache_active:
            embedding_cache.write(effective_key, vector, dim)
        return vector
    except Exception as error:
        code = getattr(error, "code", None)
        if code == 429:
            print("ベクトル化エラー: Gemini API の利用枠超過（429）。再試行しません。")
        elif code is not None:
            print(f"ベクトル化エラー: Gemini API HTTP {code}。")
        else:
            print(f"ベクトル化エラー: {type(error).__name__}。")
        return None
