"""
Embedding service — wraps Google Gemini's embedding API.
Falls back to a lightweight local model (sentence-transformers) when
GEMINI_API_KEY is not set, so the app works offline / in tests.

Gemini text-embedding-004 produces 768-dimensional vectors.
"""

from __future__ import annotations
from app.config import get_settings

settings = get_settings()


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    import google.generativeai as genai
    genai.configure(api_key=settings.gemini_api_key)
    results: list[list[float]] = []
    # Batch in 100s to stay within API limits
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = genai.embed_content(
            model=settings.embedding_model,
            content=batch,
            task_type="retrieval_document",
        )
        embeddings = response["embedding"]
        # Single string returns one vector; list returns list of vectors
        if isinstance(embeddings[0], float):
            embeddings = [embeddings]
        results.extend(embeddings)
    return results


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Fallback: sentence-transformers (all-MiniLM-L6-v2, dim=384)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "Neither GEMINI_API_KEY nor sentence-transformers is available. "
            "Install sentence-transformers or set GEMINI_API_KEY."
        )
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts, normalize_embeddings=False)
    return [v.tolist() for v in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts and return vectors."""
    if settings.gemini_api_key:
        return _embed_gemini(texts)
    return _embed_local(texts)


def embed_query(question: str) -> list[float]:
    """Embed a single query string (uses retrieval_query task type)."""
    if settings.gemini_api_key:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        response = genai.embed_content(
            model=settings.embedding_model,
            content=question,
            task_type="retrieval_query",
        )
        return response["embedding"]
    return _embed_local([question])[0]
