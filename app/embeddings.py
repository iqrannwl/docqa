from __future__ import annotations
from app.config import get_settings

settings = get_settings()


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai

    client = genai.Client(
                api_key=settings.gemini_api_key
            )

    results: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = client.models.embed_content(
            model=settings.embedding_model,
            contents=batch
        )

        # ✅ Correct parsing
        results.extend([e.values for e in response.embeddings])

    return results


def _embed_local(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "Neither GEMINI_API_KEY nor sentence-transformers is available."
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts, normalize_embeddings=False)
    return [v.tolist() for v in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if settings.gemini_api_key:
        return _embed_gemini(texts)
    return _embed_local(texts)


def embed_query(question: str) -> list[float]:
    if settings.gemini_api_key:
        from google import genai

        client = genai.Client(
            api_key=settings.gemini_api_key
        )
        response = client.models.embed_content(
            model=settings.embedding_model,
            contents=[question]   # must be list
        )

        return response.embeddings[0].values

    return _embed_local([question])[0]
