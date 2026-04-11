"""
LLM service — generates answers using Google Gemini.
Supports both streaming and non-streaming responses.
"""

from __future__ import annotations
from typing import AsyncIterator
from app.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are a precise, helpful document assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, say so clearly — do not invent information.
Be concise and cite which document titles your answer draws from.
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (Source: {c['title']})\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Synchronous answer generation."""
    if settings.gemini_api_key:
        return _gemini_answer(question, chunks)
    return _local_answer(question, chunks)


async def stream_answer(question: str, chunks: list[dict]) -> AsyncIterator[str]:
    """Async streaming answer generation."""
    if settings.gemini_api_key:
        async for token in _gemini_stream(question, chunks):
            yield token
    else:
        yield _local_answer(question, chunks)


# ---------- Gemini backend ----------

def _gemini_answer(question: str, chunks: list[dict]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    context = _build_context(chunks)
    prompt = _build_prompt(question, context)

    response = client.models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=settings.llm_temperature,
            max_output_tokens=settings.max_tokens,
        ),
    )
    return response.text.strip()


async def _gemini_stream(question: str, chunks: list[dict]) -> AsyncIterator[str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    context = _build_context(chunks)
    prompt = _build_prompt(question, context)

    async for chunk in client.aio.models.generate_content_stream(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=settings.llm_temperature,
            max_output_tokens=settings.max_tokens,
        ),
    ):
        if chunk.text:
            yield chunk.text


# ---------- Local fallback ----------

def _local_answer(question: str, chunks: list[dict]) -> str:
    """
    Simple extractive fallback when no Gemini API key is available.
    Returns the most relevant chunk as the 'answer'.
    """
    if not chunks:
        return "No relevant documents were found to answer your question."
    best = chunks[0]
    return (
        f"[Extractive fallback — set GEMINI_API_KEY for generative answers]\n\n"
        f"Most relevant passage from '{best['title']}':\n\n{best['content']}"
    )
