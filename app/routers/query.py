"""
Query router.

POST /query — semantic search + LLM answer generation.
Supports both regular and streaming responses.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import require_api_key
from app.schemas import QueryRequest, QueryResponse, SourceChunk
from app.embeddings import embed_query
from app.vector_store import search
from app import database as db
from app.llm import generate_answer, stream_answer
from app.config import get_settings

settings = get_settings()
router = APIRouter()


def _retrieve_chunks(question: str, top_k: int) -> list[dict]:
    """Embed question, search FAISS, hydrate with metadata."""
    query_vec = embed_query(question)
    hits = search(query_vec, top_k=top_k)

    chunks: list[dict] = []
    for faiss_id, score in hits:
        chunk = db.get_chunk_by_faiss_index(faiss_id)
        if not chunk:
            continue
        doc = db.get_document(chunk["doc_id"])
        if not doc:
            continue
        chunks.append({
            "doc_id": chunk["doc_id"],
            "title": doc["title"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "score": score,
        })
    return chunks


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    _: str = Depends(require_api_key),
):
    """
    Answer a natural-language question using semantic search over indexed documents.

    - Embeds the question and retrieves the top-k most relevant chunks via FAISS.
    - Feeds the retrieved context to an LLM to generate a grounded answer.
    - Returns the answer plus the source chunks used.

    Set `stream: true` to receive a text/event-stream response instead.
    """
    top_k = payload.top_k or settings.top_k_chunks
    chunks = _retrieve_chunks(payload.question, top_k)

    if payload.stream:
        # Return a streaming SSE response
        async def event_generator():
            async for token in stream_answer(payload.question, chunks):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming: generate full answer then return
    answer = generate_answer(payload.question, chunks)

    source_chunks = [SourceChunk(**c) for c in chunks]
    return QueryResponse(
        question=payload.question,
        answer=answer,
        sources=source_chunks,
    )
