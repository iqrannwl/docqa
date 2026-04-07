"""
Documents router.

POST   /documents          — ingest JSON documents
POST   /documents/upload   — upload PDF, TXT, or Markdown files
GET    /documents          — list all indexed documents
DELETE /documents/{doc_id} — delete a document and its vectors
"""

from __future__ import annotations
import uuid
import io
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse

from app.auth import require_api_key
from app.schemas import (
    IngestRequest, IngestResponse, DocumentIngested,
    DocumentListResponse, DocumentListItem,
    DeleteResponse,
)
from app.chunker import split_into_chunks
from app.embeddings import embed_texts
from app.vector_store import add_vectors, get_vector, rebuild_index
from app import database as db
from app.config import get_settings

settings = get_settings()
router = APIRouter()


# ── helpers ──────────────────────────────────────────────────────────────────

def _ingest_document(doc_id: str, title: str, content: str, source: str | None) -> int:
    """Chunk → embed → store. Returns number of chunks created."""
    chunks = split_into_chunks(content)
    if not chunks:
        raise ValueError("Document produced no chunks after splitting.")

    vectors = embed_texts(chunks)
    faiss_ids = add_vectors(vectors)

    db.insert_document(doc_id, title, source, len(chunks))
    for chunk_text, faiss_id in zip(chunks, faiss_ids):
        chunk_id = f"{doc_id}_{faiss_id}"
        db.insert_chunk(chunk_id, doc_id, faiss_id, chunk_text)

    return len(chunks)


def _extract_text_from_file(file: UploadFile, data: bytes) -> str:
    name = (file.filename or "").lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="pypdf is required for PDF uploads. Install it with: pip install pypdf",
            )
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=f"Unsupported file type: {name}. Accepted: .txt, .md, .pdf",
    )


# ── routes ───────────────────────────────────────────────────────────────────

@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_documents(
    payload: IngestRequest,
    _: str = Depends(require_api_key),
):
    """Ingest a list of documents (JSON). Each is chunked, embedded, and indexed."""
    ingested: list[DocumentIngested] = []
    for doc in payload.documents:
        doc_id = str(uuid.uuid4())
        num_chunks = _ingest_document(doc_id, doc.title, doc.content, doc.source)
        ingested.append(DocumentIngested(doc_id=doc_id, title=doc.title, num_chunks=num_chunks))

    return IngestResponse(ingested=ingested, total_chunks=sum(d.num_chunks for d in ingested))


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    _: str = Depends(require_api_key),
):
    """Upload a PDF, TXT, or Markdown file for indexing."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
        )

    content = _extract_text_from_file(file, data)
    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract any text from the uploaded file.",
        )

    doc_id = str(uuid.uuid4())
    title = os.path.splitext(file.filename or "Untitled")[0]
    num_chunks = _ingest_document(doc_id, title, content, file.filename)
    item = DocumentIngested(doc_id=doc_id, title=title, num_chunks=num_chunks)
    return IngestResponse(ingested=[item], total_chunks=num_chunks)


@router.get("", response_model=DocumentListResponse)
async def list_documents(_: str = Depends(require_api_key)):
    """List all indexed documents."""
    docs = db.list_documents()
    items = [DocumentListItem(**d) for d in docs]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, _: str = Depends(require_api_key)):
    """Delete a document and all its associated vector chunks from the index."""
    if not db.document_exists(doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    # Collect faiss IDs to remove
    faiss_ids_to_remove = set(db.delete_document(doc_id))

    # Rebuild FAISS index keeping all OTHER chunks
    all_chunks = db.get_chunks_for_doc.__doc__  # Just to avoid circular — we'll query differently
    # Fetch all remaining chunks from DB
    remaining_chunks = _get_all_remaining_chunks(faiss_ids_to_remove)

    rebuild_index(remaining_chunks)

    # After rebuild, FAISS IDs are now sequential 0..N-1 in insertion order.
    # Update the DB to reflect new faiss indices.
    _reassign_faiss_indices(remaining_chunks)

    return DeleteResponse(doc_id=doc_id, deleted=True, message="Document and all chunks removed.")


def _get_all_remaining_chunks(exclude_faiss_ids: set[int]) -> dict[int, list[float]]:
    """Return {old_faiss_id: vector} for chunks NOT in exclude set."""
    import sqlite3, os
    conn = sqlite3.connect(settings.metadata_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT faiss_index FROM chunks").fetchall()
    conn.close()

    result: dict[int, list[float]] = {}
    for row in rows:
        fid = row["faiss_index"]
        if fid not in exclude_faiss_ids:
            vec = get_vector(fid)
            if vec is not None:
                result[fid] = vec
    return result


def _reassign_faiss_indices(old_id_to_vector: dict[int, list[float]]):
    """Update SQLite chunk records with new sequential FAISS indices."""
    import sqlite3
    conn = sqlite3.connect(settings.metadata_db_path)
    old_ids = list(old_id_to_vector.keys())
    for new_idx, old_idx in enumerate(old_ids):
        conn.execute(
            "UPDATE chunks SET faiss_index=? WHERE faiss_index=?", (new_idx, old_idx)
        )
    conn.commit()
    conn.close()
