"""
Metadata store — SQLite via Python's built-in sqlite3.
Stores document-level metadata; FAISS handles the vectors.
"""

import sqlite3
import os
from app.config import get_settings

settings = get_settings()


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.metadata_db_path), exist_ok=True)
    conn = sqlite3.connect(settings.metadata_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                source      TEXT,
                num_chunks  INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                faiss_index INTEGER NOT NULL,
                content     TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        conn.commit()


# ---------- Document helpers ----------

def insert_document(doc_id: str, title: str, source: str | None, num_chunks: int):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, title, source, num_chunks) VALUES (?,?,?,?)",
            (doc_id, title, source, num_chunks),
        )
        conn.commit()


def insert_chunk(chunk_id: str, doc_id: str, faiss_index: int, content: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, faiss_index, content) VALUES (?,?,?,?)",
            (chunk_id, doc_id, faiss_index, content),
        )
        conn.commit()


def list_documents() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, source, num_chunks, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    return dict(row) if row else None


def get_chunks_for_doc(doc_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
    return [dict(r) for r in rows]


def get_chunk_by_faiss_index(faiss_idx: int) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM chunks WHERE faiss_index=?", (faiss_idx,)).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str) -> list[int]:
    """Delete doc + chunks. Return list of faiss indices that were removed."""
    faiss_indices = [c["faiss_index"] for c in get_chunks_for_doc(doc_id)]
    with _get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
    return faiss_indices


def document_exists(doc_id: str) -> bool:
    return get_document(doc_id) is not None
