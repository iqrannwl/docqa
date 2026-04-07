"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ---------- Ingestion ----------

class DocumentIn(BaseModel):
    title: str = Field(..., description="Human-readable document title.")
    content: str = Field(..., description="Plain-text content of the document.")
    source: Optional[str] = Field(None, description="Optional source URL or file path.")

    model_config = {"json_schema_extra": {
        "example": {
            "title": "Introduction to FastAPI",
            "content": "FastAPI is a modern, fast web framework for building APIs with Python...",
            "source": "https://fastapi.tiangolo.com"
        }
    }}


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1)


class DocumentIngested(BaseModel):
    doc_id: str
    title: str
    num_chunks: int


class IngestResponse(BaseModel):
    ingested: list[DocumentIngested]
    total_chunks: int


# ---------- Listing ----------

class DocumentListItem(BaseModel):
    id: str
    title: str
    source: Optional[str]
    num_chunks: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


# ---------- Delete ----------

class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str


# ---------- Query ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")
    stream: Optional[bool] = Field(False, description="Stream the LLM response.")

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What is semantic search?",
            "top_k": 5,
            "stream": False
        }
    }}


class SourceChunk(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    content: str
    score: float = Field(description="Cosine similarity score (higher = more relevant).")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


# ---------- Auth ----------

class TokenResponse(BaseModel):
    api_key: str
    message: str
