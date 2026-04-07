# Document QA API

A production-ready **Retrieval-Augmented Generation (RAG)** API built with FastAPI and **Google Gemini**.  
Upload documents → ask natural-language questions → get grounded, cited answers.

---

## Features

- **Semantic search** via FAISS + Gemini `text-embedding-004` (768-dim, free tier)
- **LLM-powered answers** via Gemini 1.5 Flash / Pro (grounded to your documents)
- **File uploads** — PDF, TXT, Markdown
- **Streaming responses** (SSE) for long answers
- **API key authentication** (`X-API-Key` header)
- **Async FastAPI** routes throughout
- **Built-in frontend UI** at `http://localhost:8000`
- **Persistent storage** — FAISS index + SQLite survive restarts

---

## Tech Stack

| Layer | Library | Why |
|---|---|---|
| Web framework | FastAPI | Async, type-safe, auto-docs |
| Data validation | Pydantic v2 | Zero-boilerplate schemas |
| Embeddings | Gemini `text-embedding-004` | Free tier, 768-dim, task-aware |
| Vector store | FAISS (`faiss-cpu`) | Fast local similarity search |
| Metadata store | SQLite (stdlib) | Zero-dependency persistence |
| LLM | Gemini 1.5 Flash / Pro | Fast, cheap, high-quality answers |
| PDF parsing | pypdf | Pure-Python, no system deps |

---

## Setup

### 1. Get a Gemini API key

Go to https://aistudio.google.com/app/apikey → **Create API key** (free).

### 2. Clone & install

```bash
git clone <your-repo-url>
cd docqa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and set GEMINI_API_KEY=AIza...
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(empty)_ | Required — from Google AI Studio |
| `LLM_MODEL` | `gemini-1.5-flash` | `gemini-1.5-pro`, `gemini-2.0-flash`, etc. |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model |
| `API_KEYS` | `dev-secret-key` | Comma-separated valid API keys |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `TOP_K_CHUNKS` | `5` | Chunks retrieved per query |

> **No Gemini key?** The API still works using a local `sentence-transformers` fallback for embeddings and extractive fallback for answers. Install it with:
> `pip install sentence-transformers`

### 4. Run

```bash
uvicorn app.main:app --reload
```

- Frontend UI: http://localhost:8000  
- API docs: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc

---

## API Reference

All endpoints (except `/health`) require the header:
```
X-API-Key: dev-secret-key
```

---

### POST `/documents` — Ingest documents

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d @sample_data/documents.json
```

**Request body:**
```json
{
  "documents": [
    {
      "title": "My Document",
      "content": "Full text content of the document...",
      "source": "https://optional-url.com"
    }
  ]
}
```

---

### POST `/documents/upload` — Upload a file

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "X-API-Key: dev-secret-key" \
  -F "file=@/path/to/document.pdf"
```

Supported: `.pdf`, `.txt`, `.md`

---

### GET `/documents` — List all documents

```bash
curl http://localhost:8000/documents -H "X-API-Key: dev-secret-key"
```

---

### DELETE `/documents/{doc_id}` — Delete a document

```bash
curl -X DELETE http://localhost:8000/documents/<doc_id> \
  -H "X-API-Key: dev-secret-key"
```

---

### POST `/query` — Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"question": "What is semantic search?", "top_k": 5}'
```

**Response:**
```json
{
  "question": "What is semantic search?",
  "answer": "Semantic search is a technique that understands the intent behind a query...",
  "sources": [
    {
      "doc_id": "uuid",
      "title": "Introduction to Semantic Search",
      "chunk_id": "uuid_0",
      "content": "Semantic search is a data searching technique...",
      "score": 0.92
    }
  ]
}
```

**Streaming mode:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"question": "Explain RAG", "stream": true}' \
  --no-buffer
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use temporary SQLite + FAISS instances and the local embedding fallback — no Gemini key required.

---

## Quick Start with Sample Data

```bash
# 1. Start the server
uvicorn app.main:app --reload

# 2. Index the sample documents
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d @sample_data/documents.json

# 3. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key" \
  -d '{"question": "What is FAISS and what is it used for?"}'
```

---

## Project Structure

```
docqa/
├── app/
│   ├── main.py          # FastAPI app, middleware, startup
│   ├── config.py        # Settings (GEMINI_API_KEY, etc.)
│   ├── schemas.py       # Pydantic request/response models
│   ├── auth.py          # API key dependency
│   ├── database.py      # SQLite metadata store
│   ├── chunker.py       # Text splitting
│   ├── embeddings.py    # Gemini text-embedding-004
│   ├── vector_store.py  # FAISS wrapper
│   ├── llm.py           # Gemini answer generation (streaming + sync)
│   └── routers/
│       ├── auth.py
│       ├── documents.py
│       └── query.py
├── tests/
├── sample_data/
├── static/index.html    # Frontend UI
├── requirements.txt
├── .env.example
└── README.md
```
