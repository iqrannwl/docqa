"""
Document QA API — Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.routers import documents, query, auth
from app.database import init_db

app = FastAPI(
    title="Document QA API",
    description="Semantic document search and question answering powered by vector embeddings and LLMs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the UI
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize the database/vector store on startup."""
    init_db()


@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend UI."""
    index_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Document QA API is running. Visit /docs for the API reference."}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


# Register routers
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(query.router, tags=["Query"])
