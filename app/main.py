"""
Document QA API — Main Application Entry Point
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from google.genai.errors import ClientError
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

@app.exception_handler(ClientError)
async def genai_exception_handler(request: Request, exc: ClientError):
    if exc.status_code == 429:
        return JSONResponse(
            status_code=429,
            content={"detail": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits."},
        )
    return JSONResponse(
        status_code=exc.status_code or 500,
        content={"detail": str(exc)},
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
