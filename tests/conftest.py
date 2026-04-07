"""
Pytest configuration and shared fixtures.
Uses temporary directories for FAISS index and SQLite DB so tests
never touch real data and are fully isolated.
"""

import os
import pytest
import tempfile
from fastapi.testclient import TestClient

# Point config to temp paths BEFORE importing the app
_tmp = tempfile.mkdtemp()
os.environ.setdefault("GEMINI_API_KEY", "")          # use local fallback
os.environ["FAISS_INDEX_PATH"] = os.path.join(_tmp, "faiss_index")
os.environ["METADATA_DB_PATH"] = os.path.join(_tmp, "metadata.db")
os.environ["API_KEYS"] = "test-key"


@pytest.fixture(scope="session")
def client():
    from app.main import app
    from app.database import init_db
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key"}


@pytest.fixture
def sample_docs():
    return {
        "documents": [
            {
                "title": "FastAPI Basics",
                "content": (
                    "FastAPI is a modern Python web framework for building APIs quickly. "
                    "It uses type hints and Pydantic for data validation. "
                    "FastAPI automatically generates OpenAPI documentation."
                ),
                "source": "https://fastapi.tiangolo.com",
            },
            {
                "title": "Vector Search",
                "content": (
                    "Vector search finds semantically similar documents using embeddings. "
                    "FAISS is a library by Meta AI for efficient similarity search. "
                    "Cosine similarity measures the angle between two vectors."
                ),
                "source": "https://example.com/vector-search",
            },
        ]
    }
