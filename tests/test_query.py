"""
Tests for POST /query — semantic search and answer generation.
"""

import pytest


@pytest.fixture(autouse=True)
def seed_docs(client, auth_headers, sample_docs):
    """Ensure documents are indexed before each query test."""
    client.post("/documents", json=sample_docs, headers=auth_headers)


def test_query_requires_auth(client):
    r = client.post("/query", json={"question": "What is FastAPI?"})
    assert r.status_code == 401


def test_query_returns_answer_and_sources(client, auth_headers):
    r = client.post(
        "/query",
        json={"question": "What is FastAPI used for?"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["answer"]) > 10
    assert "sources" in body
    assert len(body["sources"]) >= 1
    # Each source must have required fields
    for src in body["sources"]:
        assert "doc_id" in src
        assert "title" in src
        assert "content" in src
        assert "score" in src


def test_query_respects_top_k(client, auth_headers):
    r = client.post(
        "/query",
        json={"question": "vector embeddings similarity", "top_k": 1},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert len(r.json()["sources"]) <= 1


def test_query_short_question_rejected(client, auth_headers):
    r = client.post(
        "/query",
        json={"question": "Hi"},
        headers=auth_headers,
    )
    assert r.status_code == 422


def test_query_source_scores_descending(client, auth_headers):
    r = client.post(
        "/query",
        json={"question": "FAISS vector search library", "top_k": 5},
        headers=auth_headers,
    )
    sources = r.json()["sources"]
    if len(sources) > 1:
        scores = [s["score"] for s in sources]
        assert scores == sorted(scores, reverse=True)


def test_query_no_docs_still_responds(client, auth_headers):
    """Even with no matching context the API should return a graceful answer."""
    r = client.post(
        "/query",
        json={"question": "What is quantum entanglement?"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert "answer" in r.json()
