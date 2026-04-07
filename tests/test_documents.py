"""
Tests for POST /documents, GET /documents, DELETE /documents/{id},
and POST /documents/upload.
"""

import io
import pytest


def test_ingest_requires_auth(client):
    r = client.post("/documents", json={"documents": []})
    assert r.status_code == 401


def test_ingest_documents(client, auth_headers, sample_docs):
    r = client.post("/documents", json=sample_docs, headers=auth_headers)
    assert r.status_code == 201
    body = r.json()
    assert len(body["ingested"]) == 2
    assert body["total_chunks"] >= 2
    for item in body["ingested"]:
        assert "doc_id" in item
        assert item["num_chunks"] >= 1


def test_list_documents(client, auth_headers, sample_docs):
    # Ingest first
    client.post("/documents", json=sample_docs, headers=auth_headers)
    r = client.get("/documents", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2
    titles = [d["title"] for d in body["documents"]]
    assert "FastAPI Basics" in titles or "Vector Search" in titles


def test_delete_document(client, auth_headers, sample_docs):
    # Ingest and capture a doc_id
    r = client.post("/documents", json={"documents": [sample_docs["documents"][0]]}, headers=auth_headers)
    doc_id = r.json()["ingested"][0]["doc_id"]

    del_r = client.delete(f"/documents/{doc_id}", headers=auth_headers)
    assert del_r.status_code == 200
    assert del_r.json()["deleted"] is True

    # Should be gone from list
    list_r = client.get("/documents", headers=auth_headers)
    ids = [d["id"] for d in list_r.json()["documents"]]
    assert doc_id not in ids


def test_delete_nonexistent_returns_404(client, auth_headers):
    r = client.delete("/documents/does-not-exist", headers=auth_headers)
    assert r.status_code == 404


def test_upload_txt_file(client, auth_headers):
    content = b"This is a plain text document about machine learning and neural networks."
    r = client.post(
        "/documents/upload",
        files={"file": ("ml_notes.txt", io.BytesIO(content), "text/plain")},
        headers=auth_headers,
    )
    assert r.status_code == 201
    body = r.json()
    assert body["ingested"][0]["title"] == "ml_notes"
    assert body["total_chunks"] >= 1


def test_upload_markdown_file(client, auth_headers):
    content = b"# Introduction\n\nThis document explains semantic search in detail.\n"
    r = client.post(
        "/documents/upload",
        files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
        headers=auth_headers,
    )
    assert r.status_code == 201


def test_upload_unsupported_type(client, auth_headers):
    r = client.post(
        "/documents/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
        headers=auth_headers,
    )
    assert r.status_code == 415


def test_ingest_empty_list_rejected(client, auth_headers):
    r = client.post("/documents", json={"documents": []}, headers=auth_headers)
    assert r.status_code == 422
