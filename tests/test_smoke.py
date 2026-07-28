"""Smoke tests — no model calls, just API surface.

Run:  .venv/bin/python -m pytest tests/ -q
"""
import os
import tempfile

os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("ANON_IP_DAILY", "1000")
os.environ.setdefault("SIGNUP_IP_DAILY", "1000")
# fresh DB per run: never touch the real data/onegov.db
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="onegov-test-"))

from fastapi.testclient import TestClient          # noqa: E402

import server                                      # noqa: E402

client = TestClient(server.app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    j = r.json()
    assert {"updated", "instruments", "chunks"} <= set(j)
    assert j["chunks"] > 0


def test_catalog():
    r = client.get("/api/catalog")
    assert r.status_code == 200
    assert isinstance(r.json(), list) and r.json()


def test_feedback_validation():
    r = client.post("/api/feedback", json={"verdict": "meh"})
    assert r.status_code == 400
    r = client.post("/api/feedback", json={"verdict": "up", "question": "t"})
    assert r.status_code == 200


def test_download_traversal_blocked():
    assert client.get("/api/download/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/download/.hidden").status_code == 404


def test_chat_validation():
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 400
    r = client.post("/api/chat", json={"message": "x" * 9000})
    assert r.status_code == 400


def test_rate_limit():
    # empty-message requests still count nothing until _check_rate passes;
    # hammer with invalid payloads that pass rate check first
    codes = [client.post("/api/chat", json={"message": ""}).status_code
             for _ in range(server.RATE_LIMIT + 3)]
    assert 429 in codes


def test_whatsnew():
    r = client.get("/api/whatsnew")
    assert r.status_code == 200
    items = r.json()
    assert items and {"title", "doc_type", "year", "file"} <= set(items[0])
    r = client.get("/api/whatsnew", params={"n": 3})
    assert len(r.json()) == 3


def test_share_404():
    assert client.get("/s/nope").status_code == 404
    assert client.get("/api/answer/nope").status_code == 404


def test_admin_token():
    assert client.get("/api/admin/stats").status_code == 403
    r = client.get("/api/admin/stats", params={"token": "test-token"})
    assert r.status_code == 200
    assert "chats" in r.json()
