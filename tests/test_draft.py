"""Draft endpoint validation tests.

No model calls: only the request-validation paths are exercised, so
these run fast and offline.

Run:  .venv/bin/python -m pytest tests/test_draft.py -q
"""
import os
import tempfile
import uuid

os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("RATE_LIMIT", "30")
os.environ.setdefault("ANON_IP_DAILY", "1000")
os.environ.setdefault("SIGNUP_IP_DAILY", "1000")
os.environ.setdefault("FREE_QUESTIONS", "1000")
os.environ.setdefault("DAILY_FREE", "1000")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="onegov-test-"))
os.environ["PAYNOW_ID"] = ""
os.environ["PAYNOW_KEY"] = ""
os.environ["SMTP_HOST"] = ""

from fastapi.testclient import TestClient          # noqa: E402

import server                                      # noqa: E402
from agent import DRAFT_TYPES, DRAFT_STRUCTURE, DRAFT_PROMPT  # noqa: E402

client = TestClient(server.app, base_url="https://testserver")

FACTS = ("My client is owed four months rent of 200 dollars per month "
         "by a commercial tenant in Harare under a written lease.")


def test_draft_types_have_structures():
    assert set(DRAFT_TYPES) == {"demand", "heads", "opinion", "contract",
                                "affidavit", "application", "summons",
                                "plea", "lease", "will", "resolution"}
    assert set(DRAFT_STRUCTURE) == set(DRAFT_TYPES)
    for s in DRAFT_STRUCTURE.values():
        assert DRAFT_PROMPT.format(structure=s)  # formats cleanly


def test_draft_rejects_unknown_type():
    r = client.post("/api/draft/stream",
                    json={"doc_type": "poem", "facts": FACTS})
    assert r.status_code == 400


def test_draft_rejects_short_facts():
    r = client.post("/api/draft/stream",
                    json={"doc_type": "demand", "facts": "help"})
    assert r.status_code == 400


def test_draft_rejects_long_facts():
    r = client.post("/api/draft/stream",
                    json={"doc_type": "demand", "facts": "x" * 8001})
    assert r.status_code == 400


def test_refine_rejects_huge_previous_draft():
    r = client.post("/api/draft/stream",
                    json={"doc_type": "demand", "facts": FACTS,
                          "previous_draft": "x" * 30001})
    assert r.status_code == 400


def test_refine_allows_short_instruction_with_previous_draft():
    # short facts alone are rejected ...
    r = client.post("/api/draft/stream",
                    json={"doc_type": "demand", "facts": "x" * 4})
    assert r.status_code == 400
    # ... and still rejected in refine mode below the 5-char floor
    r = client.post("/api/draft/stream",
                    json={"doc_type": "demand", "facts": "x" * 4,
                          "previous_draft": "some draft text"})
    assert r.status_code == 400


def test_draft_requires_fields():
    assert client.post("/api/draft/stream", json={}).status_code == 422
    assert client.post("/api/draft/stream",
                       json={"doc_type": "demand"}).status_code == 422
