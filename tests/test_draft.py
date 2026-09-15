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


def test_draft_prompt_asks_for_the_three_closing_sections_in_order():
    """Every draft ends Drafting notes -> Sources -> What to do next.

    The order is load-bearing rather than cosmetic: "What to do next" is
    the only part a self-represented reader can act on, so it is the
    part they should finish on. It is also easy to reintroduce the
    ambiguity that was here before — two separate rules each said
    "after the Drafting notes", and which section landed second was left
    to the model.
    """
    p = DRAFT_PROMPT.format(structure=DRAFT_STRUCTURE["summons"])
    order = [p.index("**Drafting notes**"),
             p.index("**Sources**"),
             p.index("**What to do next**")]
    assert order == sorted(order), "closing sections named out of order"


def test_draft_prompt_requires_a_jurisdiction_check():
    """A summons issued out of a court that cannot hear the claim is a
    nullity, and one issued out of the High Court when the Magistrates
    Court would do costs the client more than the claim. The limits move
    by statutory instrument, so the prompt must send the model to look
    them up rather than trust a remembered figure."""
    p = DRAFT_PROMPT.format(structure=DRAFT_STRUCTURE["summons"])
    assert "Monetary Limits" in p and "Small Claims" in p
    # The budget has to leave room for that search on top of the 2-4
    # substantive ones the prompt already asks for.
    assert "at most 7 tool calls" in p


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
