"""FAQ answer cache: key normalisation, hit-extended TTL, invalidation."""
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


class _Req:
    def __init__(self, message, lang="en"):
        self.message, self.lang = message, lang


@pytest.fixture
def key():
    k = "test-" + uuid.uuid4().hex
    yield k
    with server._db() as c:
        c.execute("DELETE FROM cache WHERE key=?", (k,))


@pytest.mark.parametrize("raw, expected", [
    ("What is the speed limit?", "what is the speed limit"),
    ("  WHAT   IS the Speed Limit ?? ", "what is the speed limit"),
    ("Hi, please can you tell me what is the speed limit?",
     "what is the speed limit"),
    ("I want to know about maternity leave", "maternity leave"),
    ("Can you explain the speed limit, thanks", "the speed limit"),
])
def test_normalise_question(raw, expected):
    assert server._normalise_question(raw) == expected


def test_phrasings_of_one_question_share_a_key():
    a = server._cache_key(_Req("What is the speed limit?"))
    b = server._cache_key(_Req("hi please can you tell me what is the "
                               "speed limit"))
    assert a == b


@pytest.mark.parametrize("x, y", [
    # opposite questions: sharing a cache entry would be a wrong answer
    ("can I evict my tenant", "can I be evicted by my landlord"),
    ("what notice must I give", "what notice must I receive"),
    ("section 71 of the Constitution", "section 17 of the Constitution"),
])
def test_different_questions_do_not_share_a_key(x, y):
    assert server._cache_key(_Req(x)) != server._cache_key(_Req(y))


def test_language_separates_keys():
    assert (server._cache_key(_Req("maternity leave", "en")) !=
            server._cache_key(_Req("maternity leave", "sn")))


def test_corpus_version_is_in_the_key(monkeypatch):
    """A re-ingest must invalidate cached answers — the new corpus may
    contain the very instrument that changes them."""
    before = server._cache_key(_Req("maternity leave"))
    monkeypatch.setattr(server, "_corpus_version", lambda: "999999999")
    assert server._cache_key(_Req("maternity leave")) != before


def test_roundtrip_and_hit_counting(key):
    server._cache_put(key, "ANSWER", ["q1"])
    assert server._cache_get(key) == ("ANSWER", ["q1"], {})
    with server._db() as c:
        hits = c.execute("SELECT hits FROM cache WHERE key=?",
                         (key,)).fetchone()[0]
    assert hits == 1


def test_everything_hung_off_the_answer_survives_the_cache(key):
    """Sources, trail and follow-up chips come back on a cache hit.

    A hit has no agent run behind it, so anything not stored here
    silently disappears — on the questions asked most often, which is
    exactly where the UI can least afford to look broken."""
    meta = {"steps": ["search", "read"],
            "sources": [{"instrument": "Labour Act", "citation": "s 18"}],
            "followups": ["How much notice must an employer give?"],
            "unverified": []}
    server._cache_put(key, "ANSWER", ["maternity leave"], meta)
    assert server._cache_get(key) == ("ANSWER", ["maternity leave"], meta)


def test_popular_answers_outlive_the_base_ttl(key):
    server._cache_put(key, "ANSWER", [])
    aged = time.time() - (server.CACHE_TTL + 60)
    with server._db() as c:
        c.execute("UPDATE cache SET ts=?, hits=3 WHERE key=?", (aged, key))
    assert server._cache_get(key) is not None

    with server._db() as c:
        c.execute("UPDATE cache SET ts=?, hits=0 WHERE key=?", (aged, key))
    assert server._cache_get(key) is None


def test_refresh_preserves_popularity(key):
    """A stale answer to a hot question is still a hot question."""
    server._cache_put(key, "OLD", [])
    with server._db() as c:
        c.execute("UPDATE cache SET hits=5 WHERE key=?", (key,))
    server._cache_put(key, "NEW", [])
    with server._db() as c:
        answer, hits = c.execute("SELECT answer, hits FROM cache WHERE "
                                 "key=?", (key,)).fetchone()
    assert (answer, hits) == ("NEW", 5)


def test_missing_key_returns_none():
    assert server._cache_get("no-such-key-" + uuid.uuid4().hex) is None
