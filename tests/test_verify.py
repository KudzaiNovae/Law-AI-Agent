"""Citation verification and query decomposition — no network, no model.

The agent tests drive LawAgent with a scripted fake provider so the
repair round can be observed deterministically.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_mod        # noqa: E402
import decompose                 # noqa: E402
import verify                    # noqa: E402

CHUNKS = [
    {"doc_title": "Constitution Of Zimbabwe Consolidated 2023",
     "section": "71"},
    {"doc_title": "Constitution Of Zimbabwe Consolidated 2023",
     "section": "56"},
    {"doc_title": "Labour Act [Chapter 28:01] (consolidated as at "
                  "2016-12-31)", "section": "18"},
    {"doc_title": "SI 2021-127 Road Traffic Regulations", "section": "4"},
]


# ---------------- verify.py ----------------

@pytest.mark.parametrize("text, expected", [
    ("Section 71 of the Constitution protects property.", [("Constitution", "71")]),
    ("Section 12(4) of the Labour Act applies.", [("Labour Act", "12")]),
    ("Under section 56 of the Constitution and section 18 of the Labour "
     "Act, both apply.", [("Constitution", "56"), ("Labour Act", "18")]),
    # nothing identifiable to check — must not be guessed at
    ("Section 4 of the Act requires 30 days notice.", []),
    ("The Labour Act governs employment.", []),
])
def test_claims(text, expected):
    assert verify.claims(text) == expected


@pytest.mark.parametrize("text", [
    "Section 71 of the Constitution protects property rights.",
    "Section 18 of the Labour Act covers maternity leave.",
    "Section 4 of Statutory Instrument 127 of 2021 sets the limit.",
    "Section 4 of the Act requires 30 days notice.",
    "Sorry, the corpus does not appear to cover that.",
    "",
])
def test_supported_citations_are_not_flagged(text):
    """False positives are worse than false negatives: a correct answer
    must never be sent back for repair."""
    assert verify.check(text, CHUNKS) == []


def test_flags_section_not_retrieved():
    out = verify.check("Section 99 of the Labour Act is decisive.", CHUNKS)
    assert len(out) == 1 and "99" in out[0]


def test_flags_instrument_never_retrieved():
    out = verify.check("Section 12 of the Mines and Minerals Act applies.",
                       CHUNKS)
    assert len(out) == 1 and "no tool result" in out[0]


def test_no_chunks_means_no_opinion():
    assert verify.check("Section 99 of the Labour Act.", []) == []


# ---------------- decompose.py ----------------

@pytest.mark.parametrize("q, compound", [
    ("maternity leave", False),
    ("section 71 of the Constitution", False),
    ("can my landlord evict me and keep my deposit?", True),
    ("Is SI 118 of 2023 still in force? What replaced it?", True),
])
def test_looks_compound(q, compound):
    assert decompose.looks_compound(q) is compound


@pytest.mark.parametrize("raw, expected", [
    ('{"queries": ["a bb", "c dd"]}', ["a bb", "c dd"]),
    ('```json\n{"queries": ["a bb", "c dd"]}\n```', ["a bb", "c dd"]),
    ('Sure! {"queries": ["a bb", "a bb", "c dd"]}', ["a bb", "c dd"]),
    ("not json", []),
    ('{"queries": []}', []),
])
def test_parse(raw, expected):
    assert decompose._parse(raw) == expected


# ---------------- fake provider ----------------

class _Fn:
    def __init__(self, name, args):
        self.name, self.arguments = name, args


class _TC:
    def __init__(self, name, args):
        self.index, self.id, self.type = 0, "call_0", "function"
        self.function = _Fn(name, args)

    def model_dump(self):
        return {"index": 0, "id": self.id, "type": "function",
                "function": {}}


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Chunk:
    def __init__(self, delta):
        self.choices = [type("C", (), {"delta": delta})()]


class _Completions:
    """Replays a scripted list of turns; each turn is text or a tool call."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        turn = self.script.pop(0)
        if turn[0] == "tool":
            return iter([_Chunk(_Delta(tool_calls=[
                _TC(turn[1], json.dumps(turn[2]))]))])
        return iter([_Chunk(_Delta(content=turn[1]))])


class _FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": _Completions(script)})()


class _StubRetriever:
    """Returns one Labour Act s 18 chunk for any search."""

    def search(self, query, **kw):
        return [{"doc_title": "Labour Act [Chapter 28:01] (consolidated as "
                              "at 2016-12-31)",
                 "section": "18", "doc_type": "act",
                 "citation": "Labour Act, s 18 (Maternity leave)",
                 "text": "Female employees are entitled to maternity "
                         "leave of ninety days." * 3,
                 "source_file": "labour.pdf", "score": 0.5}]

    def multi_search(self, queries, **kw):
        return self.search(queries[0], **kw)

    def related_updates(self, *a, **kw):
        return []


def _agent(script, monkeypatch):
    monkeypatch.setattr(agent_mod, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(agent_mod, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(agent_mod, "OLLAMA_MODEL", "")
    # decomposition is a separate LLM call; exercised by its own tests
    monkeypatch.setattr(decompose, "subqueries", lambda *a, **k: [])
    a = agent_mod.LawAgent(retriever=_StubRetriever())
    client = _FakeClient(script)
    a.chain = [{"name": "fake", "model": "fake-1", "client": client}]
    return a, client


BAD = ("Section 99 of the Labour Act gives ninety days.\n\n"
       "**Sources**\n\nLabour Act [Chapter 28:01]\n")
GOOD = ("Section 18 of the Labour Act gives ninety days.\n\n"
        "**Sources**\n\nLabour Act [Chapter 28:01]\n")


def test_repair_round_fixes_unsupported_citation(monkeypatch):
    a, client = _agent([
        ("tool", "search_legislation", {"query": "maternity leave"}),
        ("text", BAD),      # cites s 99, which was never retrieved
        ("text", GOOD),     # corrected after the citation check
    ], monkeypatch)
    events = []
    a.on_event = lambda label, kind="search": events.append((kind, label))

    assert a.ask("how much maternity leave?") == GOOD
    assert a._repaired is True
    assert ("verify", "checking citations") in events
    # the repair prompt went into history and named the offending section
    prompts = [m["content"] for m in a.messages
               if m.get("role") == "user"
               and "CITATION CHECK" in (m.get("content") or "")]
    assert len(prompts) == 1 and "99" in prompts[0]


def test_clean_answer_costs_no_extra_round(monkeypatch):
    a, client = _agent([
        ("tool", "search_legislation", {"query": "maternity leave"}),
        ("text", GOOD),
    ], monkeypatch)

    assert a.ask("how much maternity leave?") == GOOD
    assert a._repaired is False
    assert len(client.chat.completions.calls) == 2   # no repair round


def test_repair_runs_at_most_once(monkeypatch):
    """A second bad draft is returned as-is rather than looping."""
    a, client = _agent([
        ("tool", "search_legislation", {"query": "maternity leave"}),
        ("text", BAD),
        ("text", BAD),      # still wrong; must not trigger another repair
    ], monkeypatch)

    assert a.ask("how much maternity leave?") == BAD
    assert len(client.chat.completions.calls) == 3


def test_verification_can_be_disabled(monkeypatch):
    monkeypatch.setattr(agent_mod, "VERIFY_CITATIONS", False)
    a, client = _agent([
        ("tool", "search_legislation", {"query": "maternity leave"}),
        ("text", BAD),
    ], monkeypatch)

    assert a.ask("how much maternity leave?") == BAD
    assert len(client.chat.completions.calls) == 2
