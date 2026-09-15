"""Cited-section lookup and follow-up splitting.

/api/passage is what makes a citation checkable: the reader clicks it
and sees the provision in the instrument's own words. Resolving to the
wrong instrument would be worse than failing, so the matching rules get
their own tests.
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="onegov-test-"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException          # noqa: E402

import agent as agent_mod                  # noqa: E402
import server                              # noqa: E402


# ---------------- citation parsing ----------------

@pytest.mark.parametrize("cite, name, section", [
    ("Labour Act, s 18 (Maternity leave)", "Labour Act", "18"),
    ("section 71 of the Constitution", "Constitution", "71"),
    ("Section 12(4) of the Labour Act", "Labour Act", "12"),
    ("Labour Act [Chapter 28:01]", "Labour Act [Chapter 28:01]", None),
    ("Constitution Of Zimbabwe Consolidated 2023, s 56 (Equality)",
     "Constitution Of Zimbabwe Consolidated 2023", "56"),
])
def test_split_citation(cite, name, section):
    assert server._split_citation(cite) == (name, section)


def test_section_number_does_not_leak_into_the_instrument_name():
    """'Labour Act, s 18' used to resolve to the Sericulture Act
    [Chapter 18:18]: the section number matched the chapter number."""
    name, _ = server._split_citation("Labour Act, s 18 (Maternity leave)")
    assert "18" not in name


# ---------------- instrument resolution ----------------

@pytest.mark.parametrize("cite", [
    "Road Traffic Act [Chapter 13:11]",
    "Labour Act [Chapter 28:01]",
    "Unlawful Organizations Act [Chapter 11:13]",
    "Criminal Law (Codification and Reform) Act [Chapter 9:23]",
])
def test_resolved_chapter_matches_the_cited_chapter(cite):
    """13:11 and 11:13 share their digits, so bag-of-tokens matching used
    to answer "Road Traffic Act" with the Unlawful Organizations Act."""
    hit = server._best_instrument(cite)
    if not hit:
        pytest.skip("not in this corpus")
    found = server._CHAPTER_RE.search(hit[0])
    if found:            # a title that names no chapter can't contradict
        assert found.groups() == server._CHAPTER_RE.search(cite).groups()


def test_lettered_si_is_a_different_instrument():
    """SI 127A of 2021 is not SI 127 of 2021, but the letter tokenises
    away, leaving the two titles otherwise identical."""
    plain = server._best_instrument("SI 127 of 2021")
    lettered = server._best_instrument("SI 127A of 2021")
    if not (plain and lettered):
        pytest.skip("not in this corpus")
    assert plain[0] != lettered[0]
    assert not re.search(r"\b127\s*A\b", plain[0], re.I)


# ---------------- /api/passage ----------------

def test_returns_the_cited_section_text():
    p = server.passage("section 71 of the Constitution")
    assert p["exact"] is True
    assert p["section"] == "71"
    assert "property" in p["text"].lower()
    assert p["url"].startswith("/api/download/")


def test_section_can_be_passed_separately():
    a = server.passage("section 71 of the Constitution")
    b = server.passage("Constitution", section="71")
    assert a["text"] == b["text"]


def test_unknown_instrument_is_a_404():
    with pytest.raises(HTTPException) as e:
        server.passage("The Imaginary Widgets Act 1999")
    assert e.value.status_code == 404


def test_unknown_section_falls_back_to_the_instrument():
    """Front matter is still more use to the reader than an error, so
    long as exact=False says what they are looking at."""
    p = server.passage("Constitution of Zimbabwe", section="998")
    assert p["exact"] is False and p["text"]
    assert p["section"] is None


# ---------------- follow-up chips ----------------
# The model marks its suggested next questions with a trailing ">>" line
# each (SYSTEM_PROMPT rule 6) and the agent strips them into structured
# data, so they never reach an export, a share or the PDF.


def _extract(answer):
    """(answer as the reader sees it, chips) from a raw model answer."""
    a = agent_mod.LawAgent.__new__(agent_mod.LawAgent)
    a.followups = []
    return a._extract_followups(answer), a.followups


def test_followups_are_stripped_off_the_answer():
    body, qs = _extract(
        "Ninety days.\n\n**Sources**\n\nLabour Act\n"
        ">> Who pays maternity leave?\n"
        ">> Can my employer refuse it?\n")
    assert ">>" not in body
    assert body.endswith("Labour Act")
    assert qs == ["Who pays maternity leave?",
                  "Can my employer refuse it?"]


def test_at_most_three_followups():
    _, qs = _extract("Text.\n" + "".join(
        f">> Question number {i} here?\n" for i in range(6)))
    assert len(qs) == 3


def test_a_quote_mid_answer_is_not_a_followup():
    """Only the trailing run is taken. A '>>' earlier in the prose is
    the reader's own text or a blockquote, and must survive."""
    answer = "The Act says:\n>> no person shall...\n\nSo you may not."
    body, qs = _extract(answer)
    assert body == answer and qs == []


@pytest.mark.parametrize("answer", [
    "A plain answer with no markers at all.",
    "",
])
def test_answers_without_followups_are_untouched(answer):
    body, qs = _extract(answer)
    assert (body, qs) == (answer, [])
