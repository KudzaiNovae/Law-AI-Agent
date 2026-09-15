"""Query rewriting and result de-duplication.

These are the rules that decide what the agent gets to read, so each one
here earned its place by having produced a wrong answer first. The
corpus-level checks live in tests/evals.jsonl instead — this file only
covers the pure functions, so it runs without Chroma.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="onegov-test-"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index.retriever import (            # noqa: E402
    ACRONYMS, HybridRetriever, Q_CASELAW_RE, _near_duplicate, _stem,
    _word_set, normalize_refs)


# ---------------- acronym expansion ----------------

@pytest.mark.parametrize("query, expected", [
    ("NSSA pension contributions", "National Social Security Authority"),
    ("VAT registration threshold", "Value Added Tax"),
    ("what ZIMRA charges on imports", "Zimbabwe Revenue Authority"),
])
def test_acronyms_are_expanded_to_corpus_vocabulary(query, expected):
    """Nobody writes "National Social Security Authority", and the Act
    is titled nothing else — so the acronym alone matched no chunk."""
    assert expected.lower() in normalize_refs(query).lower()


def test_the_acronym_itself_is_kept():
    """A few instruments are titled by the acronym. Expanding in place
    would lose them, so the expansion is appended."""
    assert "NSSA" in normalize_refs("NSSA pension contributions")


def test_expanding_twice_changes_nothing():
    """normalize_refs runs on more than one path; a query that made the
    round trip twice used to carry the expansion twice."""
    once = normalize_refs("NSSA and VAT questions")
    assert normalize_refs(once) == once


def test_a_query_that_spells_it_out_is_left_alone():
    q = "National Social Security Authority NSSA contribution rates"
    assert normalize_refs(q) == q


def test_si_rewriting_still_works_alongside_expansion():
    assert "SI 96 of 2026" in normalize_refs("SI 2026-096 VAT regulations")


def test_every_acronym_expansion_is_distinct():
    assert len(set(ACRONYMS.values())) == len(ACRONYMS)


# ---------------- case-law detection ----------------

@pytest.mark.parametrize("query", [
    "case law on defamation",
    "judgments on unlawful arrest damages",
    "how have the courts interpreted section 56",
    "what has been decided by the courts about spoliation",
])
def test_case_law_questions_are_recognised(query):
    assert Q_CASELAW_RE.search(query)


@pytest.mark.parametrize("query", [
    # one judgment is the reader's own, and this is a procedure question
    "appealing a High Court judgment to the Supreme Court",
    "penalty for assault",
    "quantum of damages for defamation",
])
def test_ordinary_questions_are_not_mistaken_for_case_law(query):
    """The legislation prior is switched off for case-law questions. Do
    that by mistake and a question about statute returns judgments."""
    assert not Q_CASELAW_RE.search(query)


# ---------------- near-duplicate collapse ----------------

# Two councils' copies of one template penalty schedule. The fine
# differs, and the PDF's two columns interleave differently — which is
# why an exact fingerprint doesn't catch them.
BYLAW_A = ("4 Failure to submit a copy of the Environmental Impact 500,00\n"
           "Assessment Report, Environmental Management\n"
           "Plan, or Environmental Impact Assessment Certificate\n"
           "issued by the Environmental Management Agency\nto Council")
BYLAW_B = ("4 Failure to submit a copy of the Environmental Impact 0,00 per\n"
           "Assessment Report, Environmental Management inspection\n"
           "Plan, or Environmental Impact Assessment\n"
           "Certificate issued by the Environmental Management\n"
           "Agency to Council")
OTHER = ("97 Projects for which environmental impact assessment is "
         "required. No developer shall implement a project for which "
         "an environmental impact assessment is required.")


def test_copies_of_the_same_template_are_near_duplicates():
    assert _near_duplicate(_word_set(BYLAW_A), _word_set(BYLAW_B))


def test_a_different_provision_on_the_same_subject_is_not():
    """It shares most of its vocabulary with the by-laws and must still
    survive — it is the provision the reader actually wants."""
    assert not _near_duplicate(_word_set(BYLAW_A), _word_set(OTHER))


def test_empty_text_is_never_a_duplicate():
    assert not _near_duplicate(_word_set(""), _word_set(BYLAW_A))


def test_amounts_do_not_distinguish_copies():
    """Numbers are dropped deliberately: the fine is the only thing that
    changes from one council's schedule to the next."""
    assert _word_set("penalty 500,00 per inspection") == \
        _word_set("penalty 2 750.00 per inspection")


# ---------------- stemming ----------------

def test_plurals_stem_together():
    assert _stem("practitioners") == _stem("practitioner")


def test_short_words_keep_their_s():
    for w in ("gas", "is", "as"):
        assert _stem(w) == w


# ---------------- what the cross-encoder is shown ----------------

def test_the_passage_names_its_instrument():
    """A provision rarely repeats its own title, so on text alone the
    reranker could not tell the Labour Act's section 18 from the five
    collective bargaining SIs that merely grant maternity leave "in
    terms of the Labour Act" — and it ranked all five above it."""
    p = HybridRetriever._passage(
        {"doc_title": "Labour Act [Chapter 28:01]",
         "citation": "Labour Act [Chapter 28:01], s 18 (Maternity leave)",
         "text": "A female employee is entitled to ninety days."})
    assert p.startswith("Labour Act [Chapter 28:01], s 18 (Maternity leave)")
    assert "ninety days" in p


def test_a_chunk_with_no_citation_still_names_its_instrument():
    p = HybridRetriever._passage(
        {"doc_title": "Labour Act", "citation": "", "text": "Some text."})
    assert p == "Labour Act\nSome text."


def test_an_untitled_chunk_is_passed_through_unchanged():
    p = HybridRetriever._passage(
        {"doc_title": "", "citation": "", "text": "Some text."})
    assert p == "Some text."
