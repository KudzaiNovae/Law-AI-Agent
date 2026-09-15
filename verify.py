"""Citation verification: did the answer cite what it actually read?

The worst failure mode in legal RAG is not "I don't know". It is a
fluent, confident answer hung on a section number the model never
retrieved. A reader cannot tell that apart from a correct answer, and
for a free legal-information service used by people who cannot afford a
lawyer, that is the error that actually costs someone something.

So before an answer is finalised, every "section N of <instrument>"
claim in it is checked against the chunks the tools really returned this
turn. Unsupported claims are handed back to the caller, which gives the
model one chance to correct them (see LawAgent._verify_citations).

Deliberate bias: false negatives over false positives. A claim is only
flagged when its instrument clearly resolves to something retrieved and
the section is clearly absent. Wrongly scolding the model about a
correct citation makes the next draft worse, not better, so when this
module is unsure it stays quiet.
"""
import re

# Words that carry no identifying power in an instrument title.
_NOISE = {"the", "of", "and", "a", "an", "act", "chapter", "consolidated",
          "as", "at", "to", "zimbabwe", "regulations", "rules", "no",
          "order", "for", "in", "republic"}

# "section 71 of the Constitution" / "s 12(4) of the Labour Act" — the
# citation form the system prompt asks for, and the only one parsed:
# looser patterns ("the Act, s 4") produce more noise than signal.
# Anchored at "section N of", with the instrument name read from the
# text that follows rather than captured here. Consuming the name inside
# the match would hide any second claim in the same sentence, since
# "section 56 of the Constitution and section 18 of the Labour Act"
# would swallow the whole tail as one phrase.
_ANCHOR_RE = re.compile(
    r"\bs(?:ection)?s?\.?\s*(\d{1,3}[A-Za-z]?)\s*(?:\([^)]{1,8}\))?\s+"
    r"of\s+(?:the\s+)?",
    re.I)
_PHRASE_CHARS = 70

# Words that can sit inside an instrument title without ending it
# ("Mines and Minerals Act", "Companies and Other Business Entities Act").
_CONNECTIVES = {"of", "and", "or", "the", "for", "to", "in", "other"}


def _tokens(text: str) -> set[str]:
    """Identifying tokens of a title or phrase.

    "Statutory Instrument 127 of 2021" and the corpus title
    "SI 2021-127 ..." have to land on the same tokens, so the long forms
    are folded to the abbreviations the catalogue actually uses.
    """
    text = re.sub(r"statutory\s+instruments?", " si ", text, flags=re.I)
    text = re.sub(r"general\s+notices?", " gn ", text, flags=re.I)
    toks = re.findall(r"[a-z0-9]+", text.lower())
    # strip leading zeros so "096" and "96" agree
    toks = [t.lstrip("0") or "0" if t.isdigit() else t for t in toks]
    return {t for t in toks if t not in _NOISE and len(t) > 1}


def _base_section(sec: str) -> str:
    """'12(4)' -> '12'; retrieved chunks are keyed by whole sections."""
    return re.sub(r"\(.*$", "", sec).strip().lower()


def retrieved_index(chunks: list[dict]) -> dict[str, set[str]]:
    """Map each retrieved instrument title to the sections seen from it."""
    idx: dict[str, set[str]] = {}
    for c in chunks:
        title = (c.get("doc_title") or "").strip()
        if not title:
            continue
        sec = _base_section(str(c.get("section") or ""))
        idx.setdefault(title, set())
        if sec:
            idx[title].add(sec)
    return idx


def _resolve(phrase: str, index: dict[str, set[str]]) -> list[str]:
    """Retrieved titles the phrase plausibly names.

    Numbers must all be present ("SI 127 of 2021" must not resolve to
    SI 128), and at least one distinctive word must be shared.
    """
    p = _tokens(phrase)
    if not p:
        return []
    nums = {t for t in p if t.isdigit()}
    words = p - nums
    out = []
    for title in index:
        toks = _tokens(title)
        if nums and not nums <= toks:
            continue
        if words and not (words & toks):
            continue
        out.append(title)
    return out


def _trim_phrase(phrase: str) -> str:
    """Cut a captured phrase back to the title it starts with.

    The regex grabs everything up to the next clause boundary, so
    "the Constitution protects property rights" arrives whole. Titles are
    capitalised, numeric or bracketed, so stop at the first lowercase
    word that isn't a connective — otherwise stray numbers from the
    surrounding prose ("section 4 of the Act requires 30 days") get
    mistaken for part of the instrument's name.
    """
    words = phrase.split()
    kept: list[str] = []
    for w in words:
        core = w.strip("'\"*_()[],.;:")
        if not core:
            break
        low = core.lower()
        if low in _CONNECTIVES or core[0].isupper() or core[0].isdigit() \
                or core[0] in "[(":
            kept.append(w)
            continue
        break
    while kept and kept[-1].strip("'\"*_()[],.;:").lower() in _CONNECTIVES:
        kept.pop()
    return " ".join(kept).strip(" '\"*_,.;:")


def claims(answer: str) -> list[tuple[str, str]]:
    """(instrument phrase, section) pairs asserted by the answer.

    Phrases with no identifying tokens left after trimming are dropped:
    "section 4 of the Act" names nothing checkable, and guessing would
    only manufacture false alarms.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for m in _ANCHOR_RE.finditer(answer):
        sec = m.group(1)
        tail = answer[m.end():m.end() + _PHRASE_CHARS]
        tail = re.split(r"[.,;:\n)]", tail)[0]
        phrase = _trim_phrase(tail.strip(" '\"*_"))
        if not phrase or not _tokens(phrase):
            continue
        key = (phrase.lower(), sec.lower())
        if key not in seen:
            seen.add(key)
            out.append((phrase, sec))
    return out


def check(answer: str, chunks: list[dict]) -> list[str]:
    """Citation problems in `answer`, as sentences the model can act on.

    Empty list means "nothing provably wrong" — which is not the same as
    "everything is right", since only section-level claims are checked.
    """
    if not answer or not chunks:
        return []
    index = retrieved_index(chunks)
    if not index:
        return []
    problems: list[str] = []
    for phrase, sec in claims(answer):
        titles = _resolve(phrase, index)
        if not titles:
            problems.append(
                f'You cite "section {sec} of {phrase}", but no tool result '
                f"this turn came from an instrument by that name.")
            continue
        # Only complain about a missing section when we actually know
        # which sections were retrieved from that instrument.
        known = set()
        for t in titles:
            known |= index[t]
        if not known:
            continue
        if _base_section(sec) not in known:
            shown = ", ".join(sorted(known, key=_sortable)[:12])
            problems.append(
                f'You cite "section {sec} of {phrase}", but the sections '
                f"retrieved from {titles[0]} were: {shown}.")
    return problems


def _sortable(s: str):
    m = re.match(r"(\d+)", s)
    return (int(m.group(1)) if m else 9999, s)
