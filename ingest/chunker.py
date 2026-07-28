"""Section-aware chunking for Zimbabwean legal text.

Legal documents have natural retrieval units: numbered sections.
We split on section headings and keep chapter/part context in metadata,
so every chunk can be cited as "Constitution, s 56" or "SI 127/2021, s 4".

Chunks that exceed MAX_CHARS are split on subsection boundaries
("(1)", "(2)", ...) with the section heading repeated for context.
"""
import re
from dataclasses import dataclass, field, asdict

MAX_CHARS = 4000

# "56. Equality and non-discrimination" or "56 Equality..." at line start
SECTION_RE = re.compile(r"^(\d{1,3}[A-Z]?)\.?\s+([A-Z][^\n]{3,120})$", re.M)
CHAPTER_RE = re.compile(r"^\s*(CHAPTER\s+\d+[A-Z]?)\s*$", re.M | re.I)
PART_RE = re.compile(r"^\s*(PART\s+[IVXLC\d]+[^\n]*)$", re.M)
SUBSECTION_RE = re.compile(r"^\(\d+[a-z]?\)", re.M)


@dataclass
class Chunk:
    text: str
    doc_title: str
    doc_type: str          # constitution | act | si | policy
    section: str = ""      # e.g. "56"
    section_title: str = ""
    chapter: str = ""
    part: str = ""
    source_file: str = ""

    def citation(self) -> str:
        bits = [self.doc_title]
        if self.section:
            bits.append(f"s {self.section}")
        if self.section_title:
            bits.append(f"({self.section_title})")
        return ", ".join(bits[:2]) + (f" {bits[2]}" if len(bits) > 2 else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation()
        return d


def chunk_document(text: str, doc_title: str, doc_type: str,
                   source_file: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    matches = list(SECTION_RE.finditer(text))

    if not matches:
        # No detectable sections (e.g. a policy PDF) — fall back to
        # paragraph-window chunking.
        return _fallback_chunks(text, doc_title, doc_type, source_file)

    # Preamble before the first section (long title, preamble, etc.)
    preamble = text[: matches[0].start()].strip()
    if len(preamble) > 100:
        chunks.extend(_sized(preamble, doc_title, doc_type, source_file,
                             section="", section_title="Preamble",
                             chapter="", part=""))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chapter = _last_before(CHAPTER_RE, text, start)
        part = _last_before(PART_RE, text, start)
        chunks.extend(_sized(body, doc_title, doc_type, source_file,
                             section=m.group(1),
                             section_title=m.group(2).strip(),
                             chapter=chapter, part=part))
    return chunks


def _last_before(pattern: re.Pattern, text: str, pos: int) -> str:
    last = ""
    for m in pattern.finditer(text, 0, pos):
        last = m.group(1).strip()
    return last


def _sized(body: str, doc_title: str, doc_type: str, source_file: str,
           **meta) -> list[Chunk]:
    """Split an oversized section on subsection boundaries."""
    if len(body) <= MAX_CHARS:
        return [Chunk(text=body, doc_title=doc_title, doc_type=doc_type,
                      source_file=source_file, **meta)]
    header = body.splitlines()[0]
    parts, buf = [], ""
    for line in body.splitlines(keepends=True):
        if SUBSECTION_RE.match(line.strip()) and len(buf) > MAX_CHARS // 2:
            parts.append(buf)
            buf = header + " (cont.)\n"
        buf += line
        if len(buf) > MAX_CHARS:          # hard stop for wall-of-text sections
            parts.append(buf)
            buf = header + " (cont.)\n"
    if buf.strip():
        parts.append(buf)
    return [Chunk(text=p.strip(), doc_title=doc_title, doc_type=doc_type,
                  source_file=source_file, **meta) for p in parts]


def _fallback_chunks(text: str, doc_title: str, doc_type: str,
                     source_file: str) -> list[Chunk]:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > MAX_CHARS and buf:
            chunks.append(buf)
            buf = ""
        buf += p + "\n\n"
    if buf.strip():
        chunks.append(buf)
    return [Chunk(text=c.strip(), doc_title=doc_title, doc_type=doc_type,
                  source_file=source_file) for c in chunks]
