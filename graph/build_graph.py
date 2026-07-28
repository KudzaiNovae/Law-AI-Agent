"""Build the legal knowledge graph: which instrument amends/repeals which.

Edges are extracted two ways and merged:
  1. Title parser  — "X Amendment Act 2023" amends "X Act". Free, instant,
     near-100% precision, but no section detail and misses repeals that
     are buried in the text.
  2. LLM pass      — Gemini reads each amending instrument's opening text
     and returns the exact target(s), action and affected sections.
     Cached per source file (keyed by a text hash), so reruns only hit
     the API for new instruments.

Output: data/processed/graph.jsonl, one edge per line:
  {"source": ..., "relation": "amends|repeals|abolishes",
   "target": <resolved corpus title or null>, "target_text": <raw name>,
   "sections": [...], "year": int, "method": "title"|"llm"}

Usage:
    python -m graph.build_graph            # title parser + LLM
    python -m graph.build_graph --no-llm   # title parser only (no API)
"""
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import os

from config import (PROCESSED_DIR, GEMINI_API_KEY, GEMINI_MODEL,
                    GEMINI_BASE_URL, ANTHROPIC_API_KEY)

# Anthropic is preferred when a key is set: Haiku is cheap, has no
# free-tier daily cap, and extraction is a one-off per instrument.
GRAPH_MODEL = os.getenv("GRAPH_MODEL", "claude-haiku-4-5")
from index.retriever import STOPWORDS, _tokenize, YEAR_RE

GRAPH_PATH = PROCESSED_DIR / "graph.jsonl"
CACHE_DIR = PROCESSED_DIR / "graph_cache"

AMEND_WORDS = ("amendment", "abolition", "repeal")
# words to strip from an amending title to recover its target's title
STRIP_WORDS = STOPWORDS | {"amendment", "abolition", "repeal", "chapter",
                           "consolidated", "as", "at", "zimbabwe", "bill"}
# doc types that can carry edges (bills in gazettes are proposals, not law)
EDGE_TYPES = {"act", "si", "gn"}
MAX_LLM_CHARS = 7000

EXTRACT_PROMPT = """\
You extract amendment relationships from Zimbabwean legislation.
Given the opening text of an amending instrument, identify every
instrument it amends, repeals or abolishes. Look for phrases like
"the principal Act", "is amended by", "is repealed", short-title
citations and Chapter numbers.

Return ONLY strict JSON, no prose:
{"edges": [{"target": "<full name of the instrument affected>",
            "action": "amends" | "repeals" | "abolishes",
            "sections": ["<section numbers affected, if stated>"]}]}
If the text names no target, return {"edges": []}."""


def _stem(t: str) -> str:
    return t[:-1] if t.endswith("s") and len(t) > 3 else t


def _sig(title: str) -> set[str]:
    return {_stem(t) for t in _tokenize(title) if not t.isdigit()} - STRIP_WORDS


def load_instruments() -> list[dict]:
    """Distinct instruments with concatenated text, from chunks.jsonl."""
    by_title: dict[str, dict] = {}
    for line in (PROCESSED_DIR / "chunks.jsonl").read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t = r["doc_title"]
        inst = by_title.setdefault(t, {
            "title": t,
            "doc_type": r.get("doc_type", "policy"),
            "source_file": r.get("source_file", ""),
            "year": None,
            "text": "",
        })
        if len(inst["text"]) < MAX_LLM_CHARS:
            inst["text"] += "\n" + r.get("text", "")
    for inst in by_title.values():
        m = YEAR_RE.search(inst["title"])
        inst["year"] = int(m.group()) if m else None
    return list(by_title.values())


def is_amending(inst: dict) -> bool:
    tl = inst["title"].lower()
    if inst["doc_type"] not in EDGE_TYPES:
        return False
    if "no. 20" in tl or "no 20" in tl:   # the 2013 Constitution itself
        return False
    return any(w in tl for w in AMEND_WORDS)


class Resolver:
    """Map a raw instrument name to the best-matching corpus title."""

    def __init__(self, instruments: list[dict]):
        self.instruments = instruments
        self._sigs = [(_sig(i["title"]), i) for i in instruments]

    def resolve(self, name: str, exclude_title: str | None = None,
                skip_amending: bool = False) -> str | None:
        q = _sig(name)
        if not q:
            return None
        best, best_score = None, 0.0
        for tset, inst in self._sigs:
            if not tset or inst["title"] == exclude_title:
                continue
            is_amend = any(w in inst["title"].lower() for w in AMEND_WORDS)
            if skip_amending and is_amend:
                continue
            overlap = len(q & tset)
            if overlap < min(2, len(q)):
                continue
            score = overlap + overlap / len(tset)
            # principal (non-amending) texts are the natural targets
            if is_amend:
                score -= 0.5
            if score > best_score:
                best, best_score = inst["title"], score
        return best


def title_edges(amending: list[dict], resolver: Resolver) -> list[dict]:
    out = []
    for inst in amending:
        tl = inst["title"].lower()
        relation = ("abolishes" if "abolition" in tl
                    else "repeals" if "repeal" in tl else "amends")
        # only principal (non-amending) targets: prevents duplicate copies
        # of the same amendment act pointing at each other
        target = resolver.resolve(inst["title"], exclude_title=inst["title"],
                                  skip_amending=True)
        if not target:
            continue
        out.append({"source": inst["title"], "relation": relation,
                    "target": target, "target_text": target,
                    "sections": [], "year": inst["year"], "method": "title"})
    return out


def _llm_client():
    """Returns (provider, client). Anthropic if a key is set, else Gemini."""
    if ANTHROPIC_API_KEY:
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    from openai import OpenAI
    return "gemini", OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


def _extract_json(s: str) -> dict:
    """Parse the model's JSON, tolerating code fences / surrounding prose."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group())
            except ValueError:
                pass
    return {}


def llm_edges_for(inst: dict, provider: str, client,
                  resolver: Resolver) -> list[dict]:
    text = inst["text"][:MAX_LLM_CHARS]
    h = hashlib.sha1(text.encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{inst['source_file'] or inst['title'][:60]}.json"
    if cache.exists():
        try:
            c = json.loads(cache.read_text("utf-8"))
            if c.get("hash") == h:
                return c["edges"]
        except ValueError:
            pass

    user_msg = f"Instrument: {inst['title']}\n\n{text}"
    if provider == "anthropic":
        resp = client.messages.create(
            model=GRAPH_MODEL, max_tokens=1000,
            system=EXTRACT_PROMPT,
            messages=[{"role": "user", "content": user_msg}])
        raw = _extract_json("".join(b.text for b in resp.content
                                    if b.type == "text"))
        time.sleep(1.2)   # ~50 requests/minute
    else:
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{"role": "system", "content": EXTRACT_PROMPT},
                      {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"})
        raw = _extract_json(resp.choices[0].message.content or "{}")
        time.sleep(4)   # free tier: stay under 15 requests/minute
    raw_edges = raw.get("edges", []) if isinstance(raw, dict) else []

    edges = []
    for e in raw_edges:
        tname = str(e.get("target", "")).strip()
        action = str(e.get("action", "amends")).lower()
        if not tname or action not in ("amends", "repeals", "abolishes"):
            continue
        target = resolver.resolve(tname, exclude_title=inst["title"])
        edges.append({
            "source": inst["title"], "relation": action,
            "target": target, "target_text": tname,
            "sections": [str(s) for s in (e.get("sections") or [])][:20],
            "year": inst["year"], "method": "llm"})

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"hash": h, "edges": edges},
                                ensure_ascii=False), "utf-8")
    return edges


def merge(title_e: list[dict], llm_e: list[dict]) -> list[dict]:
    """LLM edges (richer) win over title edges for the same pair."""
    seen: dict[tuple, dict] = {}
    for e in title_e + llm_e:   # llm later -> overwrites
        key = (e["source"], e["relation"], e["target"] or e["target_text"])
        seen[key] = e
    return sorted(seen.values(),
                  key=lambda e: (e["year"] or 0, e["source"]), reverse=True)


def main() -> None:
    use_llm = "--no-llm" not in sys.argv
    instruments = load_instruments()
    resolver = Resolver(instruments)
    amending = [i for i in instruments if is_amending(i)]
    print(f"{len(instruments)} instruments, {len(amending)} amending")

    t_edges = title_edges(amending, resolver)
    print(f"title parser: {len(t_edges)} edges")

    l_edges: list[dict] = []
    if use_llm:
        provider, client = _llm_client()
        print(f"llm provider: {provider} "
              f"({GRAPH_MODEL if provider == 'anthropic' else GEMINI_MODEL})")
        for n, inst in enumerate(amending, 1):
            got = None
            for attempt in range(4):
                try:
                    got = llm_edges_for(inst, provider, client, resolver)
                    break
                except Exception as exc:
                    if "429" in str(exc):   # free tier: 15 req/min
                        time.sleep(35)
                        continue
                    print(f"  ! {inst['title'][:60]}: {exc}")
                    time.sleep(10)
                    break
            if got is None:
                continue
            l_edges.extend(got)
            if n % 25 == 0:
                print(f"  llm {n}/{len(amending)} ({len(l_edges)} edges)",
                      flush=True)
        print(f"llm: {len(l_edges)} edges")

    edges = merge(t_edges, l_edges)
    with GRAPH_PATH.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    resolved = sum(1 for e in edges if e["target"])
    print(f"wrote {len(edges)} edges ({resolved} resolved to corpus titles) "
          f"-> {GRAPH_PATH}")


if __name__ == "__main__":
    main()
