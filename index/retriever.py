"""Hybrid retrieval: Chroma vector search + BM25 keyword search,
merged with reciprocal rank fusion (RRF).

Legal queries often contain exact tokens ("SI 127 of 2021", "section 71")
that pure vector search misses — BM25 catches those.
"""
import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROCESSED_DIR, CHROMA_DIR, COLLECTION_NAME,
                    TOP_K, VECTOR_K, BM25_K)
from index.build_index import chunk_id

RRF_K = 60
MIN_CHUNK_CHARS = 120   # drop TOC lines / bare headings from results

# Cross-encoder reranker (ONNX via fastembed, CPU, ~10ms per pair).
# Reranking the fused candidates before the top_k cut measurably lifts
# precision over RRF scores alone. RERANK=0 disables; loads lazily and
# fails soft (missing package / model download failure = no rerank).
import os
RERANK = os.getenv("RERANK", "1") != "0"
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
_reranker = None          # None = not tried, False = unavailable


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _reranker = TextCrossEncoder(RERANK_MODEL)
        except Exception as e:
            print(f"[retriever] reranker unavailable ({e}); "
                  "falling back to RRF order")
            _reranker = False
    return _reranker or None
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Query understanding: exact references beat semantic similarity.
Q_INSTR_RE = re.compile(
    r"\b(?:s\.?\s?i\.?|statutory\s+instrument|act\s*(?:no\.?)?|"
    r"general\s+notice|g\.?n\.?)\s*(\d{1,4})[a-z]?\s*(?:of|/)\s*"
    r"((?:19|20)\d{2})", re.I)
Q_SECTION_RE = re.compile(r"\bs(?:ection)?\.?\s+(\d{1,3}[A-Za-z]?)\b", re.I)

# Catalogue display format: "SI 2026-096", "GN 2023-1880", "Act 2026-6"
# (year first, zero-padded number). Rewrite to the canonical "SI 96 of
# 2026" so Q_INSTR_RE and BM25 can work with it.
Q_INSTR_YEARFIRST_RE = re.compile(
    r"\b(s\.?i\.?|g\.?n\.?|act)\s*((?:19|20)\d{2})[-_/\s]0*(\d{1,4})([a-z]?)\b",
    re.I)


# Common abbreviations users type -> corpus vocabulary.
# "CAB 3" = Constitutional Amendment Bill No. 3.
Q_CAB_RE = re.compile(r"\bC\.?A\.?B\.?\s*(?:no\.?\s*)?(\d{1,2})\b", re.I)


def normalize_refs(query: str) -> str:
    """'SI 2026-096 ...' -> 'SI 96 of 2026 ...' (idempotent otherwise).
    Also expands abbreviations: 'CAB 3' -> the Bill's full name."""
    query = Q_CAB_RE.sub(
        lambda m: f"Constitution of Zimbabwe Amendment Bill No {m.group(1)}",
        query)
    return Q_INSTR_YEARFIRST_RE.sub(
        lambda m: f"{m.group(1).rstrip('.').upper()} {int(m.group(3))}"
                  f"{m.group(4)} of {m.group(2)}", query)

# Boost sizes relative to RRF scores (one rank-1 hit ~ 1/61 ~ 0.016)
BOOST_INSTRUMENT = 0.05   # query names the exact instrument
BOOST_SECTION = 0.02      # query names a section number
BOOST_TITLE = 0.008       # query words overlap the instrument title

STOPWORDS = {"the", "of", "and", "act", "regulations", "amendment", "no",
             "notice", "order", "rules", "for", "in", "to", "a", "an"}


def _tokenize(text: str) -> list[str]:
    # strip leading zeros from numbers so "096" matches "96"
    return [t.lstrip("0") or "0" if t.isdigit() else t
            for t in re.findall(r"[a-z0-9]+", text.lower())]


class HybridRetriever:
    def __init__(self):
        self._load()

    def _load(self):
        """(Re)load index from disk. Called again if chunks.jsonl changes,
        so a long-running server picks up re-indexes without a restart."""
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = client.get_collection(COLLECTION_NAME)

        chunks_path = PROCESSED_DIR / "chunks.jsonl"
        self._mtime = chunks_path.stat().st_mtime
        self.records = []
        for l in chunks_path.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                self.records.append(json.loads(l))
            except json.JSONDecodeError:
                continue  # skip corrupt lines; build_index skips identically
        self.bm25 = BM25Okapi([_tokenize(r["text"]) for r in self.records])

        # Chroma ids are stable content hashes (see build_index.chunk_id);
        # map them back to record positions for RRF merging.
        pos: dict[str, int] = {}
        self._id2idx: dict[str, int] = {}
        for i, r in enumerate(self.records):
            sf = r.get("source_file", "")
            self._id2idx[chunk_id(r, pos.get(sf, 0))] = i
            pos[sf] = pos.get(sf, 0) + 1

        # Knowledge graph: who amends/repeals whom (graph/build_graph.py).
        # Optional; related_updates falls back to a heuristic without it.
        self.graph_in: dict[str, list[dict]] = {}
        self.graph_out: dict[str, list[dict]] = {}
        gp = PROCESSED_DIR / "graph.jsonl"
        if gp.exists():
            for l in gp.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if e.get("target"):
                    self.graph_in.setdefault(e["target"], []).append(e)
                self.graph_out.setdefault(e["source"], []).append(e)

        # Catalog of distinct instruments (for list_legislation tool)
        seen: set[str] = set()
        self.instruments: list[dict] = []
        for r in self.records:
            title = r["doc_title"]
            if title in seen:
                continue
            seen.add(title)
            m = YEAR_RE.search(title)
            self.instruments.append({
                "title": title,
                "doc_type": r.get("doc_type", "policy"),
                "year": int(m.group()) if m else None,
            })

    def _maybe_reload(self):
        try:
            mtime = (PROCESSED_DIR / "chunks.jsonl").stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            self._load()

    def catalog(self, doc_type: str | None = None,
                year: int | None = None) -> list[dict]:
        """Distinct instruments in the corpus, newest first."""
        self._maybe_reload()
        out = [i for i in self.instruments
               if (not doc_type or i["doc_type"] == doc_type)
               and (not year or i["year"] == year)]
        out.sort(key=lambda i: (i["year"] or 0, i["title"]), reverse=True)
        return out

    def _best_title(self, name: str,
                    prefer_principal: bool = False) -> str | None:
        """Corpus title best matching `name` (same scoring as before).
        prefer_principal: penalize amendment/repeal instruments so that
        'Labour Act' resolves to the principal Act, not its amendments."""
        qset = set(_tokenize(normalize_refs(name)))
        best_title, best_score = None, 0.0
        for inst in self.instruments:
            tset = set(_tokenize(inst["title"]))
            if not tset:
                continue
            overlap = len(qset & tset)
            # favour tight titles over long gazette titles with same overlap
            score = overlap + overlap / len(tset)
            if prefer_principal and any(
                    w in inst["title"].lower()
                    for w in ("amendment", "abolition", "repeal")):
                score -= 0.5
            if score > best_score:
                best_title, best_score = inst["title"], score
        return best_title if best_title and best_score >= 2 else None

    def get_instrument(self, name: str, max_chars: int = 16000) -> list[dict]:
        """All chunks of the single instrument best matching `name`, in
        document order, capped at max_chars."""
        self._maybe_reload()
        best_title = self._best_title(name)
        if not best_title:
            return []
        out, used = [], 0
        for r in self.records:
            if r["doc_title"] != best_title:
                continue
            out.append(r)
            used += len(r.get("text", ""))
            if used > max_chars:
                break
        return out

    def _sig(self, title: str) -> set[str]:
        """Stemmed content words of a title (matches graph/build_graph)."""
        def stem(t: str) -> str:
            return t[:-1] if t.endswith("s") and len(t) > 3 else t
        return {stem(t) for t in _tokenize(title)
                if not t.isdigit()} - STOPWORDS - {
                    "chapter", "consolidated", "as", "at", "zimbabwe",
                    "amendment", "abolition", "repeal", "bill"}

    def related_updates(self, title: str,
                        after_year: int | None = None) -> list[str]:
        """Instruments that amend, repeal or abolish `title`.

        Graph first (exact edges from graph/build_graph.py), falling back
        to the title-word overlap heuristic when the graph has no entry.
        Used to warn the agent that a consolidated text may be stale."""
        # --- knowledge graph leg ---
        edges = self.graph_in.get(title)
        if edges is None and self.graph_in:
            qsig = self._sig(title)
            if qsig:
                edges = []
                for key, es in self.graph_in.items():
                    ksig = self._sig(key)
                    if ksig and len(qsig & ksig) >= min(2, len(qsig),
                                                        len(ksig)):
                        edges.extend(es)
        if edges:
            seen_e: set[str] = set()
            titles = []
            for e in sorted(edges, key=lambda e: e.get("year") or 0,
                            reverse=True):
                if after_year and e.get("year") and e["year"] < after_year:
                    continue
                if e["source"] == title or e["source"] in seen_e:
                    continue
                seen_e.add(e["source"])
                titles.append(e["source"])
            if titles:
                return titles[:8]

        # --- heuristic fallback ---
        def stem(t: str) -> str:
            return t[:-1] if t.endswith("s") and len(t) > 3 else t
        base = {stem(t) for t in _tokenize(title)
                if not t.isdigit()} - STOPWORDS - {
                    "chapter", "consolidated", "as", "at", "zimbabwe"}
        if not base:
            return []
        out = []
        for inst in self.instruments:
            tl = inst["title"].lower()
            if not any(w in tl for w in
                       ("amendment", "abolition", "repeal")):
                continue
            if inst["title"] == title:
                continue
            if after_year and inst["year"] and inst["year"] < after_year:
                continue
            tset = {stem(t) for t in _tokenize(inst["title"])
                    if not t.isdigit()} - STOPWORDS
            if len(base & tset) >= min(2, len(base)):
                out.append(inst["title"])
        out.sort(key=lambda t: (YEAR_RE.search(t) and
                                int(YEAR_RE.search(t).group())) or 0,
                 reverse=True)
        return out[:8]

    def amendment_history(self, name: str) -> dict:
        """Knowledge-graph view of one instrument: what amends/repeals it
        (amended_by) and what it amends/repeals (amends), newest first."""
        self._maybe_reload()
        title = self._best_title(name, prefer_principal=True)
        if not title:
            return {"error": "No instrument matching that name in the "
                             "corpus."}

        def fmt(e: dict, other: str) -> dict:
            d = {"instrument": e[other] or e["target_text"],
                 "relation": e["relation"], "year": e.get("year")}
            if e.get("sections"):
                d["sections_affected"] = e["sections"]
            return d

        newest = lambda es: sorted(es, key=lambda e: e.get("year") or 0,
                                   reverse=True)
        in_edges = self.graph_in.get(title, [])
        if not in_edges:
            # the corpus may hold duplicate copies of the same Act whose
            # edges resolved to the other copy: match by title signature
            qsig = self._sig(title)
            if qsig:
                for key, es in self.graph_in.items():
                    ksig = self._sig(key)
                    if ksig and len(qsig & ksig) >= min(2, len(qsig),
                                                        len(ksig)):
                        in_edges = in_edges + es
        seen_src: set[str] = set()
        amended_by = []
        for e in newest(in_edges):
            if e["source"] in seen_src or e["source"] == title:
                continue
            seen_src.add(e["source"])
            amended_by.append(fmt(e, "source"))
        amends = [fmt(e, "target")
                  for e in newest(self.graph_out.get(title, []))]
        out: dict = {"instrument": title,
                     "amended_or_repealed_by": amended_by,
                     "amends_or_repeals": amends}
        if not amended_by and not amends:
            out["note"] = ("No amendment relationships recorded for this "
                           "instrument in the graph. Absence of an edge "
                           "does not guarantee the text is current; "
                           "search for amendments to confirm.")
        return out

    def _apply_boosts(self, query: str, scores: dict[int, float]) -> None:
        """Exact references in the query override fuzzy similarity.

        1. "SI 107 of 2026" / "Act 3 of 2026": pull in every chunk of the
           named instrument, strongly boosted.
        2. "section 71": boost candidate chunks whose citation is s 71.
        3. Title words: lightly boost candidates from instruments whose
           title shares 2+ meaningful words with the query.
        """
        # 1. named instruments
        refs = [(int(n), y) for n, y in Q_INSTR_RE.findall(query)]
        if refs:
            for idx, r in enumerate(self.records):
                title_l = r.get("doc_title", "").lower()
                t_nums = [int(x) for x in re.findall(r"\d{1,4}", title_l)]
                for num, yr in refs:
                    if int(yr) in t_nums and num in t_nums:
                        scores[idx] = scores.get(idx, 0) + BOOST_INSTRUMENT
                        break

        # 2. named sections (only among current candidates)
        secs = {s.lower() for s in Q_SECTION_RE.findall(query)}
        # avoid treating instrument numbers ("SI 107") as section numbers
        secs -= {str(n) for n, _ in refs}
        if secs:
            for idx in list(scores):
                cite = self.records[idx].get("citation", "").lower()
                if any(re.search(rf"\bs\s+{re.escape(s)}\b", cite)
                       for s in secs):
                    scores[idx] += BOOST_SECTION

        # 3. title-word overlap (only among current candidates)
        q_words = set(_tokenize(query)) - STOPWORDS
        if len(q_words) >= 2:
            for idx in list(scores):
                t_words = set(_tokenize(
                    self.records[idx].get("doc_title", ""))) - STOPWORDS
                if len(q_words & t_words) >= 2:
                    scores[idx] += BOOST_TITLE

    def search(self, query: str, top_k: int = TOP_K,
               doc_type: str | None = None,
               year: int | None = None) -> list[dict]:
        self._maybe_reload()
        query = normalize_refs(query)
        scores: dict[int, float] = {}

        # Vector leg
        where = {"doc_type": doc_type} if doc_type else None
        res = self.col.query(query_texts=[query], n_results=VECTOR_K,
                             where=where)
        for rank, cid in enumerate(res["ids"][0]):
            idx = self._id2idx.get(cid)
            if idx is None:
                continue  # Chroma and chunks.jsonl out of sync mid-reindex
            scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        # BM25 leg
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)
        added = 0
        for rank, idx in enumerate(bm25_ranked):
            if bm25_scores[idx] <= 0 or added >= BM25_K:
                break
            if doc_type and self.records[idx].get("doc_type") != doc_type:
                continue
            scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
            added += 1

        # Precision boosts from exact references in the query
        self._apply_boosts(query, scores)

        # Rank, then dedupe: one chunk per citation, skip TOC-line chunks.
        # Collect ~3x top_k candidates so the reranker has room to work.
        ranked = sorted(scores, key=scores.get, reverse=True)
        cand_cap = max(top_k * 3, 24) if RERANK else top_k
        out: list[dict] = []
        seen_cites: set[str] = set()
        for i in ranked:
            r = self.records[i]
            if len(r.get("text", "")) < MIN_CHUNK_CHARS:
                continue
            if year and str(year) not in r.get("doc_title", ""):
                continue
            cite = r.get("citation", "") or f"{r['doc_title']}#{i}"
            if cite in seen_cites:
                continue
            seen_cites.add(cite)
            # copy with the hybrid score so callers can judge relevance
            out.append({**r, "score": round(scores[i], 4)})
            if len(out) >= cand_cap:
                break

        # Cross-encoder rerank of the candidate pool. When the query
        # holds an exact reference ("SI 127 of 2021", "section 71") the
        # boosted RRF #1 is almost always right, so pin it first and let
        # the reranker order the rest.
        rr = _get_reranker() if RERANK and len(out) > top_k else None
        if rr:
            try:
                docs = [(r.get("doc_title", "") + " " +
                         (r.get("citation") or "") + " " +
                         r["text"][:1200]) for r in out]
                ce = list(rr.rerank(query, docs))
                pin = (1 if (Q_INSTR_RE.search(query)
                             or Q_SECTION_RE.search(query)) else 0)
                order = list(range(pin)) + sorted(
                    range(pin, len(out)), key=lambda i: ce[i],
                    reverse=True)
                out = [out[i] for i in order]
            except Exception as e:
                print(f"[retriever] rerank failed ({e}); using RRF order")
        return out[:top_k]
