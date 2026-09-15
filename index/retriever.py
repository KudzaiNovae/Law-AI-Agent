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
                    TOP_K, VECTOR_K, BM25_K, RERANK_CANDIDATES,
                    RERANK_VECTOR_K, RERANK_BM25_K)
from index import rerank
from index.build_index import chunk_id

RRF_K = 60
MIN_CHUNK_CHARS = 120   # drop TOC lines / bare headings from results
MAX_PER_INSTRUMENT = 2  # per-instrument cap when fusing sub-queries
# Council by-laws are made from a common template, so sixty-odd SIs carry
# the same penalty schedule word for word, differing only in the fine.
# They are separate instruments, so the per-instrument cap never fires,
# and they fill the whole result set: "environmental impact assessment
# certificate" returned the same sentence from five different councils
# and never reached the Environmental Management Act.
MAX_PER_TEMPLATE = 2
DUPLICATE_JACCARD = 0.8
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

# Everyone here says "NSSA", nobody says "National Social Security
# Authority" — but the corpus is drafted in the long form, so the
# acronym matches nothing at all in either the vector space or BM25.
# Only acronyms whose expansion actually names instruments in this
# corpus are listed; expanding to words that appear nowhere would just
# dilute the query.
ACRONYMS = {
    "nssa": "National Social Security Authority",
    "zimra": "Zimbabwe Revenue Authority",
    "ema": "Environmental Management Agency",
    "potraz": "Postal and Telecommunications Regulatory Authority",
    "zacc": "Zimbabwe Anti-Corruption Commission",
    "rbz": "Reserve Bank of Zimbabwe",
    "zera": "Zimbabwe Energy Regulatory Authority",
    "zec": "Zimbabwe Electoral Commission",
    "zimsec": "Zimbabwe Schools Examinations Council",
    "zrp": "Zimbabwe Republic Police",
    "vat": "Value Added Tax",
    "paye": "Pay As You Earn employment income tax",
}
Q_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(sorted(ACRONYMS, key=len, reverse=True)) + r")\b", re.I)


def normalize_refs(query: str) -> str:
    """'SI 2026-096 ...' -> 'SI 96 of 2026 ...' (idempotent otherwise).
    Also expands abbreviations: 'CAB 3' -> the Bill's full name."""
    query = Q_CAB_RE.sub(
        lambda m: f"Constitution of Zimbabwe Amendment Bill No {m.group(1)}",
        query)
    # Appended, not substituted: a handful of instruments are titled by
    # the acronym itself, and "vat"/"ema" have innocent English senses.
    # Keeping both forms costs a few tokens and can only add recall.
    low = query.lower()
    seen: set[str] = set()
    extra = [ACRONYMS[a] for a in
             (m.group(1).lower() for m in Q_ACRONYM_RE.finditer(query))
             # already spelled out (or already expanded once — this
             # function is applied more than once on some paths)
             if not (a in seen or seen.add(a)
                     or ACRONYMS[a].lower() in low)]
    if extra:
        query = query + " " + " ".join(extra)
    return Q_INSTR_YEARFIRST_RE.sub(
        lambda m: f"{m.group(1).rstrip('.').upper()} {int(m.group(3))}"
                  f"{m.group(4)} of {m.group(2)}", query)

# Boost sizes relative to RRF scores (one rank-1 hit ~ 1/61 ~ 0.016)
BOOST_INSTRUMENT = 0.05   # query names the exact instrument
BOOST_SECTION = 0.02      # query names a section number
BOOST_TITLE = 0.008       # query words overlap the instrument title
# ...counting only words that name a subject. A word in more than this
# share of instrument titles is structure, not subject: see _load.
TITLE_DF_MAX = 0.05
# "What does the law say?" is a question about the law, and a third of
# this corpus is judgments — which quote the statute at length and so
# match the wording of a plain-language question better than the
# statute's own terse drafting does. Without a prior, "penalty for
# assault" returns five assault convictions and not the Criminal Law
# Code. Small: enough to lift the instrument past judgments of similar
# relevance, not enough to bury a judgment that is squarely on point.
BOOST_LEGISLATION = 0.02
LEGISLATION_TYPES = {"constitution", "act", "si", "gn"}
# ...unless the reader is asking for case law, in which case the
# judgments are the answer. (The UI's case-law mode instead tells the
# model to pass doc_type="judgment", which skips the prior outright.)
_SAID = r"(?:held|said|says?|stated|found|decided|interpreted|ruled|awarded)"
# Plural only: one "judgment" is usually the reader's own — "appealing a
# High Court judgment" is a question about procedure, and treating it as
# a request for case law buried the Supreme Court Act under the judgments
# of courts that happened to use the word.
Q_CASELAW_RE = re.compile(
    r"\b(?:case[\s-]?law|judgments|judgements|precedents?|"
    rf"decided\s+cases?|authorities|courts?\s+(?:\w+\s+){{0,2}}{_SAID}|"
    rf"{_SAID}\s+by\s+the\s+courts?)\b", re.I)
# Cross-encoder scores are 0..1, ~60x the RRF scale the boosts were tuned
# against. Rescale so an exact instrument match still dominates there.
BOOST_RERANK_SCALE = 10.0

STOPWORDS = {"the", "of", "and", "act", "regulations", "amendment", "no",
             "notice", "order", "rules", "for", "in", "to", "a", "an"}


def _word_set(text: str) -> frozenset[str]:
    """Words of a chunk, for near-duplicate detection. Numbers are left
    out: the fine amount is the only thing that separates one council's
    copy of a template by-law from the next."""
    return frozenset(re.findall(r"[a-z]+", text.lower()))


def _near_duplicate(a: frozenset[str], b: frozenset[str]) -> bool:
    """Jaccard overlap above DUPLICATE_JACCARD.

    Exact fingerprints don't work on this corpus: these instruments are
    laid out in two columns and the PDF text extractor interleaves them
    differently each time, so identical schedules come out with the
    words in different orders. Comparing bags of words rides over that.
    Measured on the council by-laws, copies score 0.84-1.00 and
    genuinely different provisions 0.15 — the gap is wide enough that
    the threshold isn't delicate.
    """
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= DUPLICATE_JACCARD


def _tokenize(text: str) -> list[str]:
    # strip leading zeros from numbers so "096" matches "96"
    return [t.lstrip("0") or "0" if t.isdigit() else t
            for t in re.findall(r"[a-z0-9]+", text.lower())]


def _stem(token: str) -> str:
    """Crude plural stripping — enough that a reader's "complaint against
    a legal practitioner" reaches the Legal Practitioners Act."""
    return token[:-1] if token.endswith("s") and len(token) > 3 else token


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

        # How many instruments each title word appears in, so the title
        # boost can ignore the words that are just how titles are built
        # here. Measured on this corpus the two populations don't
        # overlap: subject words top out around 1.5% of instruments
        # ("deeds"), scaffolding starts at 10% ("zimbabwe", "high",
        # "court", "state", "versus", and "judgment" at 49%).
        self.title_df: dict[str, int] = {}
        for inst in self.instruments:
            for w in {_stem(t) for t in _tokenize(inst["title"])}:
                self.title_df[w] = self.title_df.get(w, 0) + 1
        self._df_cutoff = max(1, int(len(self.instruments) * TITLE_DF_MAX))

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
        return {_stem(t) for t in _tokenize(title)
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
        base = {_stem(t) for t in _tokenize(title)
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
            tset = {_stem(t) for t in _tokenize(inst["title"])
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

    def _boosts(self, query: str, scores: dict[int, float],
                doc_type: str | None = None
                ) -> tuple[dict[int, float], dict[int, float]]:
        """(kept, prior) bonus per record index, kept apart on purpose.

        The split is by what a cross-encoder can and cannot know.

        KEPT bonuses survive reranking. Rules 1 and 2 answer a reference
        the user actually typed — name an instrument and you want that
        instrument, whatever a relevance model thinks. Rule 4 is not a
        relevance claim at all but one about authority: for a question
        about the law, the enacted text outranks a judgment discussing
        it. A cross-encoder scores topical fit and has no notion of
        precedence, so it cannot make that call.

        The PRIOR is rule 3 alone: a guess at topical relevance from
        title words, which is exactly the job the cross-encoder does
        properly. Applying both double-counts, and on the reranked scale
        the guess wins — so the reranked path drops it (see _rerank).

        1. "SI 107 of 2026" / "Act 3 of 2026": pull in every chunk of the
           named instrument, strongly boosted — including chunks that
           neither retrieval leg returned.                       [kept]
        2. "section 71": boost candidate chunks whose citation is s 71.
                                                                 [kept]
        3. Title words: lightly boost candidates from instruments whose
           title shares 2+ meaningful words with the query.      [prior]
        4. Legislation prior: unless the question is about case law,
           prefer the enacted text over judgments discussing it.  [kept]
        """
        kept: dict[int, float] = {}
        prior: dict[int, float] = {}

        # 1. named instruments
        refs = [(int(n), y) for n, y in Q_INSTR_RE.findall(query)]
        if refs:
            for idx, r in enumerate(self.records):
                title_l = r.get("doc_title", "").lower()
                t_nums = [int(x) for x in re.findall(r"\d{1,4}", title_l)]
                for num, yr in refs:
                    if int(yr) in t_nums and num in t_nums:
                        kept[idx] = kept.get(idx, 0) + BOOST_INSTRUMENT
                        break

        # Candidates for the narrower boosts below: what retrieval found,
        # plus whatever rule 1 just pulled in.
        cands = set(scores) | set(kept)

        # 2. named sections
        secs = {s.lower() for s in Q_SECTION_RE.findall(query)}
        # avoid treating instrument numbers ("SI 107") as section numbers
        secs -= {str(n) for n, _ in refs}
        if secs:
            for idx in cands:
                cite = self.records[idx].get("citation", "").lower()
                if any(re.search(rf"\bs\s+{re.escape(s)}\b", cite)
                       for s in secs):
                    kept[idx] = kept.get(idx, 0) + BOOST_SECTION

        # 3. title-word overlap
        # "appealing a High Court judgment to the Supreme Court" shares
        # {judgment, high, court} with half the corpus, so before the
        # cutoff this rule boosted every judgment in the candidate set
        # and the Supreme Court Act never surfaced.
        q_words = {_stem(w) for w in set(_tokenize(query)) - STOPWORDS
                   if self.title_df.get(_stem(w), 0) <= self._df_cutoff}
        if len(q_words) >= 2:
            for idx in cands:
                t_words = {_stem(w) for w in set(_tokenize(
                    self.records[idx].get("doc_title", ""))) - STOPWORDS}
                if len(q_words & t_words) >= 2:
                    prior[idx] = prior.get(idx, 0) + BOOST_TITLE

        # 4. legislation prior
        if not doc_type and not Q_CASELAW_RE.search(query):
            for idx in cands:
                if self.records[idx].get("doc_type") in LEGISLATION_TYPES:
                    kept[idx] = kept.get(idx, 0) + BOOST_LEGISLATION
        return kept, prior

    @staticmethod
    def _passage(c: dict) -> str:
        """What the cross-encoder is shown for a candidate.

        The chunk's own text is not enough. Half of legal questions name
        the instrument ("what does the Labour Act say about maternity
        leave"), but a provision rarely repeats its own title, so on text
        alone the model cannot tell the Labour Act's section 18 from the
        five collective bargaining agreements that merely say maternity
        leave is granted "in terms of the Labour Act" — and it ranked
        them all above it. Prefixing the citation (which carries the
        title and the section heading) puts the Act back on top.
        """
        head = c.get("citation") or c.get("doc_title") or ""
        return f"{head}\n{c['text']}" if head else c["text"]

    def _rerank(self, query: str, cands: list[dict],
                kept: dict[int, float]) -> list[dict]:
        """Rescore the candidate pool with a cross-encoder, keeping the
        exact-reference bonuses in play. No-op if reranking is off.

        The title-word prior is deliberately dropped here: it is a
        cheap guess at the question the cross-encoder now answers
        directly, and on this scale the guess wins. Section 113 of the
        Constitution, "States of public emergency", scored a near-perfect
        0.9998 for "declaration of a state of public emergency" and still
        came 13th behind a solid-waste disaster SI on 0.987 — because
        that SI's title happened to repeat the query's words and drew a
        boost worth more than the whole gap between them.
        """
        rs = rerank.scores(query, [self._passage(c) for c in cands])
        if rs is None:
            return cands
        for c, s in zip(cands, rs):
            c["rerank"] = round(s, 4)
            # Cross-encoder relevance is 0..1, whereas the bonuses are
            # sized against RRF's ~0.016-per-rank scale. Lift them onto
            # the new scale or "SI 107 of 2026" would stop outranking a
            # merely on-topic chunk.
            c["score"] = round(
                s + kept.get(c["_idx"], 0.0) * BOOST_RERANK_SCALE, 4)
        cands.sort(key=lambda c: c["score"], reverse=True)
        return cands

    def search(self, query: str, top_k: int = TOP_K,
               doc_type: str | None = None,
               year: int | None = None) -> list[dict]:
        self._maybe_reload()
        query = normalize_refs(query)
        scores: dict[int, float] = {}
        # Reranking is only worth its latency if the legs over-fetch and
        # the cross-encoder narrows down; at 20+20 the pool barely exceeds
        # top_k and there is nothing for it to rescue.
        reranking = rerank.enabled()
        vector_k = RERANK_VECTOR_K if reranking else VECTOR_K
        bm25_k = RERANK_BM25_K if reranking else BM25_K

        # Vector leg
        where = {"doc_type": doc_type} if doc_type else None
        res = self.col.query(query_texts=[query], n_results=vector_k,
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
            if bm25_scores[idx] <= 0 or added >= bm25_k:
                break
            if doc_type and self.records[idx].get("doc_type") != doc_type:
                continue
            scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
            added += 1

        # Precision boosts. Both kinds shape the candidate pool; only
        # `kept` survives reranking, so the title-word prior can promote
        # a chunk into the pool without then outvoting the cross-encoder
        # about where in that pool it belongs.
        kept, prior = self._boosts(query, scores, doc_type)
        for idx, b in kept.items():
            scores[idx] = scores.get(idx, 0) + b
        for idx, b in prior.items():
            scores[idx] = scores.get(idx, 0) + b

        # Rank, then dedupe: one chunk per citation, skip TOC-line chunks.
        # With a reranker attached, keep a deeper pool — its whole job is
        # to find the chunk RRF buried, so handing it only top_k wastes it.
        want = max(top_k, RERANK_CANDIDATES) if reranking else top_k
        ranked = sorted(scores, key=scores.get, reverse=True)
        out: list[dict] = []
        spare: list[dict] = []
        seen_cites: set[str] = set()
        kept_words: list[frozenset[str]] = []
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
            hit = {**r, "score": round(scores[i], 4), "_idx": i}
            # Held back rather than dropped: when the query really is
            # about the by-laws, the copies are the answer and there is
            # nothing else to fill the list with.
            words = _word_set(r.get("text", ""))
            if sum(_near_duplicate(words, w) for w in kept_words) \
                    >= MAX_PER_TEMPLATE:
                spare.append(hit)
                continue
            kept_words.append(words)
            out.append(hit)
            if len(out) >= want:
                break
        if len(out) < want:
            out += spare[:want - len(out)]

        if len(out) > 1:
            out = self._rerank(query, out, kept)
        out = out[:top_k]
        for r in out:
            r.pop("_idx", None)
        return out

    def multi_search(self, queries: list[str], top_k: int = TOP_K,
                     doc_type: str | None = None,
                     year: int | None = None) -> list[dict]:
        """Search several sub-queries and fuse the result lists with RRF.

        Used by query decomposition (decompose.py). Fusing on rank rather
        than raw score matters here: the sub-queries are different
        questions whose scores aren't comparable, and rank fusion stops
        whichever sub-query happens to match more chunks from crowding
        the others out of the final eight.
        """
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return []
        if len(queries) == 1:
            return self.search(queries[0], top_k=top_k,
                               doc_type=doc_type, year=year)

        fused: dict[str, float] = {}
        best: dict[str, dict] = {}
        for q in queries:
            # Over-fetch per sub-query: a provision sitting mid-list for
            # two of them should outrank one that topped a single list.
            hits = self.search(q, top_k=top_k * 2,
                               doc_type=doc_type, year=year)
            for rank, r in enumerate(hits):
                cite = r.get("citation") or r.get("doc_title", "")
                if not cite:
                    continue
                fused[cite] = fused.get(cite, 0.0) + 1 / (RRF_K + rank + 1)
                if cite not in best or \
                        r.get("score", 0) > best[cite].get("score", 0):
                    best[cite] = r
        # Diversity cap. Dedupe is per citation, but a long judgment can
        # yield a dozen separately-cited chunks and swallow most of the
        # result set — which defeats the point of having split the
        # question up. Cap each instrument, then backfill from the
        # overflow if that left us short.
        ranked = sorted(fused, key=fused.get, reverse=True)
        picked: list[str] = []
        spare: list[str] = []
        per: dict[str, int] = {}
        for c in ranked:
            title = best[c].get("doc_title", "")
            if per.get(title, 0) >= MAX_PER_INSTRUMENT:
                spare.append(c)
                continue
            per[title] = per.get(title, 0) + 1
            picked.append(c)
            if len(picked) >= top_k:
                break
        picked += spare[:max(0, top_k - len(picked))]
        return [{**best[c], "score": round(fused[c], 4)} for c in picked]
