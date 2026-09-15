#!/usr/bin/env python
"""Retrieval regression evals — no model calls, runs in seconds.

Each line of tests/evals.jsonl is {"q": query, "expect_any": [substrings]}.
A case passes if any expected substring appears (case-insensitive) in the
doc_title or citation of the top-K search results.

A case may also carry "known_gap": "<why>". That marks a question this
corpus cannot answer because the instrument simply has not been ingested
— not a retrieval bug. Those are reported separately and do not fail the
run, so the list doubles as the ingestion to-do list. If one starts
passing, the gap has been filled and the flag should come off.

Run after re-ingesting or changing retrieval code:
    .venv/bin/python scripts/run_evals.py
Exit code 1 if any case fails, so it can gate CI/cron.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index.retriever import HybridRetriever  # noqa: E402

TOP_K = 5
CASES = Path(__file__).resolve().parent.parent / "tests" / "evals.jsonl"


def main() -> int:
    retriever = HybridRetriever()
    cases = [json.loads(line) for line in
             CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed, gaps, closed = 0, 0, []
    for c in cases:
        results = retriever.search(c["q"])[:TOP_K]
        haystack = " | ".join(
            (r.get("doc_title", "") + " " + r.get("citation", ""))
            for r in results).lower()
        ok = any(e.lower() in haystack for e in c["expect_any"])
        gap = c.get("known_gap")
        if gap and not ok:
            gaps += 1
            print("GAP   " + c["q"])
            print("        " + gap)
            continue
        print(("PASS  " if ok else "FAIL  ") + c["q"])
        if gap:
            closed.append(c["q"])
        if not ok:
            failed += 1
            for r in results:
                print("        got:", r.get("citation") or r.get("doc_title"))
    scored = len(cases) - gaps
    print(f"\n{scored - failed}/{scored} passed"
          + (f", {gaps} known corpus gaps skipped" if gaps else ""))
    for q in closed:
        print(f"  note: '{q}' is marked known_gap but passes now — "
              "the instrument has been ingested, drop the flag")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
