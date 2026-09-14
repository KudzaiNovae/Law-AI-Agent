"""Mark duplicate judgments in data/raw/_manifest.json.

The same judgment often appears in several sources (JSC official PDF,
ZimLII full text) and ZimLII itself sometimes lists one judgment under
two citation numbers.  Citation numbers and dates disagree across
sources, so matching is done on (year, court group, party names):
ZimLII party surnames must be a subset of the JSC full-name parties.

Kept entry: ZimLII (clean full text, Held/flynote).  Duplicates get
``duplicate_of: <kept filename>`` — ingest and /api/caselaw skip them.
Safe to re-run any time (idempotent); run it after every scrape since
running scrapers can clobber earlier marks on merge-save.
"""
from __future__ import annotations

import collections
import json
import re
from datetime import datetime
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

STOP = {"the", "of", "and", "others", "another", "pvt", "ltd", "private",
        "limited", "no", "na", "anor", "ors", "state", "s"}


def _isod(s: str) -> str:
    s = (s or "").strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    try:
        return datetime.strptime(s, "%d %B %Y").strftime("%Y-%m-%d")
    except ValueError:
        return ""


def _group(court: str, title: str) -> str:
    c = f"{court or ''} {title}".lower()
    if "constitutional" in c or "zwcc" in c or "ccz" in c:
        return "CC"
    if "supreme" in c or "zwsc" in c or re.search(r"\bsc\s?\d", c):
        return "SC"
    if "labour" in c or "zwlc" in c:
        return "LC"
    if "magistr" in c:
        return "MC"
    return "HC"


def _norm(s: str) -> set[str]:
    return set(re.sub(r"[^a-z ]", " ", s.lower()).split()) - STOP


def _parties(t: str) -> list[set[str]]:
    t = re.sub(r"^Judgment [^:]+:\s*", "", t)
    t = re.sub(r"\((?:[^()]*)\)", " ", t)
    t = re.sub(r"\[\d{4}\].*$", "", t)
    t = re.sub(r"(?i)(?<=[a-z0-9])versus", " versus ", t)  # "+23versus"
    parts = re.split(r"\s+v(?:ersus|s)?[\s.]+", t, flags=re.I)
    return [_norm(p) for p in parts[:2]]


def main() -> None:
    mp = RAW_DIR / "_manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))

    judg = {}
    for k, v in manifest.items():
        if v.get("doc_type") != "judgment":
            continue
        t = v.get("title", "")
        judg[k] = (v.get("source", ""), _isod(v.get("date", "")),
                   _group(v.get("court", ""), t), _parties(t))
        v.pop("duplicate_of", None)  # recompute from scratch

    def usable(p):
        return len(p) == 2 and p[0] and p[1]

    # -- 1. ZimLII internal dups: same exact date + group + parties -----
    zim_dups = 0
    by_key: dict = {}
    for k, (src, d, g, p) in sorted(judg.items()):
        if src != "zimlii.org" or not d or not usable(p):
            continue
        key = (d, g, frozenset(p[0]), frozenset(p[1]))
        if key in by_key:
            manifest[k]["duplicate_of"] = by_key[key]
            zim_dups += 1
        else:
            by_key[key] = k

    # -- 2. JSC dups of ZimLII: same year + group, party-subset match ---
    zim_by_year = collections.defaultdict(list)
    for k, (src, d, g, p) in judg.items():
        if (src == "zimlii.org" and d and usable(p)
                and "duplicate_of" not in manifest[k]):
            zim_by_year[d[:4]].append((k, g, p))
    jsc_dups = 0
    for k, (src, d, g, p) in judg.items():
        if src != "jsc.org.zw" or not d or not usable(p):
            continue
        for zk, zg, zp in zim_by_year.get(d[:4], []):
            if zg == g and zp[0] <= p[0] and zp[1] <= p[1]:
                manifest[k]["duplicate_of"] = zk
                jsc_dups += 1
                break

    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                  encoding="utf-8")
    print(f"zimlii internal dups: {zim_dups}")
    print(f"jsc dups of zimlii:   {jsc_dups}")
    print(f"total marked:         {zim_dups + jsc_dups}")


if __name__ == "__main__":
    main()
