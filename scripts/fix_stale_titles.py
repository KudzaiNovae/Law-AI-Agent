"""Invalidate chunk caches whose doc_title no longer matches the manifest.

The per-file cache (data/processed/file_cache) is keyed by size+mtime, so
editing _manifest.json alone does not invalidate it — chunks keep the old
title. This script finds those files and deletes their caches; the next
`python -m ingest.run` re-extracts them with the correct titles.

Usage:
    python scripts/fix_stale_titles.py
    python -m ingest.run && python -m index.build_index
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR

CACHE_DIR = PROCESSED_DIR / "file_cache"


def main() -> None:
    manifest = json.loads((RAW_DIR / "_manifest.json").read_text("utf-8"))

    # current title per source_file according to chunks.jsonl
    chunk_title: dict[str, str] = {}
    for line in (PROCESSED_DIR / "chunks.jsonl").read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        chunk_title.setdefault(r.get("source_file", ""), r.get("doc_title", ""))

    stale = [f for f, t in chunk_title.items()
             if f in manifest and manifest[f]["title"] != t]
    if not stale:
        print("All chunk titles match the manifest. Nothing to do.")
        return
    n = 0
    for f in stale:
        print(f"stale: {f}\n  chunks:   {chunk_title[f][:70]}\n"
              f"  manifest: {manifest[f]['title'][:70]}")
        for c in CACHE_DIR.glob(f"{f}.*.jsonl"):
            c.unlink()
            n += 1
    print(f"\n{len(stale)} file(s) stale, {n} cache file(s) removed.")
    print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
