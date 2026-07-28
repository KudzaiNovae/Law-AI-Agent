"""Build the Chroma vector index from processed chunks.

Uses Chroma's default local embedding model (all-MiniLM, runs on CPU,
no API key needed). Swap in a hosted embedding model later if recall
isn't good enough.

Incremental by default: chunk ids are stable hashes of
(source_file, position, text), so unchanged chunks are left alone,
new ones are embedded and added, and chunks that disappeared from
chunks.jsonl are deleted. Use --full to rebuild from scratch.

Usage:
    python -m index.build_index           # incremental
    python -m index.build_index --full    # wipe and rebuild everything
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, CHROMA_DIR, COLLECTION_NAME

BATCH = 100
META_KEYS = ("doc_title", "doc_type", "section", "section_title",
             "chapter", "part", "citation", "source_file")


def chunk_id(rec: dict, pos: int) -> str:
    """Stable id: same chunk -> same id across runs."""
    h = hashlib.sha1()
    h.update(rec.get("source_file", "").encode())
    h.update(str(pos).encode())
    h.update(rec.get("text", "").encode())
    return h.hexdigest()[:20]


def load_records() -> list[dict]:
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        sys.exit("No chunks found. Run: python -m ingest.run")
    records, bad = [], 0
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1  # skip corrupt lines; retriever skips them identically
    if bad:
        print(f"WARNING: skipped {bad} corrupt line(s) in chunks.jsonl")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="delete the collection and re-embed everything")
    args = ap.parse_args()

    records = load_records()
    print(f"{len(records)} chunks in chunks.jsonl")

    # position counter per source file keeps ids stable when other files
    # are added or removed
    pos: dict[str, int] = {}
    ids = []
    for r in records:
        sf = r.get("source_file", "")
        ids.append(chunk_id(r, pos.get(sf, 0)))
        pos[sf] = pos.get(sf, 0) + 1

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if args.full:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    col = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    existing: set[str] = set()
    if col.count():
        got = col.get(include=[])  # ids only
        existing = set(got["ids"])
    print(f"{len(existing)} chunks already indexed")

    wanted = dict(zip(ids, records))  # id -> record (dedupes identical ids)
    to_add = [i for i in wanted if i not in existing]
    to_delete = [i for i in existing if i not in wanted]

    if to_delete:
        for i in range(0, len(to_delete), 500):
            col.delete(ids=to_delete[i:i + 500])
        print(f"deleted {len(to_delete)} stale chunks")

    if not to_add:
        print("Nothing new to index.")
    for i in range(0, len(to_add), BATCH):
        batch_ids = to_add[i:i + BATCH]
        batch = [wanted[b] for b in batch_ids]
        col.add(
            ids=batch_ids,
            documents=[r["text"] for r in batch],
            metadatas=[{k: r.get(k, "") for k in META_KEYS}
                       for r in batch],
        )
        print(f"  indexed {min(i + BATCH, len(to_add))}/{len(to_add)} new")

    print(f"Done. Collection '{COLLECTION_NAME}' at {CHROMA_DIR} "
          f"({col.count()} chunks)")
    print("Next: python cli.py \"What does the Constitution say about "
          "property rights?\"")


if __name__ == "__main__":
    main()
