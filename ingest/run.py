"""Process everything in data/raw/ into chunked JSONL.

Filename convention decides metadata (or edit MANIFEST below):
    constitution_*.html  -> doc_type=constitution
    *si_*.pdf / si*.pdf  -> doc_type=si
    *_act*.pdf           -> doc_type=act
    otherwise            -> doc_type=policy

Usage:
    python -m ingest.run
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR
from ingest.extract import extract_text
from ingest.chunker import chunk_document

# Override auto-detection per file if you want precise titles:
MANIFEST: dict[str, dict] = {
    "constitution_of_zimbabwe_2013.html": {
        "title": "Constitution of Zimbabwe (2013)",
        "doc_type": "constitution",
    },
}


def load_raw_manifest() -> dict:
    """Sidecar metadata written by importers (e.g. import_lawsafrica)."""
    p = RAW_DIR / "_manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {}


RAW_MANIFEST = load_raw_manifest()


def infer_meta(path: Path) -> dict:
    if path.name in MANIFEST:
        return MANIFEST[path.name]
    if path.name in RAW_MANIFEST:
        m = RAW_MANIFEST[path.name]
        return {"title": m["title"], "doc_type": m.get("doc_type", "act")}
    name = path.stem.lower()
    # Scraper prefixes (si_/gn_/act_) are authoritative — check them first,
    # otherwise a "GN ... Constitution Notice" gets typed as constitution.
    if name.startswith("si_"):
        doc_type = "si"
    elif name.startswith("gn_"):
        doc_type = "gn"
    elif name.startswith("act_"):
        doc_type = "act"
    elif name.startswith("gazette_"):
        doc_type = "gazette"       # full Gazette issue (JSC scraper)
    elif name.startswith("report_"):
        doc_type = "report"        # tabled report (parlzim scraper)
    elif "constitution" in name:
        doc_type = "constitution"
    elif "_si_" in name or "statutory" in name:
        doc_type = "si"
    elif "act" in name:
        doc_type = "act"
    else:
        doc_type = "policy"

    stem = path.stem
    # Strip scraper's doc-type prefix ("si_", "gn_", "act_") from the title
    for p in ("si_", "gn_", "act_", "report_"):
        if stem.lower().startswith(p):
            stem = stem[len(p):]
            break
    title = stem.replace("_", " ").title()
    # "Si 2026 100 Postal..." -> "SI 2026-100 Postal..."
    title = re.sub(r"^(Si|Gn)\s+(\d{4})\s+(\d+)\s*",
                   lambda m: f"{m.group(1).upper()} {m.group(2)}-{m.group(3)} ",
                   title)
    return {"title": title, "doc_type": doc_type}


CACHE_DIR = PROCESSED_DIR / "file_cache"


def _cache_path(path: Path) -> Path:
    """Per-file chunk cache, keyed by size+mtime so edits invalidate it."""
    st = path.stat()
    return CACHE_DIR / f"{path.name}.{st.st_size}.{int(st.st_mtime)}.jsonl"


def main() -> None:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="ignore the per-file cache and re-extract everything")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.full:
        for p in CACHE_DIR.glob("*.jsonl"):
            p.unlink()

    out_path = PROCESSED_DIR / "chunks.jsonl"
    # Write to a temp file, rename at the end: the running server reloads
    # chunks.jsonl whenever its mtime changes, so it must never see a
    # partial file (an interrupted run used to leave a truncated one).
    tmp_path = out_path.with_suffix(".tmp")
    total = fresh = cached = 0
    live_caches: set[str] = set()
    with tmp_path.open("w", encoding="utf-8") as out:
        for path in sorted(RAW_DIR.iterdir()):
            if path.is_dir() or path.name.startswith((".", "_")):
                continue
            size = path.stat().st_size
            # Huge Gazette PDFs are bulk lists (vehicle regs, voters
            # roll) or image scans — noise for legal QA. Skip. Tabled
            # reports (report_*) are real prose, so allow them up to 25MB.
            cap = 25_000_000 if path.name.startswith("report_") else 8_000_000
            if size > cap:
                print(f"skip (>{cap // 1_000_000}MB, likely bulk "
                      f"list/scan): {path.name}")
                continue
            # Voter's-roll alteration notices are bulk name-list tables:
            # noise for QA, and pdfplumber OOMs parsing the big ones.
            if re.search(r"voter_?s?_roll|voters_roll", path.name, re.I):
                print(f"skip (voters-roll bulk list): {path.name}")
                continue

            cache = _cache_path(path)
            live_caches.add(cache.name)
            if cache.exists():
                lines = [l for l in
                         cache.read_text(encoding="utf-8").splitlines()
                         if l.strip()]
                for line in lines:
                    out.write(line + "\n")
                total += len(lines)
                cached += 1
                continue

            meta = infer_meta(path)
            print(f"processing {path.name}  ({meta['doc_type']})")
            try:
                text = extract_text(path)
            except Exception as e:  # one bad file must not kill the run
                print(f"  FAILED ({type(e).__name__}): {e}")
                continue
            chunks = chunk_document(text, meta["title"], meta["doc_type"],
                                    source_file=path.name)
            payload = "".join(
                json.dumps(c.to_dict(), ensure_ascii=False) + "\n"
                for c in chunks)
            out.write(payload)
            # stale caches for this file (old size/mtime) then fresh one
            for old in CACHE_DIR.glob(f"{path.name}.*.jsonl"):
                old.unlink()
            cache.write_text(payload, encoding="utf-8")
            live_caches.add(cache.name)
            total += len(chunks)
            fresh += 1
            print(f"  {len(chunks)} chunks")

    # drop caches for raw files that no longer exist
    for p in CACHE_DIR.glob("*.jsonl"):
        if p.name not in live_caches:
            p.unlink()

    tmp_path.replace(out_path)  # atomic
    print(f"\nWrote {total} chunks -> {out_path} "
          f"({fresh} files extracted, {cached} from cache)")
    print("Next: python -m index.build_index")


if __name__ == "__main__":
    main()
