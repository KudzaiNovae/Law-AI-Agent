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
    if name.startswith("judgment_"):
        doc_type = "judgment"
    elif name.startswith("si_"):
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
    # act_2025_3_x -> "X (No. 3 of 2025)"
    am = re.match(r"^(\d{4})\s+(\d+[A-Za-z]?)\s+(.*)$", title)
    if doc_type == "act" and am:
        title = f"{am.group(3)} (No. {am.group(2)} of {am.group(1)})"
    return {"title": title, "doc_type": doc_type}


# A Zimbabwean national ID ("63-1002015-E-42"), and the same digits as
# pdfplumber renders them out of a right-to-left table, which is how the
# deceased-voter registers come out.
NATIONAL_ID_RE = re.compile(
    r"\b\d{2}\s?-\s?\d{6,7}\s?-?\s?[A-Z]\b|\b[A-Z]\s?-\s?\d{6,7}\s?-\s?\d{2}\b")
# A register has to be both long enough to be a bulk roll and mostly
# made of ID numbers. Counting alone is not enough: the Constitutional
# Court presidential election petition annexes voter affidavits and
# carries 112 IDs, so a bare count threshold low enough to catch a small
# roll also throws away a landmark judgment.
#
# Density separates them with room to spare. Measured over this corpus:
#
#   deceased-voter registers      1 ID per 140-155 chars
#   presidential election petition    1 ID per 1 106 chars
#   densest real Gazette issue        1 ID per 5 985 chars
#
# 500 sits between the first two groups with 3x margin below and 2x
# above. Both conditions must hold, so a short document with a dense
# handful of IDs — a deed transfer, say — is not a register either.
MIN_NATIONAL_IDS = 100
MIN_CHARS_PER_ID = 500


def is_personal_register(text: str) -> bool:
    """A bulk roll of named individuals — a voters roll, a deceased
    register — rather than an instrument.

    Worth its own rule for two reasons. Each row becomes a chunk, so one
    such file can outweigh a hundred Acts in the index and answer
    nothing. And the rows carry national ID numbers, addresses and dates
    of birth, which have no business being served back out of a public
    search API however lawfully the Gazette published them.

    The test is density rather than count: a register is made of ID
    numbers, where a judgment only mentions some.
    """
    found = len(NATIONAL_ID_RE.findall(text))
    if found < MIN_NATIONAL_IDS:
        return False
    return len(text) / found < MIN_CHARS_PER_ID


# Directories of named people or companies that carry no ID numbers, so
# the rule above cannot see them. Listed one by one on purpose: every
# general rule tried for these also caught real law — the closest put
# the Criminal Procedure (Sentencing Guidelines) Regulations one point
# away from a roll of election candidates. A list cannot make that
# mistake, and it can be read and argued with.
DIRECTORY_FILES = {
    # 5 580 rows: company name, e-mail address and mobile number
    "gn_gn_2022_1508_registered_suppliers_of_goods.pdf",
    # 5 185 rows, one award recipient's name each
    "gazette_government_gazette_extraordinary_vol_civ_no_59.pdf",
    # rolls of 2023 election candidates. These answer "who stood in
    # Mutare North", which is not a question about the law, and they
    # crowd out the Electoral Act on "harmonised election nomination
    # court" — see the eval of that name.
    "gn_gn_2023_1129_party_list_nomination_results.pdf",
    "gn_gn_2023_1130_nomination_court_results_for_national_assembly_"
    "direct_election.pdf",
}


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
            if path.name in DIRECTORY_FILES:
                print(f"skip (directory of names, not law): {path.name}")
                continue
            # same judgment available from another source (see
            # ingest/dedupe_judgments.py) — index only the kept copy
            if RAW_MANIFEST.get(path.name, {}).get("duplicate_of"):
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
            # The filename rule above only catches registers that say so
            # in their name; these arrive titled by Gazette volume.
            if is_personal_register(text):
                print("  skip (register of named individuals)")
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
