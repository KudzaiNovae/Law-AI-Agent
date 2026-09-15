"""OCR the scanned PDFs in data/raw, in place.

Veritas and the JSC publish some judgments as page images — the 2013
Constitutional Court presidential election order among them. pdfplumber
gets nothing out of those, so they reach the index as zero chunks and
the reader is told the law is silent when it is merely unreadable.

Detection is by extraction, not by filename or size: a file is scanned
if pdfplumber yields almost no text from it. That is the same test
ingest/extract.py warns on, so this script and ingest agree by
construction.

Originals are moved to data/raw_prescan/ rather than overwritten, so a
bad OCR pass is reversible.

    sudo apt install ocrmypdf tesseract-ocr     # once
    python scripts/ocr_scanned.py               # dry run: list candidates
    python scripts/ocr_scanned.py --apply
    python -m ingest.run && python -m index.build_index

Re-running is safe: an OCR'd file now yields text, so it is no longer a
candidate.
"""
import argparse
import contextlib
import io
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, DATA_DIR, PROCESSED_DIR        # noqa: E402

PRESCAN_DIR = DATA_DIR / "raw_prescan"
# Matches the threshold ingest/extract.py warns at.
MIN_TEXT = 200


def _text_of(path: Path) -> str | None:
    """Extracted text, or None if the file cannot be parsed at all."""
    from ingest.extract import _from_pdf
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # mute the warning
            return _from_pdf(path)
    except Exception:
        return None


def _chars_per_file() -> dict[str, int]:
    """How much text each file yielded at the last ingest.

    Re-parsing 7 600 PDFs to find a dozen scans costs the better part of
    an hour; ingest already measured every one of them, so read that.
    """
    import json
    path = PROCESSED_DIR / "chunks.jsonl"
    if not path.exists():
        return {}
    chars: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            name = c.get("source_file")
            chars[name] = chars.get(name, 0) + len(c.get("text", ""))
    return chars


def _ingest_skips(p: Path) -> bool:
    """Files ingest drops for reasons unrelated to being scanned. They
    yield no text, so without this they all look like scans."""
    import re
    from ingest.run import DIRECTORY_FILES, RAW_MANIFEST
    return bool(
        p.stat().st_size > _cap(p)
        or re.search(r"voter_?s?_roll|voters_roll", p.name, re.I)
        or p.name in DIRECTORY_FILES
        or RAW_MANIFEST.get(p.name, {}).get("duplicate_of"))


def candidates() -> list[Path]:
    """PDFs that pdfplumber cannot read.

    Two passes. The first is a cheap shortlist from the last ingest's
    output — a file that already produced plenty of text is not a scan,
    and needs no reopening. The second confirms the shortlist by actually
    extracting, because a file can produce nothing for reasons other than
    being a scan. Only the shortlist pays for extraction.
    """
    from ingest.run import is_personal_register

    chars = _chars_per_file()
    pdfs = [p for p in sorted(RAW_DIR.iterdir())
            if p.is_file() and p.suffix.lower() == ".pdf"]

    if chars:
        shortlist = [p for p in pdfs
                     if chars.get(p.name, 0) < MIN_TEXT
                     and not _ingest_skips(p)]
        print(f"  {len(pdfs)} PDFs, {len(shortlist)} yielded no text at the "
              f"last ingest — checking those")
    else:
        shortlist = pdfs
        print(f"  no chunks.jsonl yet, so every one of {len(pdfs)} PDFs has "
              f"to be opened — this takes a while. Run ingest first to "
              f"make it quick.")

    out = []
    for i, p in enumerate(shortlist, 1):
        # Opening a 14 MB scan takes tens of seconds, so say which file
        # is being read rather than looking hung.
        print(f"    [{i}/{len(shortlist)}] {p.name[:58]}", flush=True)
        text = _text_of(p)
        if text is None:                 # unparseable; OCR may rescue it
            out.append(p)
            continue
        if len(text.strip()) >= MIN_TEXT:
            continue
        # A roll of names that happens to be a scan stays unread on
        # purpose: OCR would only manufacture more personal data.
        if is_personal_register(text):
            continue
        out.append(p)
    return out


# ocrmypdf's exit code for a password-protected input.
ENCRYPTED = "encrypted"


def _ocr(src: Path, dst: Path, mode: str, lang: str) -> tuple[int, str | None]:
    """Run ocrmypdf. Returns (chars recovered, error or None)."""
    try:
        subprocess.run(
            ["ocrmypdf", mode, "--optimize", "3",
             "--language", lang, "--quiet", str(src), str(dst)],
            check=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        return 0, ENCRYPTED if e.returncode == 8 else f"ocrmypdf exit {e.returncode}"
    except subprocess.TimeoutExpired:
        return 0, "timed out after 30 min"
    return len((_text_of(dst) or "").strip()), None


def _cap(p: Path) -> int:
    """ingest/run.py skips oversize files. OCR adds a text layer and can
    push a file over, which would trade an unreadable instrument for a
    silently dropped one."""
    return 25_000_000 if p.name.startswith("report_") else 8_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually OCR (default: list candidates)")
    ap.add_argument("--lang", default="eng")
    args = ap.parse_args()

    if args.apply and not shutil.which("ocrmypdf"):
        sys.exit("ocrmypdf not installed: sudo apt install ocrmypdf "
                 "tesseract-ocr")

    print("scanning data/raw for unreadable PDFs ...")
    todo = candidates()
    if not todo:
        print("Nothing to OCR.")
        return
    mb = sum(p.stat().st_size for p in todo) / 1e6
    for p in todo:
        print(f"  {p.stat().st_size/1e6:>6.1f} MB  {p.name}")
    print(f"\n{len(todo)} files, {mb:.0f} MB")
    if not args.apply:
        print("Dry run — rerun with --apply to OCR them.")
        return

    PRESCAN_DIR.mkdir(exist_ok=True)
    done = failed = 0
    for i, p in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {p.name}")
        tmp = p.with_suffix(".ocr.tmp.pdf")

        # --skip-text first: it leaves any real text layer alone, so a
        # part-scanned file keeps its good pages. --force-ocr is the
        # fallback rather than the default because it rasterises
        # everything, which would degrade those good pages.
        gained, err = _ocr(p, tmp, "--skip-text", args.lang)
        if err == ENCRYPTED:
            print("  FAILED — the PDF is password-protected. Not an OCR "
                  "problem; re-download it from the source.")
            tmp.unlink(missing_ok=True)
            failed += 1
            continue
        if err is None and gained < MIN_TEXT:
            # Some JSC scans carry an empty text layer: ocrmypdf counts
            # the page as done and skips it, while pdfplumber reads
            # nothing. Rasterising past it is the only way in.
            print(f"  {gained} chars with --skip-text (empty text layer) "
                  f"— retrying with --force-ocr")
            tmp.unlink(missing_ok=True)
            gained, err = _ocr(p, tmp, "--force-ocr", args.lang)
        if err is not None:
            print(f"  FAILED ({err})")
            tmp.unlink(missing_ok=True)
            failed += 1
            continue
        if gained < MIN_TEXT:
            print(f"  no text recovered ({gained} chars) — leaving original")
            tmp.unlink(missing_ok=True)
            failed += 1
            continue

        size = tmp.stat().st_size
        shutil.move(str(p), str(PRESCAN_DIR / p.name))
        tmp.rename(p)
        note = ""
        if size > _cap(p):
            note = (f"  !! {size/1e6:.1f} MB is over the "
                    f"{_cap(p)//1_000_000} MB ingest cap — it will be "
                    f"skipped. Split it or raise the cap.")
        print(f"  OK  {gained:,} chars  ({size/1e6:.1f} MB){note}")
        done += 1

    print(f"\n{done} OCR'd, {failed} failed. Originals in {PRESCAN_DIR}")
    if done:
        print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
