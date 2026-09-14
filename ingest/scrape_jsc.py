"""Scrape Government Gazette PDFs from the JSC API into data/raw/.

The JSC site (www.jsc.org.zw/gazettes) is a JS frontend over a JSON API:
    https://jsc.org.zw:8222/gazette/?page=N

The exact JSON schema is unknown, so this scraper auto-detects the item
list and its fields. If detection fails, run with --probe to dump the raw
response for the first page, then adjust FIELD_HINTS below.

Usage:
    python -m ingest.scrape_jsc --probe          # inspect API schema
    python -m ingest.scrape_jsc                  # scrape all pages
    python -m ingest.scrape_jsc --max-pages 5
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

# jsc.org.zw:8222 serves an incomplete certificate chain (missing
# intermediate). Browsers repair it automatically; Python can't. We fetch
# public documents and send no credentials, so skip verification here.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API = "https://jsc.org.zw:8222/gazette/"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.jsc.org.zw/",
}
DELAY = 1.0

LIST_KEYS = ("results", "data", "items", "gazettes", "content", "records")
URL_FIELDS = ("file", "pdf", "url", "document", "path", "file_url",
              "download", "attachment", "fileUrl", "documentUrl")
TITLE_FIELDS = ("title", "name", "description", "gazette_name", "caption")
DATE_FIELDS = ("date", "published", "published_at", "created", "created_at",
               "gazette_date", "publication_date")


def get_json(url: str) -> dict | list | None:
    time.sleep(DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=60, verify=False)
    resp.raise_for_status()
    return resp.json()


def find_items(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in LIST_KEYS:
            v = data.get(k)
            if isinstance(v, list):
                return v
        # fall back to the first list value found
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def pick(item: dict, fields: tuple) -> str | None:
    for f in fields:
        v = item.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def pdf_url(item: dict) -> str | None:
    url = pick(item, URL_FIELDS)
    if not url:
        # any string value that looks like a file link
        for v in item.values():
            if isinstance(v, str) and re.search(r"\.pdf(\?|$)", v, re.I):
                url = v
                break
    if not url:
        return None
    if url.startswith("/"):
        url = "https://jsc.org.zw:8222" + url
    elif not url.startswith("http"):
        url = "https://jsc.org.zw:8222/" + url
    return url


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:90]


def prefix_for(title: str) -> str:
    """Map a title to a filename prefix so ingest.run types it correctly.

    The JSC feed mixes full Gazette issues with individual SIs and Acts.
    """
    t = title.strip().lower()
    if re.match(r"s\.?\s?\.?\s?i\.?[\s.]", t) or "statutory instrument" in t:
        return "si_"
    if re.match(r"act\s+no", t):
        return "act_"
    if "general notice" in t and "gazette" not in t:
        return "gn_"
    return "gazette_"


# --- dedupe -----------------------------------------------------------------
# The JSC feed lists many SIs twice (a short "nom" upload and a full-title
# upload), and most SIs/Acts already exist from the Veritas scrape. Dedupe on
# instrument identity (number + year), then on file content hash.

# The instrument number must sit right after the SI/Act marker; a lax
# pattern would misread "SI 143 ... Warrant 1 of 2023" as SI 1 of 2023.
# Letter suffixes matter: SI 227A and 227B are different instruments.
NUM_OF_YEAR_RE = re.compile(r"(?:s\.?\s?\.?\s?i\.?|act\s+no\.?)\s*(\d{1,4})"
                            r"\s*([a-z]?)\s+of\s+(\d{4})", re.I)
FILE_KEY_RES = (
    # Veritas: si_si_2023_029_x / si_2026_111_x  -> (type, year, num, letter)
    re.compile(r"^(si|act|gn)_(?:si_|act_|gn_)?(\d{4})_0*(\d{1,4})([a-z]?)_"),
    # JSC slug: si_s_i_110_of_2026 / act_act_no_6_of_2026 /
    # gn_general_notice_1585_of_2024      -> (type, num, letter, year)
    re.compile(r"^(si|act|gn)_(?:s_?i_|act_no_|general_notice_)"
               r"(\d{1,4})([a-z]?)_of_(\d{4})"),
)


def instrument_key(title: str) -> str | None:
    """'si-2026-111' for 'S.I 111 of 2026 ...', 'act-2026-6' for Acts."""
    m = NUM_OF_YEAR_RE.search(title)
    if not m:
        return None
    kind = "act" if title.strip().lower().startswith("act") else "si"
    return f"{kind}-{m.group(3)}-{int(m.group(1))}{m.group(2).lower()}"


def existing_keys() -> set[str]:
    keys: set[str] = set()
    for p in RAW_DIR.iterdir():
        name = p.stem.lower()
        m = FILE_KEY_RES[0].match(name)
        if m:
            keys.add(f"{m.group(1)}-{m.group(2)}-{int(m.group(3))}"
                     f"{m.group(4)}")
            continue
        m = FILE_KEY_RES[1].match(name)
        if m:
            keys.add(f"{m.group(1)}-{m.group(4)}-{int(m.group(2))}"
                     f"{m.group(3)}")
    return keys


def existing_hashes() -> set[str]:
    import hashlib
    return {hashlib.sha256(p.read_bytes()).hexdigest()
            for p in RAW_DIR.iterdir() if p.is_file()}


def probe() -> None:
    data = get_json(API)
    print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
    items = find_items(data)
    print(f"\n--- detected {len(items)} items on page 1")
    if items:
        first = items[0]
        print("first item keys:", list(first.keys()))
        print("title ->", pick(first, TITLE_FIELDS))
        print("date  ->", pick(first, DATE_FIELDS))
        print("pdf   ->", pdf_url(first))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="dump page-1 response and detected fields")
    ap.add_argument("--max-pages", type=int, default=100)
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    keys = existing_keys()
    hashes = existing_hashes()
    import hashlib

    new = 0
    for page in range(1, args.max_pages + 1):
        url = API if page == 1 else f"{API}?page={page}"
        print(f"\n== page {page} ==")
        try:
            data = get_json(url)
        except (requests.RequestException, ValueError) as e:
            print(f"  fetch failed: {e}")
            break
        items = find_items(data)
        if not items:
            print("  no items — stopping")
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            title = pick(item, TITLE_FIELDS) or "gazette"
            date = pick(item, DATE_FIELDS) or ""
            link = pdf_url(item)
            if not link:
                print(f"  no file link: {title[:60]}")
                continue
            stem = prefix_for(title) + slugify(f"{date} {title}")
            # normalise JSC slugs to the Veritas naming convention so
            # ingest.run derives clean titles ("SI 2026-110 ...")
            stem = re.sub(r"^si_s_?i_(\d{1,4})([a-z]?)_of_(\d{4})_",
                          r"si_si_\3_\1\2_", stem)
            stem = re.sub(r"^act_act_no_(\d{1,4})_of_(\d{4})_",
                          r"act_\2_\1_", stem)
            dest = RAW_DIR / f"{stem}.pdf"
            if dest.exists():
                continue
            key = instrument_key(title)
            if key and key in keys:
                print(f"  dup ({key}): {title[:55]}")
                continue
            try:
                time.sleep(DELAY)
                resp = requests.get(link, headers=HEADERS, timeout=120,
                                    verify=False)
                resp.raise_for_status()
                digest = hashlib.sha256(resp.content).hexdigest()
                if digest in hashes:
                    print(f"  dup (same content): {title[:55]}")
                    continue
                dest.write_bytes(resp.content)
                hashes.add(digest)
                if key:
                    keys.add(key)
                print(f"  saved {dest.name} ({len(resp.content):,} bytes)")
                new += 1
            except requests.RequestException as e:
                print(f"  FAILED {link}: {e}")

    print(f"\n{new} new gazettes in {RAW_DIR}")
    print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
