"""Scrape consolidated legislation from zimlii.org into data/raw/.

ZimLII (https://zimlii.org/legislation/) publishes Zimbabwe's consolidated
Acts (and some SIs/ordinances) as clean Akoma Ntoso HTML — the same data
as the Laws.Africa API, but no token needed. The site blocks plain
`requests` (403), so we use curl_cffi with impersonate="chrome".

Each listing row links to an expression like:
    /akn/zw/act/1985/5/eng@2016-12-31
which we save as act_<year>_<num>_<slug>.html. Titles + chapter numbers
go into data/raw/_manifest.json, which ingest/run.py reads for metadata.

Resumable: existing files are skipped.

Usage:
    python -m ingest.scrape_zimlii --probe        # list works, no download
    python -m ingest.scrape_zimlii                # all Acts
    python -m ingest.scrape_zimlii --types act si ord
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from curl_cffi import requests as creq
except ImportError:
    sys.exit("pip install curl_cffi (ZimLII blocks plain requests)")

BASE = "https://zimlii.org"
LIST_URL = f"{BASE}/legislation/"
DELAY = 2.5  # be polite — too fast gets the cf_clearance cookie revoked
MANIFEST_PATH = RAW_DIR / "_manifest.json"


def headers() -> dict:
    """The /akn/ document pages are behind a Cloudflare managed challenge
    (the /legislation/ index is not). Borrow the browser's session:
    open any Act page on zimlii.org, F12 -> Network -> first request ->
    copy Cookie (has cf_clearance) + User-Agent into .env as
    ZIMLII_COOKIE / ZIMLII_UA. cf_clearance expires after ~30 min."""
    h = {}
    if os.getenv("ZIMLII_COOKIE"):
        h["Cookie"] = os.getenv("ZIMLII_COOKIE")
    if os.getenv("ZIMLII_UA"):
        h["User-Agent"] = os.getenv("ZIMLII_UA")
    return h


def get(url: str) -> "creq.Response":
    time.sleep(DELAY)
    resp = creq.get(url, impersonate="chrome", timeout=120,
                    headers=headers())
    if resp.status_code == 403:
        raise RuntimeError(
            "403 (Cloudflare) — set/refresh ZIMLII_COOKIE and ZIMLII_UA "
            "in .env: open an Act page on zimlii.org in your browser, "
            "F12 -> Network -> first request -> copy Cookie + User-Agent. "
            "cf_clearance expires after ~30 min.")
    resp.raise_for_status()
    return resp


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]


def work_type(frbr_uri: str) -> str:
    m = re.match(r"/akn/zw/act/(si|gn|ord)/", frbr_uri)
    if m:
        return m.group(1)
    if frbr_uri.startswith("/akn/zw/act/"):
        return "act"
    return "other"


def list_works() -> list[dict]:
    """All works from the paginated legislation index."""
    works, seen, page = [], set(), 1
    while True:
        soup = BeautifulSoup(get(f"{LIST_URL}?page={page}").text,
                             "html.parser")
        found = 0
        for a in soup.select("a[href^='/akn/zw/']"):
            href = a["href"].split("?")[0]
            title = a.get_text(" ", strip=True)
            if not title or href in seen:
                continue
            seen.add(href)
            chapter = ""
            row = a.find_parent("tr")
            if row:
                m = re.search(r"Chapter\s+\d+:\d+",
                              row.get_text(" ", strip=True))
                if m:
                    chapter = m.group(0)
            works.append({"href": href, "title": title, "chapter": chapter})
            found += 1
        print(f"  page {page}: {found} works ({len(works)} total)")
        if not found:
            break
        # stop when the pager has no page beyond the current one
        pages = [int(n) for n in
                 re.findall(r"page=(\d+)", str(soup))]
        if not pages or page >= max(pages):
            break
        page += 1
    return works


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--types", nargs="+", default=["act"],
                    choices=["act", "si", "gn", "ord"])
    args = ap.parse_args()

    print("listing works...")
    works = list_works()

    wanted = []
    for w in works:
        # href: /akn/zw/act/1985/5/eng@2016-12-31 -> work uri + expression
        m = re.match(r"(/akn/zw/[\w/-]+?/(\d{4})/([\w-]+))(/|$)", w["href"])
        if not m:
            continue
        w["frbr_uri"], w["year"], w["num"] = m.group(1), m.group(2), m.group(3)
        w["type"] = work_type(w["frbr_uri"])
        if w["type"] in args.types:
            wanted.append(w)
    print(f"{len(wanted)} works of type {args.types} "
          f"(of {len(works)} listed)")

    if args.probe:
        for w in wanted:
            print(f"  [{w['type']}] {w['title'][:60]:62} {w['chapter']}")
        return

    originals = RAW_DIR / "originals"
    originals.mkdir(exist_ok=True)
    manifest = load_manifest()
    new = failed = 0

    def fetch_pdf(href: str) -> bytes | None:
        try:
            pdf = get(BASE + href)
        except RuntimeError as e:  # Cloudflare — no point continuing
            MANIFEST_PATH.write_text(
                json.dumps(manifest, indent=1, ensure_ascii=False),
                encoding="utf-8")
            sys.exit(f"\nSTOPPED after {new} saved: {e}")
        except Exception as e:
            print(f"  source FAILED: {e}")
            return None
        return pdf.content if pdf.content.startswith(b"%PDF") else None

    for w in wanted:
        title = (f"{w['title']} [{w['chapter']}]" if w["chapter"]
                 else w["title"])
        # Consolidations are frozen at an expression date (eng@YYYY-MM-DD).
        # Put it in the title so the agent and users can see the vintage.
        em = re.search(r"eng@(\d{4}-\d{2}-\d{2})", w["href"])
        expr_date = em.group(1) if em else ""
        if expr_date:
            title += f" (consolidated as at {expr_date})"
        stem = f"{w['type']}_{w['year']}_{w['num']}_{slugify(w['title'])}"
        dest = RAW_DIR / f"{stem}.html"
        pdf_dest = RAW_DIR / f"{stem}.pdf"
        orig_dest = originals / f"{stem}.pdf"
        if pdf_dest.exists():
            continue
        if dest.exists() and orig_dest.exists():
            continue

        if dest.exists():
            # HTML saved on an earlier run — backfill the original PDF
            html_text = dest.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                resp = get(BASE + w["href"])
            except RuntimeError as e:
                MANIFEST_PATH.write_text(
                    json.dumps(manifest, indent=1, ensure_ascii=False),
                    encoding="utf-8")
                sys.exit(f"\nSTOPPED after {new} saved: {e}")
            except Exception as e:
                print(f"  FAILED {w['title'][:60]}: {e}")
                failed += 1
                continue
            html_text = resp.text

        soup = BeautifulSoup(html_text, "html.parser")
        akn = soup.find(class_="akn-akomaNtoso")
        body_len = len(akn.get_text(" ", strip=True)) if akn else 0
        src = soup.find("a", href=re.compile(r"/source/?$"))

        if body_len < 1500:
            # Stub — no digitised text on ZimLII (mostly amendment and
            # appropriation Acts). The Gazette PDF is the only content:
            # index the PDF itself.
            if not src:
                print(f"  stub with no source PDF: {w['title'][:60]}")
                failed += 1
                continue
            blob = fetch_pdf(src["href"])
            if blob is None:
                failed += 1
                continue
            pdf_dest.write_bytes(blob)
            saved = pdf_dest
        else:
            # Full text — index the HTML, and keep the original Gazette
            # PDF beside it (data/raw/originals/) for the app's viewer.
            if not dest.exists():
                dest.write_text(html_text, encoding="utf-8")
            saved = dest
            if src and not orig_dest.exists():
                blob = fetch_pdf(src["href"])
                if blob is not None:
                    orig_dest.write_bytes(blob)
                    print(f"  + original pdf for {stem[:55]}")
        manifest[saved.name] = {
            "title": title,
            "doc_type": w["type"],
            "source": "zimlii.org",
            "frbr_uri": w["frbr_uri"],
            "expression_date": expr_date,
        }
        new += 1
        print(f"  saved {saved.name}")
        if new % 25 == 0:  # checkpoint so an interrupted run keeps metadata
            MANIFEST_PATH.write_text(
                json.dumps(manifest, indent=1, ensure_ascii=False),
                encoding="utf-8")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{new} new, {failed} failed -> {RAW_DIR}")
    print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
