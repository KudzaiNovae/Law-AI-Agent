"""Scrape tabled committee reports from parlzim.gov.zw into data/raw/.

Parliament publishes tabled reports (committee reports, annual reports,
Auditor-General reports) at:
    https://www.parlzim.gov.zw/tabled-reports/

How it actually works (discovered 2026-07):
  * The page itself contains no file links — it's a FileBird Document
    Library widget (WordPress plugin, div.njt-fbdl) that fetches its file
    list client-side from:
        POST /wp-json/filebird/v1/get-attachments
    with a JSON body {pagination, search, orderBy, orderType,
    selectedFolder}. selectedFolder is an encrypted folder id embedded in
    the page's data-json attribute; we read it from the live page so it
    keeps working if it changes.
  * The site sits behind a Cloudflare managed challenge. cf_clearance is
    bound to IP + User-Agent + TLS fingerprint, so plain `requests` fails
    even with a valid cookie — we use curl_cffi with impersonate="chrome".

Setup — borrow your browser's session:
    1. Open https://www.parlzim.gov.zw/tabled-reports/ in your browser.
    2. F12 -> Network -> reload -> first request -> Request Headers ->
       copy the "Cookie:" value (includes cf_clearance) and "User-Agent:".
    3. Put them in .env:
         PARLZIM_COOKIE=<cookie value>
         PARLZIM_UA=<user-agent value>
    cf_clearance expires after ~30 min — re-copy if runs start 403ing.

Resumable: already-downloaded reports are skipped (filename + sha256).
DOCX reports are converted to plain .txt at download time (the ingest
pipeline only reads pdf/html/txt).

Usage:
    python -m ingest.scrape_parlzim --probe    # list files, no download
    python -m ingest.scrape_parlzim            # download everything
"""
import argparse
import hashlib
import html as htmllib
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from curl_cffi import requests as creq
except ImportError:
    sys.exit("pip install curl_cffi  (needed to pass Cloudflare's TLS "
             "fingerprint check)")

BASE = "https://www.parlzim.gov.zw"
LIST_URL = f"{BASE}/tabled-reports/"
API_URL = f"{BASE}/wp-json/filebird/v1/get-attachments"
DELAY = 0.5
PAGE_LIMIT = 200


class CloudflareChallenge(RuntimeError):
    pass


def headers(extra: dict | None = None) -> dict:
    cookie = os.getenv("PARLZIM_COOKIE", "")
    ua = os.getenv("PARLZIM_UA", "")
    if not cookie or not ua:
        sys.exit("Set PARLZIM_COOKIE and PARLZIM_UA in .env "
                 "(see the header of this file).")
    h = {"Cookie": cookie, "User-Agent": ua, "Referer": LIST_URL}
    if extra:
        h.update(extra)
    return h


def get(url: str, **kw) -> "creq.Response":
    time.sleep(DELAY)
    resp = creq.get(url, headers=headers(), impersonate="chrome",
                    timeout=180, **kw)
    if resp.status_code == 403:
        raise CloudflareChallenge(
            "403 — cf_clearance probably expired (~30 min lifetime). "
            "Re-copy PARLZIM_COOKIE from your browser.")
    return resp


def page_widget_config() -> tuple[dict, str]:
    """Read the FileBird widget's request payload + REST nonce from the
    live tabled-reports page."""
    resp = get(LIST_URL)
    text = resp.text
    m = re.search(r'class="njt-fbdl"[^>]*data-json="([^"]+)"', text)
    if not m:
        raise RuntimeError("njt-fbdl widget not found on page — layout "
                           "changed, update page_widget_config()")
    cfg = json.loads(htmllib.unescape(m.group(1)))
    nonce = ""
    n = re.search(r'"rest_nonce"\s*:\s*"(\w+)"', text)
    if n:
        nonce = n.group(1)
    return cfg["request"], nonce


def list_files(request: dict, nonce: str) -> list[dict]:
    files, page = [], 1
    while True:
        req = dict(request)
        req["pagination"] = {"current": page, "limit": PAGE_LIMIT}
        time.sleep(DELAY)
        resp = creq.post(
            API_URL, json=req, impersonate="chrome", timeout=60,
            headers=headers({"Content-Type": "application/json",
                             "X-WP-Nonce": nonce}))
        if resp.status_code == 403:
            raise CloudflareChallenge("403 on API — cookie expired?")
        resp.raise_for_status()
        data = resp.json()
        files.extend(data.get("files", []))
        if page >= int(data.get("maxNumPages") or 1):
            break
        page += 1
    return files


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]


def docx_to_text(blob: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(blob))
    xml = z.read("word/document.xml").decode("utf-8", "replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    return htmllib.unescape(re.sub(r"<[^>]+>", "", xml))


def existing_hashes() -> set[str]:
    return {hashlib.sha256(p.read_bytes()).hexdigest()
            for p in RAW_DIR.iterdir() if p.is_file()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="list discovered files without downloading")
    args = ap.parse_args()

    request, nonce = page_widget_config()
    files = list_files(request, nonce)
    print(f"{len(files)} files listed")

    if args.probe:
        for f in files:
            print(f"  [{f.get('type'):4}] {f.get('size', ''):>7}  "
                  f"{f.get('title', '')[:70]}")
        return

    hashes = existing_hashes()
    new = skipped = failed = 0
    for f in files:
        ftype = (f.get("type") or "").lower()
        title = f.get("title") or Path(f.get("url", "x")).stem
        if ftype == "pdf":
            dest = RAW_DIR / f"report_{slugify(title)}.pdf"
        elif ftype in ("docx", "doc"):
            dest = RAW_DIR / f"report_{slugify(title)}.txt"
        else:
            print(f"  unsupported type {ftype}: {title[:60]}")
            continue
        if dest.exists():
            skipped += 1
            continue
        try:
            resp = get(f["url"])
            resp.raise_for_status()
        except Exception as e:
            print(f"  FAILED {title[:60]}: {e}")
            failed += 1
            continue
        blob = resp.content
        if ftype == "pdf" and not blob.startswith(b"%PDF"):
            print(f"  not a PDF, skipped: {title[:60]}")
            failed += 1
            continue
        digest = hashlib.sha256(blob).hexdigest()
        if digest in hashes:
            skipped += 1
            continue
        hashes.add(digest)
        if ftype in ("docx", "doc"):
            dest.write_text(docx_to_text(blob), encoding="utf-8")
        else:
            dest.write_bytes(blob)
        new += 1
        print(f"  saved {dest.name} ({len(blob):,} bytes)")

    print(f"\n{new} new, {skipped} skipped, {failed} failed -> {RAW_DIR}")
    print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
