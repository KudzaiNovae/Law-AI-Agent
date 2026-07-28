"""Scrape legislation from veritaszim.net monthly archives into data/raw/.

Veritas archives (https://veritaszim.net/archives) list everything they
publish per month: statutory instruments (SI), general notices (GN), Acts,
plus newsletters (Bill Watch, Court Watch, Cabinet briefings). This scraper
keeps only legislation, downloads the attached PDF where one exists,
otherwise saves the page text.

Resumable: already-downloaded items are skipped, so re-running only picks
up new material — point a cron job at it.

Usage:
    python -m ingest.scrape_veritas                       # last 6 months
    python -m ingest.scrape_veritas --start 202401 --end 202607
    python -m ingest.scrape_veritas --types si,gn,act
"""
import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

BASE = "https://veritaszim.net"

# Veritas sits behind Radware bot protection (415 errors / "One moment,
# please..." JS challenge). Pass it by borrowing your browser's session:
#   1. Open https://veritaszim.net in your browser, let it load.
#   2. F12 -> Network tab -> reload -> click the first request ->
#      Request Headers -> copy the full "Cookie:" value and your
#      "User-Agent:" value.
#   3. Put them in .env:
#        VERITAS_COOKIE=<paste cookie value>
#        VERITAS_UA=<paste user-agent value>
# The cookie is tied to your IP + user agent, so both must match.
import os
from dotenv import load_dotenv
load_dotenv()

HEADERS = {
    "User-Agent": os.getenv("VERITAS_UA") or (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
}
if os.getenv("VERITAS_COOKIE"):
    HEADERS["Cookie"] = os.getenv("VERITAS_COOKIE")

DELAY = 1.0  # seconds between requests — be polite to Veritas
RETRIES = 3


class BotChallengeError(RuntimeError):
    pass

# Newsletter/commentary content we do NOT index as law
SKIP_RE = re.compile(
    r"bill watch|court watch|constitution watch|election watch|"
    r"peace watch|economic governance|post-cabinet|press briefing|"
    r"portfolio committee", re.I)

TYPE_RULES = [
    (re.compile(r"^S\.?I\.?[\s-]*\d|^SI\s", re.I), "si"),
    (re.compile(r"^GN[\s-]*\d|^GN\s", re.I), "gn"),
    (re.compile(r"\bAct\b(?!.*Bill)", 0), "act"),
]


def classify(title: str) -> str | None:
    if SKIP_RE.search(title):
        return None
    for pattern, doc_type in TYPE_RULES:
        if pattern.search(title):
            return doc_type
    return None


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:90]


SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get(url: str) -> requests.Response:
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(RETRIES):
        time.sleep(DELAY + attempt * 3)  # backoff on WAF rejections
        try:
            resp = SESSION.get(url, timeout=60)
            resp.raise_for_status()
            if b"One moment, please" in resp.content[:2000]:
                raise BotChallengeError(
                    "Blocked by Veritas bot protection. Set VERITAS_COOKIE "
                    "and VERITAS_UA in .env (see instructions at top of "
                    "ingest/scrape_veritas.py).")
            return resp
        except requests.RequestException as e:
            last_exc = e
    raise last_exc


def month_items(yyyymm: str) -> list[tuple[str, str]]:
    """All (title, node_url) pairs for one archive month, across pages."""
    items, seen, page = [], set(), 0
    while True:
        url = f"{BASE}/archives/{yyyymm}"
        if page:
            url += f"?page={page}"
        try:
            soup = BeautifulSoup(get(url).text, "html.parser")
        except requests.RequestException as e:
            print(f"  archive fetch failed ({e})")
            break
        found = 0
        for a in soup.find_all("a", href=re.compile(r"^/node/\d+$")):
            title = a.get_text(strip=True)
            href = a["href"]
            if not title or title.lower().startswith("read more"):
                continue
            if href in seen:
                continue
            seen.add(href)
            items.append((title, BASE + href))
            found += 1
        if not found:
            break
        page += 1
    return items


def save_node(title: str, url: str, doc_type: str) -> bool:
    """Download one node. Returns True if something new was saved."""
    stem = f"{doc_type}_{slugify(title)}"
    if list(RAW_DIR.glob(stem + ".*")):
        return False

    try:
        soup = BeautifulSoup(get(url).text, "html.parser")
    except requests.RequestException as e:
        print(f"  FAILED {url}: {e}")
        return False

    content = (soup.find(id="content") or soup.find(class_="node")
               or soup.body or soup)

    # Prefer an attached PDF (authoritative Gazette text)
    pdf_link = content.find("a", href=re.compile(r"/files/.+\.pdf$", re.I))
    if pdf_link:
        pdf_url = pdf_link["href"]
        if pdf_url.startswith("/"):
            pdf_url = BASE + pdf_url
        try:
            data = get(pdf_url).content
            (RAW_DIR / f"{stem}.pdf").write_bytes(data)
            print(f"  saved {stem}.pdf ({len(data):,} bytes)")
            return True
        except requests.RequestException as e:
            print(f"  pdf failed ({e}), falling back to page text")

    for tag in content.find_all(["script", "style", "nav", "form"]):
        tag.decompose()
    text = content.get_text(separator="\n", strip=True)
    if len(text) < 200:
        print(f"  skipped (no usable text): {title}")
        return False
    (RAW_DIR / f"{stem}.txt").write_text(f"{title}\n\n{text}",
                                         encoding="utf-8")
    print(f"  saved {stem}.txt ({len(text):,} chars)")
    return True


def month_range(start: str, end: str) -> list[str]:
    months, y, m = [], int(start[:4]), int(start[4:])
    while (y, m) <= (int(end[:4]), int(end[4:])):
        months.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return months


def main() -> None:
    today = date.today()
    default_start = f"{today.year - (1 if today.month <= 6 else 0)}" \
                    f"{((today.month - 7) % 12) + 1:02d}"
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=default_start, help="YYYYMM")
    ap.add_argument("--end", default=f"{today.year}{today.month:02d}")
    ap.add_argument("--types", default="si,gn,act",
                    help="comma-separated: si,gn,act")
    args = ap.parse_args()
    wanted = set(args.types.split(","))

    new = 0
    try:
        for yyyymm in month_range(args.start, args.end):
            print(f"\n== {yyyymm} ==")
            for title, url in month_items(yyyymm):
                doc_type = classify(title)
                if doc_type in wanted:
                    if save_node(title, url, doc_type):
                        new += 1
    except BotChallengeError as e:
        sys.exit(f"\nSTOPPED: {e}")
    print(f"\n{new} new documents in {RAW_DIR}")
    print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
