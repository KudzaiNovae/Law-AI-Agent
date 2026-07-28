"""Download seed documents.

Primary source: Veritas (veritaszim.net) — their consolidated Constitution
(amendments up to June 2023) is a clean text PDF. ZimLII has cleaner HTML
but blocks scripted requests (403), so grab those manually via a browser
if wanted and drop them in data/raw/.

Usage:
    python -m ingest.download
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

SOURCES = [
    {
        "url": ("https://www.veritaszim.net/sites/veritas_d/files/"
                "Constitution%20Consolidated%20%282023%29.pdf"),
        "filename": "constitution_of_zimbabwe_consolidated_2023.pdf",
        "title": "Constitution of Zimbabwe (consolidated 2023)",
        "doc_type": "constitution",
    },
    # Add more direct file URLs here, or use: python -m ingest.scrape_veritas
]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def download_all() -> None:
    for src in SOURCES:
        dest = RAW_DIR / src["filename"]
        if dest.exists():
            print(f"skip (exists): {dest.name}")
            continue
        print(f"fetching {src['url']} ...")
        try:
            resp = requests.get(src["url"], headers=HEADERS, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"  saved -> {dest.name} ({len(resp.content):,} bytes)")
        except requests.RequestException as e:
            print(f"  FAILED: {e}")
            print(f"  Manually download and save as data/raw/{src['filename']}")


if __name__ == "__main__":
    download_all()
    print("\nYou can also drop any PDF/HTML/TXT files directly into data/raw/")
