"""Import consolidated Zimbabwean legislation from the Laws.Africa
Content API into data/raw/.

This fills the corpus's biggest gap: principal Acts (Labour Act, Criminal
Law Code, ...) consolidated with their amendments, plus repeal metadata.

Setup:
    1. Get an API token: https://edit.laws.africa (account settings).
    2. The Zimbabwe collection must be enabled on your account —
       email info@laws.africa (they support public-interest projects).
    3. Put LAWSAFRICA_TOKEN=... in .env

Usage:
    python -m ingest.import_lawsafrica --probe        # check access, list page 1
    python -m ingest.import_lawsafrica                # import Acts in force
    python -m ingest.import_lawsafrica --types act si # include SIs too
    python -m ingest.import_lawsafrica --include-repealed
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, LAWSAFRICA_TOKEN

API = "https://api.laws.africa/v2"
DELAY = 0.5
MANIFEST_PATH = RAW_DIR / "_manifest.json"


def session() -> requests.Session:
    if not LAWSAFRICA_TOKEN:
        sys.exit("Set LAWSAFRICA_TOKEN in .env (see module docstring).")
    s = requests.Session()
    s.headers["Authorization"] = f"Token {LAWSAFRICA_TOKEN}"
    return s


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]


def work_type(frbr_uri: str) -> str:
    """/akn/zw/act/2004/23 -> act; /akn/zw/act/si/2020/5 -> si;
    /akn/zw/act/gn/... -> gn."""
    m = re.match(r"/akn/zw/act/(si|gn|ord)/", frbr_uri)
    if m:
        return m.group(1)
    if frbr_uri.startswith("/akn/zw/act/"):
        return "act"
    return "other"


def list_works(s: requests.Session) -> list[dict]:
    works, url = [], f"{API}/akn/zw/.json"
    while url:
        time.sleep(DELAY)
        resp = s.get(url, timeout=60)
        if resp.status_code == 403:
            sys.exit("403: your token lacks access to the Zimbabwe "
                     "collection. Email info@laws.africa to enable it.")
        resp.raise_for_status()
        data = resp.json()
        works.extend(data.get("results", []))
        url = data.get("next")
        print(f"  listed {len(works)}/{data.get('count', '?')}")
    return works


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def probe(s: requests.Session) -> None:
    resp = s.get(f"{API}/akn/zw/.json", timeout=60)
    print("HTTP", resp.status_code)
    if resp.status_code != 200:
        print(resp.text[:500])
        if resp.status_code == 403:
            print("\nToken valid but Zimbabwe collection not enabled — "
                  "email info@laws.africa.")
        return
    data = resp.json()
    print(f"{data.get('count')} works available")
    for w in data.get("results", [])[:5]:
        print(f"  [{work_type(w['frbr_uri'])}] {w.get('title')}"
              f"{'  (REPEALED)' if w.get('repeal') else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--types", nargs="+", default=["act"],
                    choices=["act", "si", "gn", "ord"])
    ap.add_argument("--include-repealed", action="store_true")
    args = ap.parse_args()

    s = session()
    if args.probe:
        probe(s)
        return

    works = list_works(s)
    manifest = load_manifest()
    new = skipped = 0

    for w in works:
        uri = w.get("frbr_uri", "")
        wt = work_type(uri)
        if wt not in args.types:
            continue
        repeal = w.get("repeal")
        if repeal and not args.include_repealed:
            skipped += 1
            continue
        title = w.get("title", uri)
        # /akn/zw/act/2004/23 -> act_2004_23_<slug>.html
        m = re.search(r"/(\d{4})/([\w-]+)$", uri)
        year, num = (m.group(1), m.group(2)) if m else ("0000", "x")
        dest = RAW_DIR / f"{wt}_{year}_{num}_{slugify(title)}.html"
        if dest.exists():
            continue

        expr_uri = w.get("expression_frbr_uri") or f"{uri}/eng"
        try:
            time.sleep(DELAY)
            resp = s.get(f"{API}{expr_uri}.html", timeout=120)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
        except requests.RequestException as e:
            print(f"  FAILED {title[:60]}: {e}")
            continue

        manifest[dest.name] = {
            "title": title,
            "doc_type": wt,
            "expression_date": w.get("expression_date"),
            "repealed": bool(repeal),
            "source": "laws.africa",
            "frbr_uri": uri,
        }
        new += 1
        print(f"  saved {dest.name}")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{new} works imported, {skipped} repealed skipped")
    print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
