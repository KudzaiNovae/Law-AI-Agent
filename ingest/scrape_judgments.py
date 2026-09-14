"""Scrape court judgments into data/raw/ (doc_type=judgment).

Three sources, three very different access levels:

  zimlii     zimlii.org blocks every judgment *page* with Cloudflare (403),
             but its search API is open. We pull recent judgments
             (ordering=-date) and save a metadata "stub" per judgment:
             neutral citation, court, date, one-line holding, flynotes
             (which cite the statutes applied) and a text snippet.
             No full text — the stub tells the agent the case exists,
             what it held, and links to zimlii.org.

  lawportal  lawportalzim.co.zw — open, ~4k judgments with FULL TEXT,
             topic-indexed (mostly 2000-2021). Bulk of the corpus.

  veritas    veritaszim.net/courtjudgements — ~75 landmark constitutional
             /public-interest judgment PDFs.

Files are saved as judgment_<source>_*.html/.pdf and recorded in
data/raw/_manifest.json with doc_type="judgment" (ingest/run.py reads it).
Resumable: existing files are skipped.

Usage:
    python -m ingest.scrape_judgments --source zimlii --years 2026 2025
    python -m ingest.scrape_judgments --source veritas
    python -m ingest.scrape_judgments --source lawportal
    python -m ingest.scrape_judgments                 # all three
"""
import argparse
import html as html_mod
import json
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

try:
    from curl_cffi import requests as creq
except ImportError:
    sys.exit("pip install curl_cffi")

MANIFEST_PATH = RAW_DIR / "_manifest.json"
DELAY = 1.5


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:70]


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {}


def save_manifest(m: dict) -> None:
    # Merge with what's on disk: several scrapers may run concurrently
    # and each holds its own copy — a blind write would clobber the
    # others' entries. Keys are per-file, so update() is safe.
    merged = load_manifest()
    merged.update(m)
    m.update(merged)
    MANIFEST_PATH.write_text(
        json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")


def get(sess, url: str, **kw):
    time.sleep(DELAY)
    r = sess.get(url, timeout=60, **kw)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------- zimlii
ZIMLII_API = "https://zimlii.org/search/api/documents/"
# The API hard-caps every query at 10 pages x 10 hits = 100 results
# (page 11 -> HTTP 400), and its date filters are ignored. So we segment
# by court + year + a handful of common search terms + both sort orders,
# and dedupe on the output filename. Overlap is free; the cap is not.
ZIMLII_COURTS = [
    "Harare High Court", "Bulawayo High Court",
    "Supreme Court of Zimbabwe", "Constitutional Court of Zimbabwe",
    "Labour Court", "Masvingo High Court", "Mutare High Court",
    "Chinhoyi High Court",
]
ZIMLII_TERMS = ["court", "applicant", "accused", "appeal", "costs"]


def _zimlii_hits(sess, params: str, manifest: dict) -> tuple[int, int]:
    """Fetch up to 10 pages for one query segment. Returns (new, total)."""
    new = total = 0
    for page in range(1, 11):
        try:
            data = get(sess, f"{ZIMLII_API}?nature=Judgment&{params}"
                             f"&page={page}").json()
        except Exception:
            break
        soup = BeautifulSoup(data.get("results_html", ""), "html.parser")
        hits = soup.select("li.hit")
        if not hits:
            break
        total = data.get("count", 0)
        for li in hits:
            a = li.select_one("a[data-frbr-uri]")
            if not a:
                continue
            uri = a["data-frbr-uri"]          # /akn/zw/judgment/zwhhc/2026/179/eng@...
            m = re.match(r"/akn/zw/judgment/(\w+)/(\d{4})/(\d+)/", uri)
            if not m:
                continue
            court_code, year, num = m.group(1), m.group(2), m.group(3)
            fname = f"judgment_zimlii_{court_code}_{year}_{num}.html"
            dest = RAW_DIR / fname
            if dest.exists():
                continue
            title = a.get_text(" ", strip=True)
            spans = [s.get_text(" ", strip=True)
                     for s in li.select("span.me-3")]
            date = spans[0] if spans else ""
            court = spans[1] if len(spans) > 1 else court_code.upper()
            holding = li.select_one("div.my-2")
            holding = holding.get_text(" ", strip=True) if holding else ""
            flynotes = [f.get_text(" ", strip=True)
                        for f in li.select("ul.flynotes li")]
            snippet = li.select_one("div.snippet")
            snippet = snippet.get_text(" ", strip=True) if snippet else ""

            e = html_mod.escape
            parts = [f"<html><body><h1>{e(title)}</h1>",
                     f"<p>Court: {e(court)}. Judgment date: {e(date)}.</p>"]
            if holding:
                parts.append(f"<p>Held: {e(holding)}</p>")
            if flynotes:
                parts.append("<p>Flynote (issues and statutes considered): "
                             + " | ".join(e(f) for f in flynotes) + "</p>")
            if snippet:
                parts.append(f"<p>Extract: {e(snippet)}</p>")
            parts.append(
                f"<p>Full judgment: https://zimlii.org{e(uri)} "
                "(full text not indexed here — this is a case summary "
                "from ZimLII search metadata).</p></body></html>")
            dest.write_text("\n".join(parts), encoding="utf-8")
            manifest[fname] = {
                "title": title, "doc_type": "judgment",
                "source": "zimlii.org", "frbr_uri": uri,
                "court": court, "date": date,
            }
            new += 1
        if page * 10 >= total:
            break
    return new, total


def scrape_zimlii(manifest: dict, years: list[int]) -> int:
    """Recent-judgment stubs from the (open) search API, segmented to
    work around the 100-results-per-query cap."""
    from urllib.parse import quote
    sess = creq.Session(impersonate="chrome124")
    new = 0
    for year in years:
        for court in ZIMLII_COURTS:
            got_before = new
            count = None
            for order in ("-date", "date"):
                for term in ZIMLII_TERMS:
                    params = (f"search={quote(term)}&year={year}"
                              f"&court={quote(court)}&ordering={order}")
                    n, total = _zimlii_hits(sess, params, manifest)
                    new += n
                    count = total if count is None else max(count, total)
                    # segment fully covered by one query — no need for
                    # more terms/orderings
                    if count is not None and count <= 100:
                        break
                if count is not None and count <= 100:
                    break
            print(f"  zimlii {year} {court}: ~{count or 0} listed, "
                  f"{new - got_before} new")
            save_manifest(manifest)
    return new


def clean_zimlii_content(content) -> None:
    """Strip ZimLII page chrome from a div.document-content (in place):
    skip-links, the Summary/TOC/Search navigation panel (which duplicates
    the Held + flynote we already put in the stub header), and make
    site-relative links absolute so they don't resolve against our own
    server when the file is viewed locally."""
    for sel in (".skip-links", "#navigation-content", "#navigation-column",
                "[data-offcanvas-body]", "nav", ".toc"):
        for el in content.select(sel):
            el.decompose()
    for a in content.find_all("a", href=True):
        if a["href"].startswith("/"):
            a["href"] = "https://zimlii.org" + a["href"]


def upgrade_zimlii_full(manifest: dict, limit: int | None) -> int:
    """Replace zimlii summary stubs with full judgment text.

    Needs a Cloudflare clearance borrowed from a real browser session:
    set ZIMLII_COOKIE (the whole Cookie header, incl. cf_clearance) and
    ZIMLII_UA (the matching User-Agent) in .env. cf_clearance is tied to
    the UA and IP and expires — rerun with a fresh cookie to continue;
    already-upgraded files are skipped.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    cookie, ua = os.getenv("ZIMLII_COOKIE"), os.getenv("ZIMLII_UA")
    if not cookie or not ua:
        print("  ZIMLII_COOKIE / ZIMLII_UA not set in .env — skipping")
        return 0
    headers = {"Cookie": cookie, "User-Agent": ua}
    sess = creq.Session(impersonate="chrome124", headers=headers)

    originals = RAW_DIR / "originals"
    originals.mkdir(exist_ok=True)

    def fetch_pdf(meta: dict, fname: str) -> None:
        """Original judgment (PDF or, usually, the registry's .docx)
        -> data/raw/originals/ for the app's document viewer."""
        stem = Path(fname).stem
        if any((originals / (stem + ext)).exists()
               for ext in (".pdf", ".docx", ".rtf")):
            return
        try:
            r = get(sess, "https://zimlii.org" + meta["frbr_uri"]
                    + "/source")
        except Exception:
            return  # not every judgment has a source file
        blob = r.content
        ctype = r.headers.get("content-type", "")
        if blob.startswith(b"%PDF"):
            ext = ".pdf"
        elif blob.startswith(b"PK") and "wordprocessing" in ctype:
            ext = ".docx"
        elif ctype.startswith(("application/rtf", "text/rtf")):
            ext = ".rtf"
        else:
            return
        (originals / (stem + ext)).write_bytes(blob)
        meta["original"] = stem + ext

    # already-upgraded files are detected by content, not the manifest
    # flag — concurrent scraper runs can race on the manifest
    todo = []
    for f, m in manifest.items():
        if (m.get("doc_type") != "judgment"
                or m.get("source") != "zimlii.org"
                or not m.get("frbr_uri")):
            continue
        p = RAW_DIR / f
        if not p.exists():
            continue
        if (m.get("full_text") == "unavailable"
                or "<h2>Full judgment</h2>" in
                p.read_text(encoding="utf-8", errors="replace")):
            if m.get("full_text") != "unavailable":
                m["full_text"] = "yes"
            fetch_pdf(m, f)   # backfill PDFs for already-upgraded stubs
            continue
        todo.append((f, m))
    print(f"  {len(todo)} stubs to upgrade")
    new = 0
    for fname, meta in todo:
        if limit and new >= limit:
            break
        url = "https://zimlii.org" + meta["frbr_uri"]
        try:
            r = get(sess, url)
        except Exception as e:
            if "403" in str(e):
                print(f"  403 after {new} upgrades — cf_clearance expired. "
                      "Refresh ZIMLII_COOKIE in .env and rerun.")
                break
            print(f"  {fname}: {e}")
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        content = soup.select_one("div.document-content")
        if content:
            clean_zimlii_content(content)
        text = content.get_text(" ", strip=True) if content else ""
        fetch_pdf(meta, fname)
        if len(text) < 1200:      # PDF-only judgment or empty shell
            meta["full_text"] = "unavailable"
            continue
        stub = (RAW_DIR / fname).read_text(encoding="utf-8")
        # drop the "full text not indexed here" trailer from the stub
        stub = re.sub(r"<p>Full judgment:.*?</p>\s*</body></html>", "",
                      stub, flags=re.S)
        e = html_mod.escape
        (RAW_DIR / fname).write_text(
            stub + "\n<h2>Full judgment</h2>\n" + str(content)
            + "</body></html>", encoding="utf-8")
        meta["full_text"] = "yes"
        new += 1
        if new % 25 == 0:
            save_manifest(manifest)
            print(f"  ...{new} upgraded")
    return new


# --------------------------------------------------------------- veritas
VERITAS = "https://www.veritaszim.net"


def scrape_veritas(manifest: dict) -> int:
    sess = creq.Session(impersonate="chrome124")
    new, page = 0, 0
    seen_nodes: set[str] = set()
    while True:
        try:
            r = get(sess, f"{VERITAS}/courtjudgements?page={page}")
        except Exception as e:
            print(f"  veritas page {page} failed: {e}")
            break
        soup = BeautifulSoup(r.text, "html.parser")
        rows = [a for a in soup.select("a[href^='/node/']")
                if a.get_text(strip=True)
                and a["href"] not in ("/node/1", "/node/2", "/node/3",
                                      "/node/3495", "/node/7625")]
        found = 0
        for a in rows:
            node = a["href"]
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            title = a.get_text(" ", strip=True)
            stem = f"judgment_veritas_{slugify(title)}"
            dest = RAW_DIR / f"{stem}.pdf"
            if any((RAW_DIR / f"{stem}{ext}").exists()
                   for ext in (".pdf", ".doc", ".docx", ".rtf", ".html")):
                found += 1
                continue
            try:
                node_html = get(sess, VERITAS + node).text
            except Exception as e:
                print(f"  node {node} failed: {e}")
                continue
            att = re.findall(
                r'href="(https?://[^"]+\.(pdf|docx?|rtf))"', node_html,
                re.I)
            saved_dest = None
            if att:
                url_a, ext = att[0][0], att[0][1].lower()
                try:
                    blob = get(sess, url_a).content
                except Exception as e:
                    print(f"  attachment for {title[:40]} failed: {e}")
                    continue
                ok = (blob.startswith(b"%PDF") if ext == "pdf"
                      else len(blob) > 2000)
                if ok:
                    saved_dest = RAW_DIR / f"{dest.stem}.{ext}"
                    saved_dest.write_bytes(blob)
            if saved_dest is None:
                # no usable attachment — save the node page text itself
                nsoup = BeautifulSoup(node_html, "html.parser")
                body = (nsoup.select_one("div.content")
                        or nsoup.select_one("article") or nsoup)
                text = body.get_text("\n", strip=True)
                if len(text) < 1200:
                    continue
                e = html_mod.escape
                saved_dest = RAW_DIR / f"{dest.stem}.html"
                saved_dest.write_text(
                    f"<html><body><h1>{e(title)}</h1>\n<pre>{e(text)}"
                    f"</pre>\n<p>Source: {VERITAS}{e(node)}</p>"
                    "</body></html>", encoding="utf-8")
            manifest[saved_dest.name] = {
                "title": f"Judgment: {title}", "doc_type": "judgment",
                "source": "veritaszim.net", "url": VERITAS + node,
            }
            new += 1
            found += 1
            print(f"  saved {saved_dest.name[:70]}")
        nxt = soup.select_one("a[href*='courtjudgements?page=" +
                              str(page + 1) + "']")
        if not nxt or not found:
            break
        page += 1
    return new


# ------------------------------------------------------------------ jsc
# The JSC website (www.jsc.org.zw/judgements) is an Angular app backed by
# an open JSON API with 11k+ official judgment PDFs, updated within days
# of hand-down. Fields: judge, category, court, citation,
# judgement_number, case_number, parties, year (a full date),
# judgement_file (PDF url). Newest first. The API host has a broken cert
# chain, hence verify=False.
JSC_API = "https://jsc.org.zw:8222/judgements/"
# Some rows carry the court's numeric id instead of its name (per the
# /courts/ endpoint).
JSC_COURT_IDS = {
    "1": "Constitutional Court", "2": "Harare Supreme Court",
    "4": "Harare Labour Court", "5": "Administrative Court",
    "6": "Commercial Court", "7": "Sheriff of the High Court",
    "8": "Magistrate Court", "9": "Harare High Court",
}


def scrape_jsc_judgments(manifest: dict, limit: int | None) -> int:
    sess = creq.Session(impersonate="chrome124", verify=False,
                        headers={"Accept": "application/json"})
    new = page = 0
    url = f"{JSC_API}?page=1&page_size=100"
    while url:
        try:
            d = get(sess, url).json()
        except Exception as e:
            print(f"  jsc page {page + 1} failed: {e}")
            break
        page += 1
        for row in d.get("results", []):
            if limit and new >= limit:
                url = None
                break
            num = (row.get("judgement_number") or f"id{row['id']}")
            fname = f"judgment_jsc_{slugify(num)}.pdf"
            dest = RAW_DIR / fname
            if dest.exists() or not row.get("judgement_file"):
                continue
            try:
                blob = get(sess, row["judgement_file"]).content
            except Exception as e:
                print(f"  {num} download failed: {e}")
                continue
            if not blob.startswith(b"%PDF"):
                continue
            dest.write_bytes(blob)
            parties = re.sub(r"\s+", " ", row.get("parties") or "").strip()
            court = str(row.get("court") or "")
            court = JSC_COURT_IDS.get(court, court)
            title = (f"Judgment {num}: {parties[:150]} "
                     f"({court}, {row.get('year') or ''})")
            manifest[fname] = {
                "title": title, "doc_type": "judgment",
                "source": "jsc.org.zw",
                "court": court,
                "judge": row.get("judge") or "",
                "case_number": row.get("case_number") or "",
                "date": row.get("year") or "",
            }
            new += 1
            if new % 25 == 0:
                save_manifest(manifest)
                print(f"  ...{new} saved (page {page})")
        if url:
            url = d.get("next")
    return new


# ------------------------------------------------------------- lawportal
LAWPORTAL = "https://lawportalzim.co.zw"


def lawportal_case_urls(sess) -> list[str]:
    urls: dict[str, None] = {}
    for section in ("civil", "criminal"):
        page = 0
        while True:
            try:
                r = get(sess, f"{LAWPORTAL}/cases/{section}?pg={page}")
            except Exception as e:
                print(f"  {section} pg={page} failed: {e}")
                break
            # real case pages are /cases/<section>/<id>/<applicant>/<respondent>
            # — single-slug URLs are topic/category pages (junk)
            hrefs = set(re.findall(
                rf'href="(/cases/{section}/\d+/[^"/]+/[^"/]+)"', r.text))
            before = len(urls)
            for h in hrefs:
                urls.setdefault(h)
            print(f"  lawportal {section} pg={page}: "
                  f"{len(urls) - before} new urls ({len(urls)} total)")
            if len(urls) == before:
                break
            page += 1
    return list(urls)


def scrape_lawportal(manifest: dict, limit: int | None) -> int:
    sess = creq.Session(impersonate="chrome124")
    print("  listing lawportal cases...")
    urls = lawportal_case_urls(sess)
    print(f"  {len(urls)} case urls")
    new = 0
    for u in urls:
        cid = re.search(r"/cases/\w+/(\d+)/", u).group(1)
        fname = f"judgment_lawportal_{cid}.html"
        dest = RAW_DIR / fname
        if dest.exists():
            continue
        if limit and new >= limit:
            break
        try:
            r = get(sess, LAWPORTAL + u)
        except Exception as e:
            print(f"  case {cid} failed: {e}")
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "form"]):
            t.decompose()
        h1s = [h for h in soup.find_all("h1")
               if "law portal" not in h.get_text(strip=True).lower()]
        title = h1s[0].get_text(" ", strip=True) if h1s else ""
        # Case pages are titled "<CITATION> - <X> vs <Y>"; topic/category
        # pages (same URL shape) have a bare id or topic name — skip them.
        if not (re.match(r"[A-Z]{2,5}\d+-\d+", title)
                or " vs " in title.lower()):
            continue
        page_div = soup.find("div", class_="page") or soup.body or soup
        text = page_div.get_text("\n", strip=True)
        # drop site chrome before the case title and after the categories menu
        i = text.find(title)
        if i > 0:
            text = text[i:]
        j = text.find("Back\nMain menu")
        if j > 0:
            text = text[:j]
        if len(text) < 1000:      # category page / empty shell
            continue
        e = html_mod.escape
        dest.write_text(
            f"<html><body><h1>{e(title)}</h1>\n<pre>{e(text)}</pre>\n"
            f"<p>Source: {LAWPORTAL}{e(u)}</p></body></html>",
            encoding="utf-8")
        manifest[fname] = {
            "title": title, "doc_type": "judgment",
            "source": "lawportalzim.co.zw", "url": LAWPORTAL + u,
        }
        new += 1
        if new % 25 == 0:
            save_manifest(manifest)
            print(f"  ...{new} saved")
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                    choices=["zimlii", "zimlii-full", "veritas",
                             "lawportal", "jsc"],
                    help="default: zimlii + veritas + lawportal + jsc. "
                         "'zimlii-full' upgrades stubs to full text "
                         "(needs ZIMLII_COOKIE/ZIMLII_UA in .env).")
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2026, 2025, 2024],
                    help="years to pull from zimlii (default: last three)")
    ap.add_argument("--limit", type=int,
                    help="max new lawportal judgments this run")
    args = ap.parse_args()

    manifest = load_manifest()
    total = 0
    if args.source in (None, "zimlii"):
        print("== zimlii (recent judgment summaries) ==")
        total += scrape_zimlii(manifest, args.years)
        save_manifest(manifest)
    if args.source == "zimlii-full":
        print("== zimlii (upgrade stubs to full text) ==")
        total += upgrade_zimlii_full(manifest, args.limit)
        save_manifest(manifest)
    if args.source in (None, "veritas"):
        print("== veritas (landmark judgment PDFs) ==")
        total += scrape_veritas(manifest)
        save_manifest(manifest)
    if args.source in (None, "jsc"):
        print("== jsc (official judgment PDFs) ==")
        total += scrape_jsc_judgments(manifest, args.limit)
        save_manifest(manifest)
    if args.source in (None, "lawportal"):
        print("== lawportal (full-text judgments) ==")
        total += scrape_lawportal(manifest, args.limit)
        save_manifest(manifest)
    print(f"\n{total} new judgment files -> {RAW_DIR}")
    print("Next: python -m ingest.run && python -m index.build_index")


if __name__ == "__main__":
    main()
