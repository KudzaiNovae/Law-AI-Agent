"""Extract plain text from raw documents (PDF, HTML, TXT).

Scanned PDFs (common for older Gazettes/SIs) yield no text here —
those need OCR (e.g. `ocrmypdf input.pdf output.pdf`) first.
"""
from pathlib import Path

from bs4 import BeautifulSoup


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _from_pdf(path)
    if suffix in (".html", ".htm"):
        return _from_html(path)
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.name}")


def _from_pdf(path: Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    if len(text.strip()) < 200:
        print(f"WARNING: {path.name} extracted almost no text — likely a "
              f"scanned PDF. Run OCR first (ocrmypdf).")
    return text


def _from_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"),
                         "html.parser")
    # ZimLII Akoma Ntoso: main content lives in <div class="akn-akomaNtoso">
    main = (soup.find(class_="akn-akomaNtoso")
            or soup.find("main")
            or soup.find("article")
            or soup.body
            or soup)
    for tag in main.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return main.get_text(separator="\n", strip=True)
