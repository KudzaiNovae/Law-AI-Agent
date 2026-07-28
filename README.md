# zim-law-agent

RAG agent that answers questions about the Constitution of Zimbabwe, Acts,
statutory instruments, and policies — with citations.

## Architecture

```
data/raw/ (PDF/HTML/TXT)
   │  ingest/extract.py      text extraction (pdfplumber / BeautifulSoup)
   │  ingest/chunker.py      section-aware chunking + legal metadata
   ▼
data/processed/chunks.jsonl
   │  index/build_index.py   Chroma vector index (local embeddings)
   ▼
index/retriever.py           hybrid search: vector + BM25, RRF fusion
   ▼
agent.py                     Claude tool-use loop (search_legislation)
cli.py                       chat interface
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

## Run

```bash
python -m ingest.download    # fetch Constitution from ZimLII
python -m ingest.run         # extract + chunk -> chunks.jsonl
python -m index.build_index  # build vector index
python cli.py "What does the Constitution say about freedom of expression?"
```

## Adding more documents

1. Drop PDFs/HTML into `data/raw/` (sources: zimlii.org, veritaszim.net,
   Government Gazette). Name files so type is inferred (`labour_act.pdf`,
   `si_127_2021.pdf`) or add exact titles to `MANIFEST` in `ingest/run.py`.
2. Scanned PDFs need OCR first: `ocrmypdf in.pdf out.pdf`
3. Re-run `python -m ingest.run && python -m index.build_index`

## Roadmap ideas

- **Versioning**: track in-force status; amendments/repeals of SIs are
  frequent — an answer citing a repealed SI is a bug.
- **Freshness**: scheduled scraper for new Gazette SIs (Veritas publishes
  them promptly).
- **Better embeddings**: swap Chroma's default MiniLM for a hosted model
  (e.g. Voyage) if recall is weak.
- **Eval set**: build 30–50 Q/A pairs with known citations; measure
  retrieval hit-rate before tweaking anything.
- **API**: wrap `LawAgent` in FastAPI for a web frontend.

**Disclaimer**: outputs are legal information, not legal advice.
