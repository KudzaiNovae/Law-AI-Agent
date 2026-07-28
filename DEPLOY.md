# OneGov — run & deploy

## Local

```bash
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Data refresh (only when you want new gazettes):

```bash
python -m ingest.scrape_veritas   # needs fresh VERITAS_COOKIE in .env
python -m ingest.scrape_jsc
python -m ingest.run && python -m index.build_index
# running server picks up the new index automatically
```

## Docker

Build the index locally first. The image ships `data/processed`,
`data/chroma` and `data/raw` (raw PDFs are served by `/api/download/`
so readers can open the original documents).

```bash
docker build -t onegov .
docker run -p 8000:8000 \
  -e PROVIDER=gemini \
  -e GEMINI_API_KEY=... \
  -e GEMINI_MODEL=gemini-2.5-flash \
  onegov
```

To update the corpus in production: re-scrape + re-index locally, rebuild
the image, redeploy.

## Hosting

Any single-container host works: a VPS with Caddy in front, Fly.io,
Railway, Render. Needs ~1 GB RAM (Chroma + BM25 in memory) and ~1 GB disk.

Reverse proxy: SSE streaming needs response buffering off. The server
already sends `X-Accel-Buffering: no`; with nginx also set
`proxy_buffering off;` for `/api/chat/stream`. Caddy needs nothing.

## Before going public

- **Workers**: keep `--workers 1`. Sessions and the retriever are
  in-process; multiple workers would split sessions. Scale later by moving
  session state to Redis.
- **LLM quota**: Gemini free tier (~15 req/min) will not survive public
  traffic. Use paid Gemini or `PROVIDER=anthropic`.
- **Rate limiting**: add per-IP limits (e.g. slowapi) before public
  launch, or the LLM bill/quota is your rate limiter.
- **Secrets**: `.env` is not copied into the image — pass keys as env
  vars. Rotate any key that was ever committed or hardcoded.
- **HTTPS**: terminate TLS at the proxy (Caddy does this automatically).
