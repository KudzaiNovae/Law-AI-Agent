FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Layers ordered by change frequency (rarest first): a code edit
# ---- then only rebuilds/pushes the small code layers at the bottom;
# ---- the multi-GB data layers keep their digests and ACR skips them.

# Chroma's default embedding model, vendored from the local cache so the
# build never downloads it (the CDN download used to time out on slow
# links). Refresh with:
#   cp -r ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2 vendor/onnx_models/
COPY vendor/onnx_models/ /root/.cache/chroma/onnx_models/
# Verify it loads (runs offline — model is already in the cache path).
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as F; F()(['warm up'])"

# Raw source documents — served by /api/download/ so readers can open the
# original gazetted PDFs from the app.
COPY data/raw/ data/raw/

# Index data — built locally with ingest.run + build_index, shipped baked in.
COPY data/chroma/ data/chroma/
COPY data/processed/chunks.jsonl data/processed/chunks.jsonl
# Knowledge graph (graph/build_graph.py) — optional but ship it if built
COPY data/processed/graph.jsonl* data/processed/

# Code — changes most often, kilobytes to push.
#
# The glob is deliberate. This line used to name three modules, and when
# decompose.py and verify.py were added nobody remembered to add them
# here — so the image built and pushed clean and then crash-looped on
# ModuleNotFoundError, which is a 9 GB round trip to discover a missing
# 20 kB file. A hand-maintained list of imports is a list that goes
# stale; a glob cannot.
COPY *.py ./
COPY index/ index/
COPY ingest/ ingest/
COPY web/ web/

# Fail the build, not the deployment. Importing the app here turns a
# missing module or a syntax error into a red build on the laptop that
# made it, seconds after the edit, instead of a CrashLoopBackOff found
# half an hour later in Azure. The import is side-effect free: it opens
# no database and writes nothing.
RUN python -c "import server"

ENV PROVIDER=gemini
EXPOSE 8000

# Single worker only: sessions and the retriever live in process memory.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
