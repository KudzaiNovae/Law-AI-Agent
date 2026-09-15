"""Central config for zim-law-agent."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # drop PDFs / HTML / txt here
PROCESSED_DIR = DATA_DIR / "processed"  # chunked JSONL lives here
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "zim_legislation"

# LLM provider: "gemini" (free tier) or "anthropic"
PROVIDER = os.getenv("PROVIDER", "gemini").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
# OpenAI-compatible endpoint — lets agent.py drive Claude and Gemini
# through the same code path (and hot-swap on quota exhaustion)
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Free-tier daily quotas are PER MODEL, so the agent rotates through this
# list: when one model's quota is exhausted it benches that model and
# tries the next. Override with comma-separated GEMINI_MODELS in .env.
GEMINI_MODELS: list[str] = []
for _m in os.getenv(
        "GEMINI_MODELS",
        f"{GEMINI_MODEL},gemini-flash-latest,gemini-3.6-flash").split(","):
    _m = _m.strip()
    if _m and _m not in GEMINI_MODELS:
        GEMINI_MODELS.append(_m)

# Local Ollama as last-resort fallback (OpenAI-compatible endpoint).
# Empty OLLAMA_MODEL = disabled, so prod (no local Ollama) is unaffected.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")

# Laws.Africa Content API (principal Acts importer)
LAWSAFRICA_TOKEN = os.getenv("LAWSAFRICA_TOKEN", "")

# Retrieval
TOP_K = 8            # results returned to the agent
VECTOR_K = 20        # candidates from vector search
BM25_K = 20          # candidates from keyword search

# Cross-encoder reranking of the fused candidate pool. Runs as ONNX on
# CPU via fastembed — no torch, ~10ms a pair — so it is cheap enough to
# leave on by default; RERANK=0 turns it off. The model must therefore be
# an ONNX checkpoint fastembed can fetch, not a sentence-transformers one.
# See index/rerank.py; retrieval falls back to plain RRF when unavailable.
RERANK = os.getenv("RERANK", "1").lower() not in ("0", "false", "no", "off")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "40"))
RERANK_MAX_CHARS = 1600   # chunk prefix fed to the cross-encoder
# Reranking only pays off if the legs hand it more than it returns —
# widen them so the cross-encoder has something to dig out. 20+20 barely
# fills a 40-pool after citation dedupe, which wastes the model.
RERANK_VECTOR_K = int(os.getenv("RERANK_VECTOR_K", "60"))
RERANK_BM25_K = int(os.getenv("RERANK_BM25_K", "60"))

# Check the answer's section citations against what the tools actually
# returned, and give the model one round to fix any it cannot support
# (verify.py). Costs nothing on clean answers; on by default because a
# confidently wrong citation is this product's worst failure mode.
VERIFY_CITATIONS = os.getenv("VERIFY_CITATIONS", "1").lower() not in (
    "0", "false", "no", "off")

for d in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)
