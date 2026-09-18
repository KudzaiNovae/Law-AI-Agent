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

# Gemini 2.5+ models "think" before every reply by default, with a
# dynamic budget the model sets itself. On legal questions it routinely
# burns 30-70 s of hidden reasoning per agent round (a live question
# measured 76 s before its FIRST tool call). "low" caps the budget at
# ~1024 thinking tokens, which keeps planning sharp but bounded; "none"
# disables thinking entirely. Answers stay grounded either way — they
# come from retrieved statute text, not from long chains of thought.
GEMINI_REASONING = os.getenv("GEMINI_REASONING", "low").strip().lower()
if GEMINI_REASONING in ("", "default", "dynamic"):
    GEMINI_REASONING = ""        # send nothing: provider default

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
# CPU via fastembed — no torch — so it is on by default; RERANK=0 turns
# it off. The model must therefore be an ONNX checkpoint fastembed can
# fetch, not a sentence-transformers one. See index/rerank.py; retrieval
# falls back to plain RRF when unavailable.
#
# Cost is real, not the ~10ms/pair the model card implies: measured
# ~120ms a pair at 1600 chars on one CPU core, so a 40-pool added ~5s
# to every search. 16 pairs x 800 chars keeps the accuracy win (the
# right chunk is almost always in the fused top-16) at ~5-10x less cost.
RERANK = os.getenv("RERANK", "1").lower() not in ("0", "false", "no", "off")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "16"))
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "800"))
# Reranking only pays off if the legs hand it more than it returns —
# widen them so the cross-encoder has something to dig out.
RERANK_VECTOR_K = int(os.getenv("RERANK_VECTOR_K", "30"))
RERANK_BM25_K = int(os.getenv("RERANK_BM25_K", "30"))

# Check the answer's section citations against what the tools actually
# returned, and give the model one round to fix any it cannot support
# (verify.py). Costs nothing on clean answers; on by default because a
# confidently wrong citation is this product's worst failure mode.
VERIFY_CITATIONS = os.getenv("VERIFY_CITATIONS", "1").lower() not in (
    "0", "false", "no", "off")

for d in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)
