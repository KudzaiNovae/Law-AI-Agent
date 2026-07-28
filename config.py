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
        f"{GEMINI_MODEL},gemini-flash-latest,gemini-2.0-flash").split(","):
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

for d in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)
