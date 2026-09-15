"""Optional cross-encoder reranking for hybrid retrieval.

RRF fusion is cheap but shallow: it ranks a chunk on *where* it appeared
in two candidate lists, never on whether the chunk actually answers the
question. A cross-encoder reads (query, chunk) together and scores that
relationship directly, so the right subsection buried at RRF rank 30
can surface into the eight results the agent actually reads.

Cost: sentence-transformers + torch is ~2GB of wheels and a few hundred
ms of CPU per search, so this is opt-in via RERANK=1. Every failure path
— dependency missing, model download blocked, predict() raising — falls
back to plain RRF order. Retrieval must never hard-fail because an
optional accuracy boost is unavailable.

    pip install "sentence-transformers>=3.0"
    RERANK=1 .venv/bin/python -m uvicorn server:app
"""
import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RERANK, RERANK_MODEL, RERANK_MAX_CHARS

_lock = threading.Lock()
_model = None
_off = not RERANK        # sticky: one failure disables for the process


def enabled() -> bool:
    """True if reranking is configured and hasn't failed yet.

    Cheap enough to call per search — the model is loaded lazily on the
    first scores() call, not here, so a server that never searches never
    pays the load cost.
    """
    return not _off


def _load():
    global _model, _off
    if _model is not None or _off:
        return _model
    with _lock:
        if _model is not None or _off:   # another thread got there first
            return _model
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(RERANK_MODEL, max_length=512)
            print(f"  [rerank] loaded {RERANK_MODEL}")
        except Exception as exc:
            _off = True
            print(f"  [rerank] disabled ({exc.__class__.__name__}: {exc}); "
                  "using RRF ranking")
    return _model


def _to_prob(x: float) -> float:
    """Normalise a model output to 0..1.

    CrossEncoder applies sigmoid itself for single-label models, but that
    depends on the checkpoint's config, so only squash values that are
    clearly raw logits.
    """
    if 0.0 <= x <= 1.0:
        return x
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def scores(query: str, texts: list[str]) -> list[float] | None:
    """Relevance of each text to the query, 0..1, aligned with `texts`.

    Returns None when reranking is unavailable, which callers must read
    as "keep the order you already have".
    """
    if _off or not texts:
        return None
    model = _load()
    if model is None:
        return None
    pairs = [[query, t[:RERANK_MAX_CHARS]] for t in texts]
    try:
        raw = model.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception as exc:
        # Not sticky: a single bad batch shouldn't disable reranking for
        # the life of the process, unlike a failed load.
        print(f"  [rerank] predict failed ({exc}); using RRF order")
        return None
    return [_to_prob(float(s)) for s in raw]
