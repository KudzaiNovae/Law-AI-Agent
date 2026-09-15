"""Cross-encoder reranking for hybrid retrieval.

RRF fusion is cheap but shallow: it ranks a chunk on *where* it appeared
in two candidate lists, never on whether the chunk actually answers the
question. A cross-encoder reads (query, chunk) together and scores that
relationship directly, so the right subsection buried at RRF rank 30
can surface into the eight results the agent actually reads.

Runs as ONNX on CPU via fastembed — ~10ms a pair and no torch, so unlike
a sentence-transformers reranker it is cheap enough to leave on by
default (RERANK=0 turns it off). Every failure path — dependency
missing, model download blocked, rerank() raising — falls back to plain
RRF order. Retrieval must never hard-fail because an optional accuracy
boost is unavailable.
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
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _model = TextCrossEncoder(RERANK_MODEL)
            print(f"  [rerank] loaded {RERANK_MODEL}")
        except Exception as exc:
            _off = True
            print(f"  [rerank] disabled ({exc.__class__.__name__}: {exc}); "
                  "using RRF ranking")
    return _model


def _to_prob(x: float) -> float:
    """Normalise a model output to 0..1.

    ms-marco cross-encoders emit raw logits, but a checkpoint that has
    sigmoid baked in would emit probabilities, so only squash values
    that are clearly logits. The caller mixes these with the retrieval
    boosts, which are sized against a 0..1 scale.
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
    try:
        raw = list(model.rerank(query, [t[:RERANK_MAX_CHARS] for t in texts]))
    except Exception as exc:
        # Not sticky: a single bad batch shouldn't disable reranking for
        # the life of the process, unlike a failed load.
        print(f"  [rerank] rerank failed ({exc}); using RRF order")
        return None
    if len(raw) != len(texts):     # alignment is the caller's assumption
        return None
    return [_to_prob(float(s)) for s in raw]
