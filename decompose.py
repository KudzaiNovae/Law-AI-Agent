"""Query decomposition: turn one compound question into targeted searches.

"Can my landlord evict me and keep my deposit?" is two legal questions
wearing one coat. Embedded as a single vector it lands somewhere between
the eviction provisions and the deposit provisions and retrieves neither
well; BM25 does no better, because the decisive keywords belong to
different instruments. Splitting it into "grounds for eviction of a
tenant" and "refund of rental deposit", searching each, then fusing the
two result sets, retrieves both halves properly.

This costs one small LLM call, so it only fires on queries that actually
look compound (see looks_compound) and answers are memoised per process.
Every failure path returns [] — the caller then searches the original
query unchanged, exactly as before.
"""
import json
import re

# Cheap pre-filter. The LLM call is small but not free, and most searches
# the agent issues are already single-topic, so only pay for the ones
# that look like they bundle several questions together.
_CONJ_RE = re.compile(
    r"\b(and|or|also|as well as|plus|versus|vs\.?|but|whilst|while|"
    r"in addition to)\b", re.I)
_MULTI_Q_RE = re.compile(r"\?.*\?", re.S)

MIN_COMPOUND_CHARS = 45
MAX_SUBQUERIES = 3

_PROMPT = """\
You split Zimbabwean legal research questions into search queries.

Return ONLY a JSON object: {"queries": ["...", "..."]}

Rules:
- At most %d queries. Prefer 2. Use 1 if the question really is one topic.
- Each query is a short keyword-style search over a corpus of Zimbabwean
  legislation, statutory instruments and judgments — not a sentence.
- Each must stand alone: no "it", "that", "the above".
- Cover DIFFERENT legal issues. Do not paraphrase one issue twice.
- Keep any exact references the user gave ("SI 118 of 2023", "section 71")
  verbatim in the query that needs them.

Question: %s"""

_cache: dict[str, list[str]] = {}
_CACHE_MAX = 512


def looks_compound(query: str) -> bool:
    """Cheap test for 'worth spending an LLM call on'.

    Deliberately conservative: a false negative just means we search the
    way we always did, while a false positive costs a call and some
    latency on every such query.
    """
    q = query.strip()
    if len(q) < MIN_COMPOUND_CHARS:
        return False
    if _MULTI_Q_RE.search(q):        # two questions in one message
        return True
    return bool(_CONJ_RE.search(q))


def _parse(raw: str) -> list[str]:
    """Pull the query list out of a model response, fences and all."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw)
    # models sometimes prepend a sentence before the object
    start = raw.find("{")
    if start > 0:
        raw = raw[start:]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    qs = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(qs, list):
        return []
    out: list[str] = []
    for q in qs:
        if not isinstance(q, str):
            continue
        q = q.strip()
        # a "sub-query" longer than the original is the model rambling
        if 3 <= len(q) <= 200 and q.lower() not in {o.lower() for o in out}:
            out.append(q)
    return out[:MAX_SUBQUERIES]


def subqueries(client, model: str, question: str) -> list[str]:
    """2-3 focused searches for a compound question, or [] to leave it be.

    [] is the "carry on as before" signal and is returned for anything
    unexpected: a question that doesn't look compound, a model error, an
    unparseable response, or a split that collapsed back to one query.
    """
    if not looks_compound(question):
        return []
    key = question.strip().lower()
    if key in _cache:
        return _cache[key]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": _PROMPT % (MAX_SUBQUERIES, question)}],
            max_tokens=200,
            temperature=0,
        )
        subs = _parse(resp.choices[0].message.content or "")
    except Exception as exc:
        print(f"  [decompose] skipped ({exc.__class__.__name__}: {exc})")
        return []
    # One query back means the splitter agreed it was already focused.
    if len(subs) < 2:
        subs = []
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[key] = subs
    return subs
