#!/usr/bin/env python
"""Daily health digest from data/chat_log.jsonl — no dependencies.

Prints chats/errors/latency/feedback for the last 24h and exits 1 if the
error rate is above ERR_THRESHOLD, so cron can email you only on bad days:

    0 7 * * * cd /path/to/law-agent && python scripts/error_digest.py \
        || echo "OneGov errors" | mail -s "OneGov alert" you@example.com
"""
import datetime
import json
import os
import sys
from pathlib import Path

ERR_THRESHOLD = float(os.getenv("ERR_THRESHOLD", "0.10"))   # 10%
HOURS = int(os.getenv("DIGEST_HOURS", "24"))
DATA = Path(os.getenv("DATA_DIR",
                      str(Path(__file__).resolve().parent.parent / "data")))


def _tail(path: Path):
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[-5000:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    cutoff = (datetime.datetime.now(datetime.UTC)
              - datetime.timedelta(hours=HOURS)).isoformat()
    chats = [c for c in _tail(DATA / "chat_log.jsonl")
             if c.get("ts", "") >= cutoff]
    fb = [f for f in _tail(DATA / "feedback.jsonl")
          if f.get("ts", "") >= cutoff]
    errors = [c for c in chats if c.get("error")]
    ok = [c for c in chats if not c.get("error")]

    print(f"OneGov digest — last {HOURS}h")
    print(f"  chats:    {len(chats)}")
    print(f"  errors:   {len(errors)}")
    if ok:
        ms = sorted(c.get("ms", 0) for c in ok)
        print(f"  avg ms:   {sum(ms) // len(ms)}   "
              f"p95 ms: {ms[int(len(ms) * 0.95) - 1]}")
    up = sum(1 for f in fb if f.get("verdict") == "up")
    print(f"  feedback: {up} up / {len(fb) - up} down")
    for e in errors[-5:]:
        print(f"  ERR {e.get('ts')}: {str(e.get('error'))[:120]}")

    rate = len(errors) / max(len(chats), 1)
    if chats and rate > ERR_THRESHOLD:
        print(f"\nALERT: error rate {rate:.0%} > {ERR_THRESHOLD:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
