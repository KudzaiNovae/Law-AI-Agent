"""One-off / maintenance: move duplicate files out of data/raw/ into
data/raw_dupes/ (reversible, nothing is deleted).

Two duplicate classes:
  1. identical content (sha256) under different names
  2. same instrument (SI/Act/GN number + year) from different sources
     (Veritas vs JSC, or JSC's "nom" + full-title twin uploads)

Keeper preference: non-"nom" filename, then larger file.

Usage:
    python -m ingest.dedupe_raw           # dry run (default)
    python -m ingest.dedupe_raw --apply
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, DATA_DIR
from ingest.scrape_jsc import FILE_KEY_RES

DUPES_DIR = DATA_DIR / "raw_dupes"


def file_key(name: str) -> str | None:
    name = name.lower()
    m = FILE_KEY_RES[0].match(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{int(m.group(3))}{m.group(4)}"
    m = FILE_KEY_RES[1].match(name)
    if m:
        return f"{m.group(1)}-{m.group(4)}-{int(m.group(2))}{m.group(3)}"
    return None


def keeper_sort(p: Path):
    """Sort so the best keeper comes first: Veritas-style names (they
    produce clean titles like 'SI 2026-024 ...'), then non-"nom",
    then the larger file."""
    veritas_style = bool(FILE_KEY_RES[0].match(p.name.lower()))
    return (not veritas_style, "nom" in p.stem.lower(),
            -p.stat().st_size, len(p.name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually move files (default: dry run)")
    args = ap.parse_args()

    files = [p for p in sorted(RAW_DIR.iterdir())
             if p.is_file() and not p.name.startswith((".", "_"))]

    to_move: dict[Path, str] = {}

    # 1. identical content
    by_hash: dict[str, list[Path]] = {}
    for p in files:
        by_hash.setdefault(
            hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p)
    for group in by_hash.values():
        if len(group) < 2:
            continue
        group.sort(key=keeper_sort)
        for p in group[1:]:
            to_move[p] = f"same content as {group[0].name}"

    # 2. same instrument key
    by_key: dict[str, list[Path]] = {}
    for p in files:
        if p in to_move:
            continue
        k = file_key(p.name)
        if k:
            by_key.setdefault(k, []).append(p)
    for k, group in by_key.items():
        if len(group) < 2:
            continue
        group.sort(key=keeper_sort)
        for p in group[1:]:
            to_move[p] = f"same instrument ({k}) as {group[0].name}"

    if not to_move:
        print("No duplicates found.")
        return
    for p, why in sorted(to_move.items()):
        print(f"{'MOVE' if args.apply else 'would move'}  {p.name}\n"
              f"        -> {why}")
    print(f"\n{len(to_move)} duplicates of {len(files)} files")

    if args.apply:
        DUPES_DIR.mkdir(exist_ok=True)
        for p in to_move:
            p.rename(DUPES_DIR / p.name)
        print(f"Moved to {DUPES_DIR}")
        print("Next: python -m ingest.run && python -m index.build_index")
    else:
        print("Dry run — rerun with --apply to move them.")


if __name__ == "__main__":
    main()
