#!/usr/bin/env python3
"""Copy generated example outputs into the book sources (``make export``).

For every ``examples/<chNN>_*/output/`` folder, ``*.csv`` files are copied to
``<book root>/data/<chNN>/`` and ``*.pdf`` files to ``<book root>/figures/<chNN>/``,
where ``<chNN>`` is the chapter prefix of the example folder name.  The book root
defaults to the parent of the repository (``..``); pass ``--book-root`` otherwise.
The book compiles from these copies without the repository.

usage:  python tools/export_outputs.py [--book-root PATH] [--dry-run]
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHAPTER_RE = re.compile(r"^(ch\d+)")


def export(book_root: Path, dry_run: bool = False) -> int:
    n = 0
    for example in sorted((REPO / "examples").iterdir()):
        m = CHAPTER_RE.match(example.name)
        out = example / "output"
        if not (example.is_dir() and m and out.is_dir()):
            continue
        chapter = m.group(1)
        for suffix, target in (("*.csv", book_root / "data" / chapter), ("*.pdf", book_root / "figures" / chapter)):
            for src in sorted(out.glob(suffix)):
                dst = target / src.name
                print(f"{src.relative_to(REPO)} -> {dst}")
                if not dry_run:
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                n += 1
    print(f"{n} file(s) {'would be ' if dry_run else ''}exported to {book_root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book-root", default=str(REPO.parent), help="book source root (default: ..)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    book_root = Path(args.book_root).resolve()
    if not (book_root / "main.tex").exists():
        print(f"warning: {book_root} does not look like the book root (no main.tex)", file=sys.stderr)
    return export(book_root, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
