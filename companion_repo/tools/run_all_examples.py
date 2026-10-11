#!/usr/bin/env python3
"""Run every ``examples/*/*.py`` script in a subprocess (cwd = its own folder) and report failures.

usage:  python tools/run_all_examples.py [--filter SUBSTRING] [--quiet] [--list]

The repository root is prepended to ``PYTHONPATH`` so that ``snfusion`` is
importable without installation; matplotlib is forced to the ``Agg`` backend.
Exit status is the number of failed scripts.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def find_scripts(subdir: str = "examples", pattern: str = ""):
    scripts = sorted((REPO / subdir).glob("*/*.py"))
    return [s for s in scripts if not s.name.startswith("_") and pattern in str(s.relative_to(REPO))]


def run(script: Path, quiet: bool = False) -> bool:
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg")
    t0 = time.time()
    proc = subprocess.run([sys.executable, script.name], cwd=script.parent, env=env,
                          capture_output=quiet, text=True)
    ok = proc.returncode == 0
    print(f"[{'ok' if ok else 'FAILED'}] {script.relative_to(REPO)}  ({time.time() - t0:.1f} s)")
    if not ok and quiet:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default="", help="only run scripts whose path contains this substring")
    ap.add_argument("--quiet", action="store_true", help="hide script output (shown only on failure)")
    ap.add_argument("--list", action="store_true", help="list the scripts and exit")
    ap.add_argument("--subdir", default="examples", help="folder to scan (default: examples)")
    args = ap.parse_args()

    scripts = find_scripts(args.subdir, args.filter)
    if args.list:
        for s in scripts:
            print(s.relative_to(REPO))
        return 0
    if not scripts:
        print("no scripts found")
        return 0

    failures = [s for s in scripts if not run(s, args.quiet)]
    print("-" * 72)
    print(f"{len(scripts) - len(failures)}/{len(scripts)} scripts succeeded")
    for s in failures:
        print(f"  failed: {s.relative_to(REPO)}")
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
