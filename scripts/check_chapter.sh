#!/usr/bin/env bash
# Compile ONLY one chapter of the book in an isolated scratch copy (safe to run in parallel).
#   usage:  scripts/check_chapter.sh ch04     (prefix of the file in chapters/)
#           scripts/check_chapter.sh all      (full book)
# Prints LaTeX errors / undefined references and leaves the PDF in the scratch build folder.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CH="${1:-all}"
SCRATCH="${SCRATCH_BASE:-$HERE/build}"
BUILD="$SCRATCH/build_$CH"
mkdir -p "$BUILD"
rsync -a --delete --exclude companion_repo --exclude 'main.pdf' --exclude '.git' "$HERE/" "$BUILD/"
cd "$BUILD"
if [ "$CH" = "all" ]; then
  ONLY=""
else
  F=$(ls chapters/${CH}_*.tex | head -1 | sed 's/\.tex$//')
  ONLY="\\includeonly{$F}"
fi
pdflatex -interaction=nonstopmode "${ONLY}\\input{main.tex}" > pass1.log 2>&1
bibtex main > bibtex.log 2>&1
makeindex -s style/svind.ist main > makeindex.log 2>&1
pdflatex -interaction=nonstopmode "${ONLY}\\input{main.tex}" > pass2.log 2>&1
pdflatex -interaction=nonstopmode "${ONLY}\\input{main.tex}" > pass3.log 2>&1
echo "=== build dir: $BUILD   (pdf: $BUILD/main.pdf)"
echo "=== LaTeX errors (pass 3):"
grep -n -A3 '^!' main.log | head -60
echo "=== undefined citations / references (pass 3):"
grep -n "Citation .* undefined\|Reference .* undefined" main.log | head -40
echo "=== bibtex warnings/errors:"
grep -n -i "warning\|error" bibtex.log | grep -v "didn't find a database entry" | head -20
echo "=== overfull hboxes > 30pt:"
grep -n -E "Overfull \\\\hbox \([0-9]{2,}" main.log | head -10
echo "=== pages:"
grep -o "Output written on main.pdf ([0-9]* pages" main.log
