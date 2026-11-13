# Towards a Unified View of Sensor Fusion - revised book

*Theory, Methods, and Autonomous Driving Applications*  
K. M. Deliparaschos, G. Oliva, T. Charalambous

This is the clean revision following the technical and reference review of 10 November 2026. All earlier supplied revision markup is accepted in the baseline. The new red/blue comparison is generated against `review/baseline/`, not against an earlier author's revision round.

## Reading and editing

- `main.tex`: clean revised book; build with `latexmk -pdf main`.
- `main_red_blue.tex`: self-contained comparison source (generated), when included in the release package. Build with `latexmk -pdf main_red_blue`. Removed passages are red and additions blue. Changed bibliography entries are printed in full before/after pairs so removed identifiers remain visible.
- `output/pdf/sensor_fusion_revised.pdf`: clean reading copy.
- `output/pdf/sensor_fusion_red_blue.pdf`: review comparison, with its own pagination.
- `output/pdf/book_review_and_reassessment.pdf`: findings, improvements, remaining author actions, and all 48 exercise verdicts.
- `review/audit.html`, `review/reference_audit.csv`, `review/exercise_audit.csv`: searchable complete ledgers.
- `review/source_changes.patch`: changes to source text and companion implementations.

The book compiles without running the companion code or downloading data. The required `data/` CSVs, `figures/` assets, Springer class/style files, and all active bibliography databases are included. `bib/review_2026-11-10.bib` adds the three standards references. The current subject index is empty; a subject index is still recommended before publication.

## Rebuilding both reviewed versions

```sh
python3 scripts/build_review_versions.py
```

Requires Python, TeX Live (`latexmk`, `pdflatex`, `bibtex`, `makeindex`) and `latexdiff`. The generator preserves a common final layout and compares source units individually. Bibliography comparisons include printed identifiers. Computational figures use revised numerical data; earlier numerical results and method changes are described in the report and source patch. Existing figure colours are not revision markup.

## Companion code and verification

See `companion_repo/README.md`. All 16 example scripts, 14 computational exercise scripts, 42 tests, and the independent arithmetic checks passed in the review environment. Two representative notebooks were executed; the remaining generated notebooks are shipped unexecuted. Full revised exercise statements are exported in `companion_repo/exercises/chNN/problems.tex`.

The public comma2k19 cache is excluded from the release package. Run its supplied downloader before running that example or exercise. `review/comma2k19_manifest.json` records the 2,244 downloaded-file hashes used in this review. Set BLAS thread counts to one if the cable or GP simulations oversubscribe the machine.

## Author items before release

The report flags the historical Gabus-Fontela DEMATEL record and the unavailable unpublished cable manuscript for author evidence. Two YBS books are catalogue-confirmed. The cable benchmark remains an idealised proposed numerical model requiring independent physical assessment. Assign a permanent public companion-code release/archive URL before publication. No public repository or publication acceptance has been invented.

The original supplied sources/PDF are preserved in `review/baseline/`. One-time editing scripts under `review/` document the audit process and should not be replayed on an already revised tree.
