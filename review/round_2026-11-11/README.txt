CO-AUTHOR REVISION — 11 NOVEMBER 2026

Latest deliverables:
  output/pdf/sensor_fusion_revised_2026-11-11.pdf
  output/pdf/sensor_fusion_red_blue_cumulative_2026-11-11.pdf

Scope:
  1. Clarify generated numerical results, compilation, external datasets, and the repository.
  2. Name "Solutions to the Problems" explicitly and explain non-unique design answers.
  3. Reduce Figure 2.1 from 0.85 to 0.68 column width (20% in each dimension).
  4. Identify the UKF comparison script in a full sentence; replace long in-text code paths
     with short linked identifiers. A new Companion Code and Data guide lists 30 scripts,
     six chapter problem collections, the UKF command/output, and data access instructions.
  5. Remove "Senior Member, IEEE" from all three biographies in the clean version;
     retain each removal in red in the cumulative comparison.

Repository supplied by the authors:
  https://github.com/DrFuzzy/SNFusionCompanion
Anonymous retrieval could not confirm the remote file tree (HTTP 404). The book uses the
supplied repository entry URL and guide paths verified against the local companion files;
no remote default branch or file-specific GitHub URLs have been invented. Paths in the guide
are relative to the companion-code folder, rather than assuming an unverified repository layout.

Preservation:
  The previous marked source and both previous PDFs are preserved in before/.
  The cumulative comparison is based directly on that marked source, with additional
  recorded insertions. Reversing only these insertions reconstructs the previous source
  byte-for-byte. Existing review changes are not accepted or flattened. Earlier additions
  replaced in this round remain visible in red, followed by the new wording in blue.
  The older undated PDFs in output/pdf/ remain the 10 November delivery.

Build:
  python3 scripts/build_coauthor_revision.py
  python3 scripts/build_review_versions.py  (dispatches to the cumulative build)

Verification:
  validation.json records source preservation, local code links, PDF checks, and hashes.
  No filtering algorithms, simulation settings, numerical data, or bibliographic records
  were changed in this editorial round. The previous technical audit remains separate.
