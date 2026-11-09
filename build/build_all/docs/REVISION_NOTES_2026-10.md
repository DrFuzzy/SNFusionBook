# Revision notes — October 2026 round

Working copy prepared on 11 October 2026 for the three authors. Book accepted by Springer Nature; delivery
1 December 2026. The outline shared with the editor (chapter and section titles, order) is unchanged.

## What was done

1. **Self-contained source tree** `SN_book_complete/` (see `README.md`): clean `main.tex`, chapters in
   `chapters/`, front/back matter, `style/`, `bib/`, `figures/`, `data/`, Springer docs, build scripts.
   The book compiles without the companion repository.
2. **Previous revision rounds flattened.** All green/blue/olive/cyan markup of the earlier rounds was accepted
   (added text kept, struck-through text removed). The baseline is therefore black; this round is the only
   coloured one: **blue = added**, **red = removed**. `\revisionmarkupfalse` in `main.tex` produces the clean copy.
3. **Chapters 1–3** (written by the authors, considered final): additions only — chapter abstracts
   (Springer requires one per chapter), one "tutorial apparatus" contribution bullet in Chapter 1, two sentences in
   the organisation paragraph, a closing paragraph in Chapter 2, and ten end-of-chapter problems in Chapter 3
   with full solutions at the end of the book.
4. **Chapters 4–9** (considered too sketchy): substantially expanded inside the existing section structure,
   with equations, worked numerical examples whose numbers are produced by the companion scripts, pgfplots figures
   reading `data/chNN/*.csv`, tables, design guidelines, end-of-chapter problems with solutions, and abstracts.
5. **Companion repository** `companion_repo/` (git-initialised, ready to push to GitHub): `snfusion` Python
   library (Kalman/EKF/UKF, zonotopes, evidence theory, incomplete AHP, consensus and covariance intersection,
   metrics), one folder per worked example (script with cell markers + Jupyter notebook + outputs), exercises with
   solutions per chapter, tests, Makefile with `export` target that copies outputs into `../data` and `../figures`.

## Chapter-by-chapter summary

### Chapter 1 — Introduction
Abstract added; contribution bullet on worked examples, problems and companion repository; two sentences in
"Organization of the book". Nothing removed.

### Chapter 2 — Principles and Components
Abstract added; one closing paragraph linking the component view to the problems of later chapters. Nothing removed.

### Chapter 3 — Classical Sensor Fusion Techniques
Abstract added (names every method family). Ten problems (`prob:ch03:1`–`10`): LS vs WLS, MAP/regularised LS,
sequential = batch LS, one Kalman step by hand, EKF range–bearing Jacobian and azimuth wrapping, zonotope prediction
and strip intersection, Dempster vs Smets, covariance-intersection consistency and trace-optimal weight, incomplete AHP
by logarithmic least squares, and a computational sweep of the process-noise level in the radar cut-in EKF.
Solutions in `backmatter/solutions_ch03.tex`; scripts in `companion_repo/exercises/ch03/`. The notation table
(`tab:notation`) received new rows for symbols introduced by the later chapters. Nothing removed.

### Chapter 4 — Learning-Based and Hybrid Sensor Fusion Techniques (≈5,400 → ≈10,300 words)
- *Machine Learning for Sensor Fusion*: weighted ensembles and stacking, mixture-of-experts gating, learned
  reliability (logistic regression) feeding weighted least squares (worked example), heteroscedastic noise models and
  the Gaussian negative log-likelihood, Gaussian-process residual models, calibration (ECE, temperature scaling, NIS).
- *Deep Multimodal Fusion*: taxonomy of fusion operators (concatenation, weighted sum, gating, cross-attention) with a
  numeric instance, early/mid/late trade-off table, modality dropout, learned uncertainty heads, deep ensembles.
- *BEV and Occupancy*: pinhole projection and Lift-Splat lifting with numbers, inverse sensor model and a log-odds
  example, mIoU.
- *Localisation, Odometry and SLAM*: Gauss–Newton on the factor graph, IMU preintegration, a 1-D pose-graph example
  solved by hand (with and without loop closure), MSCKF sketch.
- *Reliability-Aware, Fuzzy and Explainable Fusion*: Mamdani vs Sugeno, centroid defuzzification, ANFIS, learned
  reliability as evidence discounting. One leftover reviewer-response sentence removed (red).
- *Hybrid Architectures*: runnable example "learned heteroscedastic covariance feeding a Kalman tracker" (fixed R vs
  learned R vs oracle; RMSE and NIS consistency; figure `fig:hybrid-hetero-kf`), evidence conflict monitor numbers,
  design guidelines. Eight problems with solutions.

### Chapter 5 — Autonomous Driving as a Representative Application (≈2,700 → ≈8,100 words of prose)
- *System Context and Sensor Suite*: table of indicative sensor characteristics (`tab:av-sensor-characteristics`),
  angular-vs-integrating error patterns, sensor placement / field-of-view coverage / redundancy (lever arm).
- *Fusion Pipeline and Architectural Levels*: spatial alignment (SE(3) extrinsics, transform composition, LiDAR-to-image
  projection), temporal alignment (pose interpolation, ego-motion compensation of rotating LiDAR sweeps with numbers,
  delayed and out-of-sequence measurements), quantitative effect of calibration and synchronisation errors
  (0.5° yaw at 60 m → 0.52 m; 20 ms at 25 m/s → 0.5–1.0 m).
- *Core Fusion Tasks*: runnable example `ex:gnss-imu-odom` (GNSS–IMU–wheel-odometry EKF with gyro-bias and wheel-scale
  states, 25-s GNSS outage, 3σ consistency, comparison with a filter without bias states; figure `fig:gnss-imu-odom`),
  runnable example `ex:camera-radar-ci` (object-level camera–radar fusion, naive vs covariance intersection, NEES under a
  shared heading error; figure and table), design guidelines.
- *Datasets, Metrics, and Evaluation*: datasets table `tab:av-datasets` (12 datasets), metric definitions
  (AP, IoU, NDS, MOTA/AMOTA, ATE/RPE, NEES, ECE), scenario-based and robustness evaluation.
- *Implementation and Validation*: latency and throughput budget, middleware / time synchronisation / transform trees,
  determinism and hardware-in-the-loop; MUEAVI material unchanged; links to the other chapters.
- Eight problems with solutions. Two one-word corrections in red ("section" → "chapter", "ROS2" → "ROS 2").

### Chapter 6 — Sensor Fusion in Multi-Agent Systems (≈1,050 → ≈6,400 words)
The chapter had no sections; eight were introduced around the existing paragraphs (kept verbatim and in order):
architectures and information-exchange levels (with the existing consensus primitive and example); graph-theoretic
preliminaries (Laplacian form, average preservation, convergence rate via the algebraic connectivity, Metropolis
weights, directed and time-varying graphs); distributed Kalman filtering (information form, Kalman-consensus filter,
consensus on information / Kullback–Leibler averaging, double counting and NEES, runnable example
`ex:mas-distributed-tracking` with table and figure comparing centralised, isolated, naive, consensus-on-information,
Kalman-consensus and covariance-intersection fusion); fusion under unknown correlations (multi-source and split
covariance intersection, inverse CI); communication constraints (intermittent observations, out-of-sequence
measurements, event-triggered transmission, quantisation); resilience, trust and security (fault models, W-MSR trimmed
consensus with runnable example `ex:mas-resilient-consensus`, chi-square consistency checks, trust scores);
cooperative perception in connected vehicles (V2X fusion levels, OPV2V / V2X-Sim / DAIR-V2X); validation protocol.
Seven problems with solutions. Nothing removed.

### Chapter 7 — Applications Beyond Autonomous Vehicles (≈1,650 → ≈5,800 words)
- *Maritime*: sources and failure modes (AIS message types and rates), track-level AIS–radar fusion (CV/CT models,
  gating), innovation-based monitoring (windowed chi-square and Ljung–Box whiteness tests, non-central chi-square
  argument), runnable example `ex:ch07-ais-innovation` with coloured AIS noise and a slow spoofing ramp
  (figure `fig:ch07-ais-innovation`), behavioural features and route models, evidence-theoretic fusion example
  (conflict masses, pignistic probabilities), MCDM alert prioritisation example with AHP weights and rank-reversal
  sensitivity.
- *Submarine cable monitoring*: DAS as a measurement model (gauge length, coupling), damped tensioned-string model in
  modal state-space form, Kalman filtering with distributed measurements and residual-profile localisation, runnable
  example `ex:ch07-cable-das` (anchor strike detection and localisation; figure `fig:ch07-cable-das`), evidence and
  decision layers. Closing paragraph on the common pattern with driving. Seven problems with solutions. Nothing removed.

### Chapter 8 — Open Challenges and Future Directions (≈1,100 → ≈4,400 words of prose)
Abstract; introductory paragraph and table `tab:challenges-tools` (challenge × tools in the book × maturity);
for each of the eight sections two paragraphs (precise statement, what the book's tools already provide, what is
missing) and three open research questions; consolidated research agenda; six problems (two computational: ECE /
temperature scaling, timestamp offset vs NIS) with solutions. Nothing removed.

### Chapter 9 — Concluding Remarks (≈430 → ≈1,500 words)
Abstract; table `tab:unified-view` (fusion task × classical tool × learned component × hybrid interface);
key take-aways per part of the book; "how to use this book" (reading paths, problems, companion repository).

## Items for the authors before 1 December
- Write the Preface (blue placeholder in `frontmatter/preface.tex`).
- Chapter abstracts are printed (`\abstract{}`); switch to `\abstract*{}` if Springer should receive them online only.
- Decide whether to keep Chapter 8 at its current length (slightly above the target).
- Check the bibliographic details of the entries added this round (`bib/additions_chNN.bib`, merged into
  `bib/references.bib`); DOIs were included where known, a few page numbers were entered from memory.
- Add the related-surveys rows (2024–2025) to Table `tab:related-surveys-comparison` (open item from the survey round).
- The two 2026 references by Oliva in Chapter 7 remain cited; update them if they have been published meanwhile.
- Before production: set `\revisionmarkupfalse`, remove `frontmatter/revision_note.tex` from `main.tex`, and consider
  exporting TikZ/pgfplots figures to PDF files (Springer production note in `docs/`).

## Verification (11 October 2026)

- Full build (`scripts/check_chapter.sh all`, pdflatex + bibtex + makeindex + 2×pdflatex): **0 LaTeX errors, 0 undefined
  citations or references**, 213 pages with the revision markup. `bib/references.bib` now holds 251 entries
  (155 original + 96 added, merged without key collisions).
- Companion repository: `make test` → 38 tests passed; `tools/run_all_examples.py` → 14/14 example scripts run;
  the Chapter 3 scripts regenerate `data/ch03/*.csv` byte-identically; initial git commit created.
- Manuscript size: ≈33,000 → ≈74,000 words (LaTeX source word count, chapters only).

---

# Round 2 — 16 October 2026 (feedback of R. Setola)

**Before this round** all the blue/red markup of the 11 October round was accepted (blue text kept, red text
removed). The marked-up sources of round 1 are archived in `SN_book_20261016_round1_markup.zip`. In the current
sources, blue marks only the additions of this round.

**Motivation.** Feedback from R. Setola (11 October): in AI the term *multimodal* names the fusion of heterogeneous
data and should be visible; the book should also be readable by computer scientists, whereas the mathematics is
written in the language of control and estimation. Agreed policy: additive changes only, no change to the outline,
no rewriting of Chapter 3.

| Where | Addition (blue) |
|---|---|
| Ch. 1 abstract | one sentence announcing the terminological note |
| Ch. 1, Motivation | one sentence: in ML/AI the same problem is called multimodal fusion or multimodal learning |
| Ch. 1, new Subsect. 1.3.1 "A Note on Terminology" | Table 1.2 aligning sensor fusion, data/information fusion (JDL), robotics, multimodal learning, data fusion / truth discovery in databases; definition of *modality*; two remarks (multimodal densities vs modalities; fusion levels vs early/intermediate/late fusion) |
| Ch. 2, after the paragraph on representation and fusion levels | one paragraph on the classical taxonomies: Dasarathy (input/output abstraction), Durrant-Whyte (complementary, competitive, cooperative), the JDL levels 0–4, and early/intermediate/late fusion |
| Ch. 3, after the notation table | one paragraph and Table 3.2, a dictionary between estimation and the vocabulary of machine learning and computer science (latent variable, HMM inference, ridge regression, Fisher information, sequential Monte Carlo, abstract domains, HodgeRank/Bradley–Terry, gossip, Byzantine-robust aggregation, ...). Invisible `\label`s were added to the seven sections of Chapter 3 so that the table can point to them; no text of Chapter 3 was changed. |
| Ch. 5, feature-level fusion sentence | "that is, the multimodal fusion networks of Chapter 4" |
| Book abstract keywords | "data fusion", "multimodal fusion" |
| Bibliography | 14 entries: steinberg1999_jdl, dasarathy1997_fusion, durrantwhyte1988_sensor, bleiholder2009_datafusion, li2016_truthdiscovery, koller2009_pgm, kschischang2001_factor, ghorbal2009_zonotope, gehr2018_ai2, jiang2011_hodgerank, bradley1952_paired, boyd2006_gossip, yin2018_byzantine, sensoy2018_edl |

Also fixed: `scripts/check_chapter.sh` now excludes the `build/` folder (it was copying itself) and the archives.

**Verification.** Full build: 0 LaTeX errors, 0 undefined citations or references, 217 pages.

**Deliberately not done** (to discuss): pseudocode and complexity table for Chapter 3; learned filters
(KalmanNet, deep state-space models) in the hybrid section of Chapter 4; digital-twin material and observer
theory (Luenberger, UIO), which stay in the course notes.
