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

## Round 2, part B — errata (16 October 2026)

An independent review of Chapters 4–9 found objective errors in material added on 11 October. Each was checked
against the shipped data files or recomputed before correction; every visible change is marked in red/blue.

| Chapter | Correction |
|---|---|
| 3 | "this survey" → "this book"; notation row for the factor information matrix $\mathbf{\Omega}_i$ |
| 4 | pinhole example: the point 1 m below the axis projects to (740, 410), not (740, 310) |
| 4 | IMU integration equation (4.18): velocity and gravity were counted twice in $\mathbf{p}_j$ |
| 4 | wrong cross-reference to the occupancy-mapping subsection; logistic σ(·) vs standard deviations σ_i clarified |
| 4 | "dominant industrial pattern" qualified, with a pointer to the end-to-end challenge of Chapter 8 |
| 5 | gyro bias of the tunnel example reconciled with the sensor table (turn-on offset vs bias instability); $\mathbf{G}_k$ disambiguated; 4D imaging radar measures elevation; Seeing Through Fog row (12k real-world samples in mixed weather plus 1.5k fog-chamber samples) |
| 4/5 | duplicate label `eq:ece` (Chapter 5 copy renamed `eq:ece-metric`) |
| 5/6 | duplicate label `eq:nees` (Chapter 6 copy renamed `eq:nees-mas`) |
| 6 | caption of Fig. 6.1 rewritten to match the plotted data; "too small by two thirds" → "about 60 % of the actual covariance"; consensus needs positive self-weights; direction of the KL divergence stated; CI does not protect against a sender that declares a small covariance; late-fusion bandwidth "some tens of kbit/s"; CPM content made consistent with Chapter 8 |
| 7 | AIS reporting interval 14–23 kn is 6 s (ITU-R M.1371), not 3⅓ s; "long before" claim for the window test softened; damping ratio consistent with the modal model |
| 7 | cable example and Fig. 7.2: the per-step NIS does exceed its threshold at t = 60.5 s (102.6 > 32); text and caption rewritten: the window test gives a sustained and specific alarm, not an earlier one |
| 8 | weather and glare are not events "far below 10⁻⁶ per hour"; $h_i$ → $\mathbf{g}_i$; standards dated (ISO 26262:2018, ISO 21448:2022, ISO/PAS 8800:2024) and no longer "emerging"; CPM specified in ETSI TS 103 324, SAE J3224 added; CPM content consistent with Chapter 6 |
| 9 | "this survey" → "this book" (3×); one sentence acknowledging the end-to-end counter-thesis |

Full build after the errata: 0 errors, 0 undefined references, 0 multiply-defined labels, 217 pages.

**Left for the authors** (judgement calls, not errors):
- the cable strain model of Chapter 7 (axial strain taken proportional to ∂w/∂x for transverse motion; to first order transverse motion gives no axial strain) and the magnitude of the impulse and drag force: please check against the model of the cable paper;
- CenterPoint 60.3/67.3 in Table 4.1 is the leaderboard entry with test-time augmentation; the CVPR paper reports 58.0/65.5 (to verify);
- Lyft Level 5 and ApolloScape are named in Chapter 5 but neither cited nor in the datasets table;
- recency of the autonomous-driving literature (no driving paper after 2023 in Chapters 4–5), safety content in Chapter 8 (ODD, degraded modes, minimal-risk manoeuvre), learned V2X cooperative perception, 3D multi-object tracking, online mapping, 4D radar;
- repeated "embedded/FPGA perspective" paragraphs in Chapter 4 and the MUEAVI/YOLOv3 detail in Chapter 5 (co-author material);
- leftover editorial comments in the sources (`% [July 2026 pass]`, `% Addressed ...`).

---

# Round 3 — November 2026 (comments of G. Oliva, 8 November 2026)

**Before this round** all the markup of the October rounds was accepted (blue kept, red removed, the leftover
`\textcolor{olive}` text of Chapter 3 turned black). The marked-up sources of the October rounds are archived in
`SN_book_20261108_round2_markup.zip`. The "Note on the Revision Markup" front-matter chapter was removed
(`frontmatter/revision_note.tex` deleted, include removed from `main.tex`). In the current sources blue/red mark only
the changes of this round. New bibliography entries of this round are in `bib/additions_2026-11_*.bib`
(included by `main.tex`). Layout-only changes are not coloured and are listed below.

## Chapters 1–3, 6, 8, 9 and front matter (main session)

| Where | Change |
|---|---|
| Book abstract | acronyms expanded (LiDAR, GNSS, IMU); set-membership estimation and MCDM added to the list of methods; "this survey" → "this book"; keywords updated (added cooperative perception, distributed estimation, hybrid classical–learning fusion, multi-criteria decision-making, object tracking, set-membership estimation; removed multimodal perception and object detection) |
| Ch. 1 abstract | rewritten for readability: contributions grouped as methods / systems / tutorial apparatus |
| Ch. 1, "Individual sensors…" | paragraph replaced: two sensors (camera, radar), each with a concrete failure case; the other sensors point to Table 5.x; LiDAR, GNSS, IMU defined at first use |
| Ch. 1, Sect. 1.2 | chapter in parentheses after each contribution; MCDM and AHP defined; one-line gloss of set-membership estimation for readers from ML; footnote with the companion-repository URL at its first mention (macro `\repourl` in `main.tex`, **placeholder to be filled**) |
| Ch. 1, Sect. 1.3 | "generic" → "general-purpose"; "domain-specific viewpoints" → "discipline-specific" + one sentence (the book is centred on one domain but draws on all communities); overclaiming sentence ("not only … but also reinterprets … through a unified, modern lens") rewritten; "short book" → "book"; new paragraph stating in which three senses the view is "unified" and why the title says "towards" |
| Ch. 1, Sect. 1.3.1 | wrong claim corrected: the estimation notation is *not* the ML notation (ML writes z for the latent state and x for the observation); JDL, SLAM, V2X defined |
| Ch. 1, Sect. 1.4 | aligned with Sect. 1.2 (Chapter 3 includes track-level fusion; Chapter 5 includes sensor suite and alignment) |
| Ch. 2 | taxonomy paragraph checked: "coincide with" → "are related to"; early/intermediate/late fusion: the terms are used differently in multimodal ML (Baltrušaitis) and in driving (Feng et al. 2021); JDL and AR/VR defined; new paragraph on the OODA loop and the Omnibus model (Bedworth & O'Brien 2000) read against Fig. 2.1 (no new figure). References of the chapter checked (all correct). No problems added to Chapter 2 (conceptual chapter) |
| Ch. 3 | "on which sensor fusion rests" → "of sensor fusion"; "particularly transparent" replaced by what the stacked form actually gives; LS defined; two new figures (`fig:mean-variance`: repeated readings with sample mean/std and correlated 2-D errors with eigen-axes; `fig:gaussian-1d-2d`: 1-D density with 68.3/95.4 % and 2-D ellipses with 39.3/86.5/98.9 %) with text (Mahalanobis distance, chi-square, link to NIS and gating); differentiability assumption before the Jacobian and "first-order Taylor expansion truncated after the linear term" for the Gauss–Newton linearisation; Table 3.2: row spacing, MAP/EKF/UKF expanded, "precision matrix", new row observability ↔ identifiability; sentence on the z/x vs x/y notation clash; rhetorical sentences rewritten ("covariance matrices are not bookkeeping", "a confidence it has not earned", "the remedy is not fewer sensors but honest models", "not merely … but", "Containment is exact; tightness is not", "The price is conservatism, and it is not small", "the conflict mass is not merely a nuisance term"); EKF, UKF, DSmT, AHP, RMSE defined |
| Ch. 4 | BEV, mAP, NDS, GPU, FPGA, IoU defined; two rhetorical phrases toned down (no structural change: see the proposal in the reply of 8 November) |
| Ch. 6 | AIS, V2V, V2I defined; rhetorical sentences rewritten ("not merely more data", "Three observations carry the lesson", "useless", "catastrophic", "honest") |
| Ch. 8 | DNN, SoC defined; "AI-based" spelled out |
| Ch. 9 | rhetorical sentences rewritten ("If a single lesson…", "not competitors but layers", "uncertainty bookkeeping", "decisive insight", "not a compromise", "represent uncertainty honestly", "a map and an invitation") |
| Companion repo | new example `examples/ch03_mean_variance_gaussian/` (script + notebook + outputs), CSVs copied to `data/ch03/`; README updated |

**Layout only (not coloured).** Seven displayed equations of Chapter 3 that were broken over two lines although
they fit on one were joined (Gaussian likelihood (3.x, label `eq:Gaussianlikelihood`), regularised LS cost, MAP
estimate (former 3.44), sequential LS cost, the normal equation of sequential LS, its solution, Gauss–Newton
increment (near former 3.64)). The remaining two-line displays of Chapters 1–4, 6, 8, 9 are either genuinely wider than
the text width or pairs of distinct equations. Row spacing (`\arraystretch` 1.25–1.3) added to the tables with
multi-line cells: 1.2 (terminology), 3.1 (notation), 3.2 (dictionary), 4.x (deep-fusion methods, fusion-level
trade-off), 8.1, 9.1.

## Chapter 5 (editorial and technical pass)

All numbers of Examples 5.x, Table 5.4 and Solutions 5.4/5.6 were re-run with the companion scripts (CSV files
byte-identical). Corrections (markup in the text): the gyro-bias 3σ bound falls on the initial straight, not "during the
first curve" (text, caption (c), Solution 5.4); "45 s" → 50 s before the tunnel; 0.60 m 2-D RMS compared with the
matching 2.1 m single-fix RMS; the 0.5 m lane-level budget is exceeded about 6 s after the tunnel entrance (not 10 s);
0.52 m compared with the fused 1σ (0.21 m) instead of the sensors' 1.55/1.58 m; gating uses the innovation σ
(eq. 3.91; also Problem 5.2); J is the inverse of the radar Jacobian's position block; "consistent iff NEES ≤ 2" removed
(trace test is necessary, not sufficient); NEES cannot be monitored online without ground truth; small-rotation formula,
transform-tree root, eq. 5.5, rollback innovation; dataset table (Argoverse 2 lidar split has no cameras, Argoverse 1
added, Waymo cameras not full surround), nuScenes' reason for centre distance instead of IoU; FPGA/HIL claims aligned
with the cited papers; Autoware (ROS 1) claim corrected. Glosses for ML readers (lever arm, SE(3)/SO(3) vs covariance R,
retrodiction, innovation, consistency, observability, dead reckoning, information matrix). Acronyms defined at first
use. Rhetorical phrases removed. Six references added and verified (ApolloScape TPAMI 2020, View-of-Delft RA-L 2022,
K-Radar NeurIPS 2022 and MAN TruckScenes NeurIPS 2024 also as table rows, Fan et al. 4D-radar survey T-IV 2024,
MultiCorrupt IV 2024); Lyft Level 5 dropped (no peer-reviewed reference, official page gone). Corrected entries
`ester1996_dbscan` and `deliparaschos2016_jee_fpga` replace the faulty `ester_dbscan_1996`, `deliparaschos_fpga_2016`,
`deliparaschos_fpgabased_2016` (no longer cited; can be deleted from references.bib). Row spacing added to the four
tables of the chapter (layout only).

**For the authors:** dropping Lyft and the two new dataset rows; merging the duplicate FPGA keys; the MUEAVI material
(co-author) is not really about fusion, its Fig. 5.5 caption repeats the paragraph, "YOLOv3" could read "YOLO-family
detectors (YOLOv3 in MUEAVI)"; Fig. 5.3a has unequal axis scales; Table 5.3 mixes resolution and accuracy.

## Chapter 7 (submarine-cable section aligned with the revised EJC paper, kept short)

The strain model is corrected: a lateral displacement stretches a straight cable only to second order, so for a
route bent in plan the channel reading is quadratic in the state (`eq:ch07-strain-channel`), monitored by an EKF with
second-order measurement prediction (`eq:ch07-second-order-prediction`); the old linear-output model, example and
figure are in deleted blocks (labels `-old`, old CSVs kept in `data/ch07/superseded/`, to be deleted when the markup
is accepted). New Example `ex:ch07-cable-das` "Anchor drag on a pre-curved 1-km span" on the paper's benchmark (numbers
printed by the rewritten `companion_repo/examples/ch07_cable_das_kf/cable_das_kf.py`; tests 38/38), new Fig. 7.3,
a short limitations paragraph (idealised benchmark, synthetic data, local axial closure not self-consistent on the
seabed, field validation). Problems 7.5 and 7.7 and their solutions updated. Clean section length ≈ +8 %.
Nothing about the reviews appears in the book. New bib: `oliva_quadratic_ejc_2026` (@unpublished, "under review at
European Journal of Control"), Jazwinski 1970, Rega 2004.

**references.bib corrections (objective):** `fioravanti_cable_med_2026` is published (Proc. 34th MED, pp. 811–816,
doi 10.1109/MED70602.2026.11598378, from the revised paper's bibliography); `baudouin_cable_pde_2018` year 2019;
`modares2026reachability` had given and family names swapped.

**For the authors:** update the EJC entry if accepted before 1 December; κ_j = 0.78 is called "transduction" in the
paper's code but is not explained in the paper or in the book (strain-optic factor of silica fibre?); the example
discretises time exactly (paper: forward Euler) and keeps the book's windowed NIS-sum test and residual-profile
localiser (paper: window mean and likelihood-ratio search).

## Front matter
List of acronyms added (`frontmatter/acronym.tex`, included after the table of contents).

## Verification (8 November 2026)
Full build with markup: 0 LaTeX errors, 0 undefined citations or references, 227 pages (`main.pdf` updated).
Clean build (`\revisionmarkupfalse`): 0 errors, 0 undefined, 227 pages.

## Chapter 4 — worked application on public driving data (added 8–9 November 2026, after the authors' go-ahead)

New subsection `subsec:c2k-application` "Worked Application: Learning a GNSS Error Model from Public Driving Data"
in the section *Hybrid Classical–Learning Architectures*, after the synthetic example (all in blue):
- data: comma2k19 (Schafer et al., arXiv:1812.05752, MIT licence; u-blox M8 fixes and raw C/N0, phone gyro, CAN speed,
  reference poses); 4 Civic drives for training (59 min), 3 Civic drives for testing (35 min), 4 RAV4 drives (47 min);
- step 1 alignment (fix epoch vs arrival: median latency 159 ms); step 2 baseline EKF (E, N, ψ, gyro bias, CAN speed
  scale) with white 2.5 m noise: RMSE 3.25 m, reported 0.63 m, mean NEES 134;
- step 3 learning: along-track offset 1.69 m; errors strongly correlated in time; Gaussian-process kernel
  σ_b² + σ_g(u)σ_g(u') exp(−|t−t'|/T) learned by maximum marginal likelihood (σ_b 2.16 m, σ_g 1.37 m, T 111 s; the
  feature-based amplitude overfits the training drives);
- step 4 hybrid EKF with the state-space form of the kernel: RMSE 2.35 m, reported 2.50 m, mean NEES 1.92, 84.9 % in
  the 86.5 % ellipse; the offset gives the accuracy, the correlation model the consistency (learning the variance
  alone: NEES 136); the NIS is far below 2 for every filter and cannot reveal the problem;
- step 5 second car (RAV4): the Civic model is conservative (NEES 0.85), the feature-based one less so (1.27);
- Table `tab:c2k-results`, Figures `fig:c2k-learning` (autocovariance and kernel; coverage curves) and
  `fig:c2k-route`; Problems 4.9 (opposite shift: NEES 5.78, coverage 51 %) and 4.10 (600 fixes in a minute are worth
  1.05 independent fixes) with solutions.
Also: Ch. 4 abstract, pointers from Sects. 4.1.3–4.1.5, Ch. 1 contributions bullet, comma2k19 row in Table 5.x
(datasets), CAN and GPS in the list of acronyms. Companion repository: `examples/ch04_comma2k19_learned_gnss/`
(`download_comma2k19.py`, stdlib only, fetches about 110 MB with HTTP Range requests into `data_cache/`, which is
git-ignored; `comma2k19_learned_gnss.py` + notebook; outputs copied to `data/ch04/c2k_*.csv`), exercise script
`exercises/ch04/solutions/prob9_10_comma2k19.py`; `tools/run_all_examples.py` skips `download_*.py`.

Full build after this addition: 0 errors, 0 undefined references, 231 pages; clean build also 0/0, 231 pages.

## Chapter 6 — refocus (9 November 2026, after the authors' go-ahead)

The chapter is now organised around one question (how can agents that only talk to their neighbours fuse estimates
of the same quantity without double counting and without being misled by a faulty agent?) and one running example,
the five-node network (ring 1–2–3–4–5–1 with chord 1–3) already used by the distributed-tracking example.
- Abstract rewritten; old introduction (two paragraphs) replaced by: the question, the running network (new
  Fig. `fig:mas-network`), four learning objectives mapped to sections, and an explicit scope statement (multi-robot
  SLAM, distributed particle filters, learned cooperative perception, coordination/control not treated).
- The four-vehicle path example is replaced by `ex:mas-network-consensus` (same initial values plus 13 m for node 5;
  one consensus step computed by hand); the Laplacian numbers now refer to the running network (9 rounds) and to the
  path obtained by removing two edges (38 rounds).
- Distributed-tracking example, table and figure unchanged (now introduced as the running network).
- Resilient consensus: new `ex:mas-resilient-net` and `fig:mas-resilient-net` on the running network: it is only
  1-robust, adding links 2–4, 2–5, 3–5 makes it 3-robust (no smaller addition does); plain consensus → 30 m,
  W-MSR → 13.33 m. The old seven-agent example and figure are in deleted blocks (old CSV kept in
  `data/ch06/superseded/`, to be deleted when the markup is accepted).
- Sections "Communication Constraints", "Cooperative Perception in Connected Vehicles", and "Validation" (≈1,170 words)
  are in deleted blocks, replaced by one section "From the Core to Real Networks" (≈500 words). The chapter had no
  sections in the outline shared with the editor (they were introduced in October), so the outline is unchanged.
- Problems 6.1 and 6.7 and their solutions rewritten for the running network; companion scripts updated
  (`examples/ch06_resilient_consensus/resilient_consensus.py`, `exercises/ch06/solutions/prob07_wmsr_path.py`,
  exercises README); tests 38/38.
- Clean length ≈5,200 words (from ≈5,750): the core sections are unchanged, the survey parts shrink, the introduction
  grows by the objectives and the scope statement.

Full build with markup: 0 errors, 0 undefined, 237 pages; clean build: 0/0, 229 pages.
