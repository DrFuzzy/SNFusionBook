# Towards a Unified View of Sensor Fusion — companion repository

**Theory, Methods, and Autonomous Driving Applications**
Kyriakos M. Deliparaschos, Gabriele Oliva, Themistoklis Charalambous — Springer Nature, 2026.

Repository: [SNFusionCompanion](https://github.com/DrFuzzy/SNFusionCompanion).

This repository accompanies the monograph. It contains

* **`snfusion/`** — a small, documented and tested reference library (pure numpy/scipy)
  implementing the methods of the book: Kalman-type filters, zonotopic set-membership
  estimation, evidence theory, incomplete AHP, track-to-track fusion / covariance
  intersection, linear consensus, motion models and consistency metrics;
* **`examples/`** — the scripts that generate the worked examples and the data behind the
  figures of the book, one folder per example, each with a cell-annotated script
  (`# %%` cells) and the corresponding Jupyter notebook;
* **`exercises/`** — end-of-chapter exercises with solutions, as scripts and notebooks;
* **`tests/`** — unit tests for the library and regression tests that check, value by value,
  that the example scripts reproduce the CSV files shipped with the book sources.

The book compiles without this repository: the generated CSV tables and figures are copied
into the book's `data/chNN/` and `figures/chNN/` folders (`make export`).

## Structure

```
companion_repo/
├── snfusion/                 reference library (importable package, version 0.1.0)
│   ├── kalman.py             KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, wrap_angle
│   ├── models.py             CV / CA / coordinated-turn models, white-noise-acceleration Q
│   ├── zonotope.py           Zonotope: Minkowski sum, linear map, strip intersection, order reduction
│   ├── evidence.py           MassFunction: Dempster / Smets rules, discounting, pignistic transform
│   ├── ahp.py                incomplete-AHP LLS completion, eigenvector priorities, consistency ratio
│   ├── consensus.py          information fusion, correlated fusion, covariance intersection, consensus
│   ├── metrics.py            RMSE, NIS, NEES, chi-square bounds
│   └── scenarios.py          the radar cut-in scenario shared by the Chapter 3 examples
├── examples/
│   ├── ch03_mean_variance_gaussian/ mean_variance_gaussian.py  (+ .ipynb, output/)
│   ├── ch03_ekf_radar_cutin/        ekf_radar_cutin.py  (+ .ipynb, output/)
│   ├── ch03_ukf_radar_cutin/        ukf_radar_cutin.py
│   ├── ch03_zonotope_radar_cutin/   zono_radar_cutin.py
│   ├── ch03_evidence_occlusion/     evidence_occlusion.py
│   ├── ch03_incomplete_ahp/         incomplete_ahp.py
│   └── ch04_* ... ch07_*            examples of the later chapters
├── exercises/ch03 ... ch08          exercises with solutions (scripts + notebooks)
├── tests/                           pytest suite (+ tests/reference/ch03: reference CSVs)
├── tools/
│   ├── py2nb.py              script -> notebook converter (no jupyter needed)
│   ├── run_all_examples.py   runs every examples/*/*.py in its own folder, reports failures
│   └── export_outputs.py     copies outputs into the book sources (used by `make export`)
├── Makefile, pyproject.toml, requirements.txt, CITATION.cff, LICENSE
```

## Installation

Python 3.9+ with numpy and scipy is enough for the library; matplotlib (and pandas for some
later-chapter examples) are needed by the example scripts, pytest by the tests.

```bash
cd companion_repo  # supplied with the book source bundle
python3 -m pip install -r requirements.txt     # numpy, scipy, matplotlib, pandas, pytest
python3 -m pip install -e .                    # or: make install  (installs snfusion)
```

Installation is optional: every example script and the test suite put the repository root
on `sys.path`, so `python3 examples/ch03_ekf_radar_cutin/ekf_radar_cutin.py` works from a
plain checkout.

Quick check:

```python
import numpy as np
from snfusion import KalmanFilter, Zonotope, MassFunction
from snfusion.ahp import incomplete_lls
from snfusion.consensus import covariance_intersection
```

## Reproducing the examples

Each example lives in its own folder, is run **from that folder** and writes to its
`output/` sub-folder (kept under version control, so the data can be inspected without
running anything). The revised scripts use fixed seeds. Compare floating-point results with numerical tolerances: library, BLAS, and eigensolver versions can affect the final digits. The Chapter 7 cable simulation uses the symmetric principal covariance square root to avoid eigenvector-sign changes in its seeded noise.

```bash
python3 examples/ch03_ekf_radar_cutin/ekf_radar_cutin.py  # from companion_repo
make examples  # all examples; download comma2k19 first
```

### Chapter 3 — Classical sensor fusion

| Folder / script | Book example | Produces | Output files |
|---|---|---|---|
| `examples/ch03_mean_variance_gaussian/mean_variance_gaussian.py` | Figures *fig:mean-variance* and *fig:gaussian-1d-2d* (Sections *Measurement uncertainty* and *Gaussian measurement model*) | Readings of two sensors with sample mean and standard deviation; correlated 2-D errors, eigen-decomposition of R and the ellipses c = 1, 2, 3 with the probability they contain (39.3, 86.5, 98.9 %) | `output/mean_variance_1d.csv`, `covariance_2d_samples.csv`, `covariance_2d_ellipses.csv` → `data/ch03/`, `mean_variance_gaussian.pdf` (preview) |
| `examples/ch03_ekf_radar_cutin/ekf_radar_cutin.py` | Example *ex:ekf-radar-cutin*, "Extended Kalman filtering of an automotive-radar cut-in" (Section *Bayesian Filtering*) | Fig. *fig:ekf-radar-cutin* (a) bird's-eye view and (b) NIS consistency; the RMSE / NIS numbers quoted in the text | `output/ekf_radar_cutin.csv` → `data/ch03/`, `ekf_radar_cutin.png` (preview) |
| `examples/ch03_ukf_radar_cutin/ukf_radar_cutin.py` | Example *ex:ukf-radar-cutin*, "Does the unscented transform help here?" | EKF vs UKF lateral and position errors (nominal and stressed near-field scenario), sigma-point weights; `--montecarlo` re-runs the update-rate sweep of Table *tab:ekf-ukf-rate* (100 paired runs per rate, seeds 1000-1099; fixed continuous acceleration spectral density 0.032 m^2/s^3) | `output/ukf_radar_cutin.csv` → `data/ch03/`, `ukf_radar_cutin.png` |
| `examples/ch03_zonotope_radar_cutin/zono_radar_cutin.py` | Example *ex:zono-radar-cutin* (Section *Set-Membership and Zonotopic Estimation*); the single-strip Example *ex:zonotope-strip* is checked in `tests/test_zonotope.py` | Fig. *fig:zono-radar-cutin* (zonotopic enclosure vs EKF 3σ envelope); containment / width / generator-count numbers of the text | `output/zono_radar_cutin.csv` → `data/ch03/`, `zono_radar_cutin.png` |
| `examples/ch03_evidence_occlusion/evidence_occlusion.py` | Example "Rain-night occlusion conflict" (Section *Evidence Theory for Conflicting Sensor Information*), BBAs of eq. *evidence-example-bbas* | Conflict K = 0.44, Dempster-fused BBA, Bel(O), Pl(O), pignistic probabilities — the numbers of Fig. *fig:evidence-theory-autonomous-driving-example* | `output/evidence_occlusion.csv`, `evidence_occlusion.png` |
| `examples/ch03_incomplete_ahp/incomplete_ahp.py` | Example "Incomplete AHP for a rain-night sensor policy" (Section *Fusion based on Multi-Criteria Decision-Making*), eq. *lls-ahp* | Priority vector of eq. *incomplete-ahp-example-weights* and the inferred ratios of Fig. *fig:incomplete-ahp-autonomous-driving-example* | `output/incomplete_ahp.csv`, `incomplete_ahp_ratios.csv`, `incomplete_ahp.png` |

The three radar scripts share the scenario module `snfusion/scenarios.py` (target, radar,
seed 20260727). The zonotope script reads the EKF output of its sibling folder to draw the
3σ comparison and regenerates it if it is missing. The figures themselves are drawn by
pgfplots inside the book from the CSV files; the PNGs are previews.

### Chapters 4–7 — worked examples of the expanded chapters

Each script is self-contained (numpy/scipy only), deterministic, writes its results to `output/` and copies the
CSV files read by the book's pgfplots figures into `../data/chNN/` (see `make export`).

<!-- examples-index:start -->
| Chapter | Script (run from its folder) | What it reproduces | Notebook |
|---|---|---|---|
| 4 | `examples/ch04_hybrid_hetero_kf/hybrid_hetero_kf.py` | Hybrid classical--learning fusion: a detector with learned heteroscedastic noise feeding a Kalman tracker. | `hybrid_hetero_kf.ipynb` |
| 4 | `examples/ch04_comma2k19_learned_gnss/comma2k19_learned_gnss.py` | Worked application on public driving data (comma2k19, MIT licence): learned GNSS error model (offset + Gaussian-process kernel fitted by marginal likelihood) inside an EKF; consistency on a second car. Run `download_comma2k19.py` first (about 110 MB, cached in `data_cache/`). | `comma2k19_learned_gnss.ipynb` |
| 4 | `examples/ch04_pose_graph_1d/pose_graph_1d.py` | Pose-graph optimisation by hand: four 1-D poses, three odometry factors and one loop closure. | `pose_graph_1d.ipynb` |
| 5 | `examples/ch05_camera_radar_ci_fusion/camera_radar_ci_fusion.py` | Object-level camera--radar track fusion: naive independent fusion versus covariance intersection. | `camera_radar_ci_fusion.ipynb` |
| 5 | `examples/ch05_gnss_imu_odometry_ekf/gnss_imu_odometry_ekf.py` | Ego-localisation by GNSS--IMU--wheel-odometry fusion with a GNSS outage. | `gnss_imu_odometry_ekf.ipynb` |
| 6 | `examples/ch06_distributed_tracking/distributed_tracking.py` | Distributed tracking of a 2-D constant-velocity target by a five-node sensor network. | `distributed_tracking.ipynb` |
| 6 | `examples/ch06_resilient_consensus/resilient_consensus.py` | Resilient consensus with a misbehaving agent: plain averaging versus W-MSR. | `resilient_consensus.ipynb` |
| 7 | `examples/ch07_ais_innovation_monitoring/ais_innovation_monitoring.py` | Example A (Chapter 7): constant-velocity Kalman fusion of coastal-radar plots and AIS position | `ais_innovation_monitoring.ipynb` |
| 7 | `examples/ch07_cable_das_kf/cable_das_kf.py` | Example B (Chapter 7): pre-curved cable, quadratic strain channels, EKF with second-order prediction. | `cable_das_kf.ipynb` |
| 7 | `examples/ch07_evidence_mcdm/evidence_mcdm.py` | Chapter 7 worked examples: evidence-theoretic fusion of heterogeneous maritime cues (Dempster | `evidence_mcdm.ipynb` |
<!-- examples-index:end -->

## Notebooks

Every script is cell-annotated (`# %%` / `# %% [markdown]`) and converted to a notebook
with the dependency-free converter `tools/py2nb.py`:

```bash
python3 tools/py2nb.py examples/ch03_ekf_radar_cutin/ekf_radar_cutin.py   # one notebook
make notebooks                                                             # all of them
```

The notebooks are shipped unexecuted; open them in Jupyter and run all cells (the first cell
sets the script path and repository root on `sys.path`; open it from its own folder so its local data paths resolve).

## Tests

```bash
make test            # = python3 -m pytest -q
```

* `tests/test_kalman.py` — KF against a hand computation, Joseph vs standard update,
  EKF/UKF/KF agreement on a linear model, sigma-point weights, angle wrapping;
* `tests/test_zonotope.py` — the single-strip Example *ex:zonotope-strip* of the book,
  Frobenius-optimal gain (closed form, tightening, containment), order reduction, Minkowski sum;
* `tests/test_evidence.py` — the rain-night example numbers, Zadeh's textbook example,
  discounting, vacuous BBA;
* `tests/test_ahp.py` — LLS completion recovers a consistent matrix, the book example,
  eigenvector priorities and consistency ratio;
* `tests/test_consensus.py` — consensus converges to the average (Metropolis / step-size
  weights), information fusion, covariance intersection consistency for any cross-covariance;
* `tests/test_metrics_models.py` — chi-square bounds, RMSE/NIS/NEES, motion models, scenario;
* `tests/test_regression_ch03.py` — runs the Chapter 3 scripts and compares their CSV output
  with `tests/reference/ch03/` and, when available, with the book's `data/ch03/`
  (tolerance 1e-9; the files are byte-identical).

## Exporting to the book

The book sources keep a copy of every generated table and figure so that they compile
without this repository. After re-running the examples:

```bash
make export                      # copies examples/*/output/*.csv -> ../data/<chNN>/
                                 #        and examples/*/output/*.pdf -> ../figures/<chNN>/
make export BOOK_ROOT=/path/to/book
```

`<chNN>` is the chapter prefix of the example folder name. From the book root, `make examples`
runs the same export.

## Exercises

Every chapter with end-of-chapter problems in the book has a folder `exercises/chNN/` containing the problem
statements (`README.md`), the solution scripts for the computational problems (`solutions/*.py`, with `# %%` cells)
and the notebooks generated from them (`tools/py2nb.py`). The analytical solutions are printed at the end of the book.

<!-- exercises-index:start -->
| Chapter | Folder | Problem statements | Solution scripts (each with a notebook) |
|---|---|---|---|
| 3 — Classical sensor fusion | `exercises/ch03/` | 10 problem statements | `ch03_hand_computations.py`, `ch03_prob10_ekf_process_noise.py` |
| 4 — Learning-based and hybrid fusion | `exercises/ch04/` | see README | `prob6_miscalibrated_head.py`, `prob7_pose_graph_2d.py` |
| 5 — Autonomous driving application | `exercises/ch05/` | see README | `prob04_gnss_outage_variants.py`, `prob06_ci_sensitivity.py` |
| 6 — Multi-agent sensor fusion | `exercises/ch06/` | see README | `_coi_common.py`, `prob05_consensus_rounds.py`, `prob06_packet_loss.py`, `prob07_wmsr_path.py` |
| 7 — Applications beyond autonomous vehicles | `exercises/ch07/` | see README | `prob06_spoof_sweep.py`, `prob07_cable_sweep.py` |
| 8 — Open challenges and future directions | `exercises/ch08/` | see README | `ch08_p2_ece.py`, `ch08_p3_timestamp_offset.py` |
<!-- exercises-index:end -->

## Citation

Please cite the book (and, if useful, this repository; see `CITATION.cff`):

> K. M. Deliparaschos, G. Oliva, T. Charalambous, *Towards a Unified View of Sensor Fusion:
> Theory, Methods, and Autonomous Driving Applications*. Springer Nature, 2026.

## License

MIT License, Copyright (c) 2026 Kyriakos M. Deliparaschos, Gabriele Oliva, Themistoklis
Charalambous. See `LICENSE`.

## Review of 10 November 2026

The revised book and review package document corrected radar initiation, exclusion of the initiating observation from NIS statistics, the implemented UKF Monte Carlo sweep, packet-loss row renormalisation, and the timestamp-offset detection-power calculation. All 16 example scripts, 14 exercise scripts, and 42 tests passed in the review environment. The comma2k19 example requires the public data cache; its absence is an error, not a successful reproduction. The review package includes the downloaded-file SHA-256 manifest.

To limit BLAS oversubscription for the cable and GP examples, optionally set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`. The notebooks are generated from the scripts and shipped unexecuted; two representative notebooks were executed in the final check. The book's 48 full problem statements are exported as `exercises/chNN/problems.tex`.

The cable example is an idealised numerical benchmark. Neither its successful execution nor a low average NEES establishes field validity or safety certification. The repository URL has been supplied by the authors; an immutable release/archive identifier remains recommended.
