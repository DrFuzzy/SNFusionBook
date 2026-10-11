"""Regression tests: the ported Chapter 3 scripts must reproduce the CSV files shipped with the book.

Each example script is executed in a subprocess (cwd = its folder, as a reader
would run it) and its ``output/*.csv`` is compared column by column with

* ``tests/reference/ch03/<name>.csv`` (a copy kept in the repository), and
* ``<book root>/data/ch03/<name>.csv`` when the repository sits inside the book sources.

The tolerance is 1e-9 on every value; with the fixed seeds the files are in
fact byte-identical.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
REFERENCE_DIRS = [REPO / "tests" / "reference" / "ch03", REPO.parent / "data" / "ch03"]

CASES = [  # (folder, script, csv name) -- the EKF must run before the zonotope example
    ("ch03_ekf_radar_cutin", "ekf_radar_cutin.py", "ekf_radar_cutin.csv"),
    ("ch03_ukf_radar_cutin", "ukf_radar_cutin.py", "ukf_radar_cutin.csv"),
    ("ch03_zonotope_radar_cutin", "zono_radar_cutin.py", "zono_radar_cutin.csv"),
]


def run_script(folder: str, script: str) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg")
    subprocess.run([sys.executable, script], cwd=EXAMPLES / folder, env=env, check=True,
                   stdout=subprocess.DEVNULL, timeout=600)


def read_csv(path: Path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    return header, data


def max_abs_difference(a: Path, b: Path) -> float:
    ha, da = read_csv(a)
    hb, db = read_csv(b)
    assert ha == hb, f"column mismatch: {ha} vs {hb}"
    assert da.shape == db.shape, f"shape mismatch: {da.shape} vs {db.shape}"
    return float(np.abs(da - db).max())


@pytest.mark.parametrize("folder,script,name", CASES, ids=[c[2] for c in CASES])
def test_ch03_csv_regression(folder, script, name):
    run_script(folder, script)
    produced = EXAMPLES / folder / "output" / name
    assert produced.exists()
    references = [d / name for d in REFERENCE_DIRS if (d / name).exists()]
    assert references, "no reference CSV found"
    for ref in references:
        diff = max_abs_difference(produced, ref)
        assert diff <= 1e-9, f"{name} differs from {ref} by {diff:.3e}"


def test_evidence_and_ahp_examples_reproduce_the_book_numbers():
    run_script("ch03_evidence_occlusion", "evidence_occlusion.py")
    run_script("ch03_incomplete_ahp", "incomplete_ahp.py")
    with open(EXAMPLES / "ch03_evidence_occlusion" / "output" / "evidence_occlusion.csv", newline="") as fh:
        ev = {r["focal"]: r for r in csv.DictReader(fh)}
    assert float(ev["empty"]["m_smets"]) == pytest.approx(0.44)
    assert float(ev["{O}"]["m_dempster"]) == pytest.approx(0.3616, abs=5e-5)
    assert float(ev["Omega"]["m_dempster"]) == pytest.approx(0.0893, abs=5e-5)
    with open(EXAMPLES / "ch03_incomplete_ahp" / "output" / "incomplete_ahp.csv", newline="") as fh:
        pr = {r["code"]: float(r["priority"]) for r in csv.DictReader(fh)}
    assert [pr[c] for c in "CLRM"] == pytest.approx([0.091, 0.183, 0.441, 0.284], abs=1e-3)
