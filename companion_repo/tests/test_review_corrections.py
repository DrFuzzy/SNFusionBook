"""Regression checks for defects found in the November 2026 independent review."""
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np
from snfusion.scenarios import initial_estimate, SIG_R, SIG_TH

REPO = Path(__file__).resolve().parents[1]


def test_radar_initiation_keeps_the_bearing_uncertainty():
    # At 60 m on the x-axis, bearing noise has lateral variance (60 sigma_theta)^2.
    x, P = initial_estimate(np.array([60., 0., -5.]))
    np.testing.assert_allclose(x[:2], [60., 0.])
    np.testing.assert_allclose(P[:2, :2], np.diag([SIG_R**2, (60*SIG_TH)**2]))
    assert P[1, 1] > 3 * P[0, 0]


def test_initiating_radar_detection_is_not_fused_twice():
    import csv
    path = REPO / 'examples/ch03_ekf_radar_cutin/output/ekf_radar_cutin.csv'
    with path.open() as stream:
        row = next(csv.DictReader(stream))
    assert row['px_est'] == row['px_meas']
    assert row['py_est'] == row['py_meas']
    assert float(row['nis']) == 0.0  # documented initiation placeholder
    assert float(row['sig_py']) > 0.5


def test_cable_noise_uses_the_unique_symmetric_covariance_root():
    # Extract the actual utility without running the 1500-step example at import.
    for path, name in [(REPO/'examples/ch07_cable_das_kf/cable_das_kf.py', 'psd_sqrt'),
                       (REPO/'exercises/ch07/solutions/prob07_cable_sweep.py', 'sq')]:
        tree = ast.parse(path.read_text())
        fun = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        ns = {'np': np}
        exec(compile(ast.Module(body=[fun], type_ignores=[]), str(path), 'exec'), ns)
        # A nondiagonal, rank-deficient covariance catches a basis-factor return.
        M = np.array([[2., 1., 1.], [1., 1., 0.], [1., 0., 1.]])
        B = ns[name](M)
        np.testing.assert_allclose(B, B.T, atol=1e-12)
        np.testing.assert_allclose(B @ B, M, atol=1e-12)
        assert np.linalg.eigvalsh(B).min() > -1e-12


def test_notebook_bootstrap_supports_file_and_cli_scripts(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('py2nb', REPO/'tools/py2nb.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    script = tmp_path/'demo.py'
    script.write_text('import argparse\nfrom pathlib import Path\n'
                      'argparse.ArgumentParser().parse_args()\n'
                      'result = Path(__file__).name\n')
    out = Path(mod.convert(str(script)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['ipykernel_launcher', '-f', 'connection.json'])
    scope = {'__name__': '__main__'}
    for cell in json.loads(out.read_text())['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(cell['source'], str(out), 'exec'), scope)
    assert scope['result'] == 'demo.py'
