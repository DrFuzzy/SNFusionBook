"""Unit tests for snfusion.metrics and snfusion.models."""

import numpy as np
import pytest

from snfusion import metrics, models, scenarios


def test_chi2_bounds_quoted_in_the_book():
    lo, hi = metrics.chi2_bounds(3, 0.95)                 # band of Fig. fig:ekf-radar-cutin
    assert lo == pytest.approx(0.216, abs=1e-3)
    assert hi == pytest.approx(9.348, abs=1e-3)
    _, one_sided = metrics.chi2_bounds(3, 0.95, two_sided=False)
    assert one_sided == pytest.approx(7.815, abs=1e-3)
    lo_n, hi_n = metrics.average_nis_bounds(3, 100)
    assert lo_n < 3.0 < hi_n


def test_rmse_nis_nees():
    est, tru = np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 5.0])
    assert metrics.rmse(est, tru) == pytest.approx(np.sqrt(5.0 / 3.0))
    S = np.array([[2.0, 0.0], [0.0, 0.5]])
    assert metrics.nis(np.array([1.0, 1.0]), S) == pytest.approx(0.5 + 2.0)
    assert metrics.nees(np.array([2.0, 0.0]), np.diag([4.0, 1.0])) == pytest.approx(1.0)
    assert metrics.fraction_inside(np.array([0.1, 1.0, 5.0, 20.0]), (0.216, 9.348)) == pytest.approx(0.5)
    xy = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert metrics.position_rmse(xy, np.zeros((2, 2))) == pytest.approx(np.sqrt(25.0 / 2.0))


def test_cv_model_matches_the_book_matrices():
    dt, sa = 0.05, 0.8
    F = models.constant_velocity(dt)
    Q = models.white_noise_acceleration_Q(dt, sa)
    np.testing.assert_array_equal(F, [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    t2, t3, t4 = dt ** 2, dt ** 3, dt ** 4
    Q_book = sa ** 2 * np.array([[t4 / 4, 0, t3 / 2, 0], [0, t4 / 4, 0, t3 / 2],
                                 [t3 / 2, 0, t2, 0], [0, t3 / 2, 0, t2]])
    np.testing.assert_array_equal(Q, Q_book)
    np.testing.assert_array_equal(scenarios.F_matrix(dt), F)
    np.testing.assert_array_equal(scenarios.Q_matrix(dt, sa), Q)


def test_other_models():
    assert models.constant_acceleration(0.1).shape == (6, 6)
    Qj = models.white_noise_jerk_Q(0.1, 1.0)
    assert np.linalg.eigvalsh(Qj).min() >= -1e-15
    np.testing.assert_allclose(models.coordinated_turn(0.1, 0.0), models.constant_velocity(0.1))
    F = models.coordinated_turn(0.5, 0.3)
    v0 = np.array([0.0, 0.0, 1.0, 0.0])
    v1 = F @ v0
    assert np.hypot(v1[2], v1[3]) == pytest.approx(1.0)        # a turn preserves speed
    x5 = models.coordinated_turn_augmented(0.5, np.array([0.0, 0.0, 1.0, 0.0, 0.3]))
    np.testing.assert_allclose(x5[:4], v1)
    assert models.coordinated_turn_Q(0.1, 1.0, 0.1).shape == (5, 5)
    np.testing.assert_array_equal(models.position_measurement(2, 4), [[1, 0, 0, 0], [0, 1, 0, 0]])


def test_scenario_measurement_model_and_jacobian():
    x = np.array([40.0, 3.0, -5.0, 0.4])
    z = scenarios.h(x)
    assert z[0] == pytest.approx(np.hypot(40.0, 3.0))
    H = scenarios.H_jacobian(x)
    eps = 1e-6
    H_num = np.array([(scenarios.h(x + eps * e) - scenarios.h(x - eps * e)) / (2 * eps) for e in np.eye(4)]).T
    np.testing.assert_allclose(H, H_num, atol=1e-6)
    ts, Xt, Z, err = scenarios.nominal_run()
    assert Xt.shape == (201, 4) and Z.shape == (201, 3)
    assert np.all(np.abs(Z[:, 1]) <= np.pi)
    np.testing.assert_allclose(scenarios.truth(5.0)[1], 3.5)
    np.testing.assert_allclose(scenarios.truth(8.0)[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(scenarios.truth(6.5)[1], 1.75)
