"""Unit tests for snfusion.zonotope (Chapter 3, Set-Membership and Zonotopic Estimation)."""

import numpy as np
import pytest

from snfusion.zonotope import Zonotope


def test_book_single_strip_example():
    # Example ex:zonotope-strip: p = 0, H = [[2, .7, -.5], [.7, 1.4, .8]], c = (0, 1), y = 1, phi = 0.6, gain = c
    Z = Zonotope(np.zeros(2), np.array([[2.0, 0.7, -0.5], [0.7, 1.4, 0.8]]))
    lo, hi = Z.interval_hull()
    np.testing.assert_allclose([lo[1], hi[1]], [-2.9, 2.9])
    Zc = Z.strip_intersection(np.array([0.0, 1.0]), 1.0, 0.6, gain=np.array([0.0, 1.0]))
    np.testing.assert_allclose(Zc.center, [0.0, 1.0])
    np.testing.assert_allclose(Zc.generators, [[2.0, 0.7, -0.5, 0.0], [0.0, 0.0, 0.0, 0.6]], atol=1e-15)
    lo, hi = Zc.interval_hull()
    np.testing.assert_allclose([lo[1], hi[1]], [0.4, 1.6])      # the strip 0.4 <= x2 <= 1.6: tightened
    np.testing.assert_allclose([lo[0], hi[0]], [-3.2, 3.2])     # full width of the prediction kept


def test_optimal_gain_tightens_and_keeps_containment():
    rng = np.random.default_rng(1)
    Z = Zonotope(rng.normal(size=3), rng.normal(size=(3, 6)))
    x_true = Z.center + Z.generators @ rng.uniform(-1, 1, 6)     # a point inside the zonotope
    assert Z.contains_exact(x_true)
    for _ in range(5):
        c = rng.normal(size=3)
        phi = 0.3
        y = c @ x_true + rng.uniform(-phi, phi)
        Zc = Z.strip_intersection(c, y, phi)                      # Frobenius-optimal gain
        assert Zc.radius_frobenius() <= Z.radius_frobenius() + 1e-12
        assert Zc.contains_exact(x_true)
        assert Zc.contains(x_true)
        # support width along the strip normal: a convex combination of the old
        # width and phi with the optimal gain, hence never above max(old, phi)
        w_before = np.abs(Z.generators.T @ c).sum()
        w_after = np.abs(Zc.generators.T @ c).sum()
        assert w_after <= max(w_before, phi) + 1e-12
        Z = Zc


def test_optimal_gain_closed_form():
    # lambda* = H H^T c / (c^T H H^T c + phi^2), eq. (zono-optimal-gain)
    H = np.array([[1.0, 0.5], [0.0, 2.0]])
    Z = Zonotope(np.zeros(2), H)
    c, y, phi = np.array([1.0, 1.0]), 0.5, 0.2
    lam = H @ H.T @ c / (c @ H @ H.T @ c + phi ** 2)
    Zc = Z.strip_intersection(c, y, phi)
    np.testing.assert_allclose(Zc.center, lam * y)
    np.testing.assert_allclose(Zc.generators[:, -1], phi * lam)


def test_order_reduction_preserves_containment():
    rng = np.random.default_rng(2)
    Z = Zonotope(np.zeros(3), rng.normal(size=(3, 12)))
    Zr = Z.reduce(6)
    assert Zr.order == 6
    assert Z.reduce(12) is Z
    for _ in range(50):
        x = Z.center + Z.generators @ rng.uniform(-1, 1, 12)
        assert Zr.contains_exact(x)
    lo, hi = Z.interval_hull()
    lor, hir = Zr.interval_hull()
    assert np.all(lor <= lo + 1e-12) and np.all(hir >= hi - 1e-12)
    with pytest.raises(ValueError):
        Z.reduce(2)


def test_minkowski_sum_and_linear_map():
    A = Zonotope([1.0, 0.0], [[1.0, 0.0], [0.0, 2.0]])
    B = Zonotope([0.0, 1.0], [[0.5], [0.5]])
    S = A + B
    assert S.order == 3
    np.testing.assert_allclose(S.center, [1.0, 1.0])
    np.testing.assert_allclose(S.half_widths(), A.half_widths() + B.half_widths())
    M = np.array([[2.0, 0.0], [0.0, -1.0]])
    L = A.linear_map(M, b=np.array([0.0, 3.0]))
    np.testing.assert_allclose(L.center, [2.0, 3.0])
    np.testing.assert_allclose(L.half_widths(), [2.0, 2.0])
    np.testing.assert_allclose((M @ A).generators, L.generators)


def test_predict_appends_process_generators():
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    Z = Zonotope.from_box([0.0, 0.0], [1.0, 1.0])
    Gq = np.array([[0.005], [0.1]])
    Zp = Z.predict(F, Gq)
    assert Zp.order == 3
    np.testing.assert_allclose(Zp.generators, np.hstack([F @ Z.generators, Gq]))


def test_vertices_2d():
    Z = Zonotope.from_box([0.0, 0.0], [1.0, 2.0])
    V = Z.vertices_2d()
    assert V.shape == (4, 2)
    assert set(map(tuple, np.round(V, 12))) == {(1, 2), (-1, 2), (-1, -2), (1, -2)}
