"""Unit tests for snfusion.ahp (Chapter 3, Multi-Criteria Decision-Making)."""

import numpy as np
import pytest

from snfusion.ahp import (comparison_matrix, completed_matrix, consistency_ratio, geometric_mean_priorities,
                          incomplete_lls, principal_eigenvector)


def test_lls_recovers_a_consistent_matrix_from_a_sparse_graph():
    rng = np.random.default_rng(3)
    w = rng.uniform(0.5, 5.0, size=6)
    w /= w.sum()
    labels = list("ABCDEF")
    # a spanning tree plus two extra edges, all consistent with w
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (1, 5)]
    comps = [(labels[i], labels[j], w[i] / w[j]) for i, j in edges]
    res = incomplete_lls(labels, comps)
    np.testing.assert_allclose(res.priorities, w, atol=1e-12)
    np.testing.assert_allclose(res.residuals, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.completed_matrix(), np.outer(w, 1.0 / w), atol=1e-12)
    assert res.log_priorities.sum() == pytest.approx(0.0, abs=1e-12)        # gauge sum y = 0
    res_first = incomplete_lls(labels, comps, gauge="first")
    assert res_first.log_priorities[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(res_first.priorities, res.priorities, atol=1e-12)


def test_book_rain_night_example():
    comps = [("L", "C", 2.0), ("R", "C", 5.0), ("M", "C", 3.0), ("R", "M", 1.5)]
    res = incomplete_lls(["C", "L", "R", "M"], comps)
    np.testing.assert_allclose(res.priorities, [0.091, 0.183, 0.441, 0.284], atol=1e-3)
    assert res.fitted_ratio("R", "L") == pytest.approx(2.41, abs=0.01)
    assert res.fitted_ratio("M", "L") == pytest.approx(1.55, abs=0.01)
    assert res.fitted_ratio("L", "C") == pytest.approx(2.0)         # L-C is a pendant edge: fitted exactly
    assert list(np.argsort(-res.priorities)) == [2, 3, 1, 0]         # R > M > L > C
    assert consistency_ratio(res.completed_matrix()) == pytest.approx(0.0, abs=1e-10)


def test_principal_eigenvector_and_consistency_ratio():
    w = np.array([0.5, 0.3, 0.2])
    A = np.outer(w, 1.0 / w)
    p, lam = principal_eigenvector(A)
    np.testing.assert_allclose(p, w, atol=1e-12)
    assert lam == pytest.approx(3.0)
    assert consistency_ratio(A) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(geometric_mean_priorities(A), w, atol=1e-12)
    # Saaty's classic mildly inconsistent 3 x 3 matrix
    B = np.array([[1.0, 3.0, 5.0], [1 / 3, 1.0, 3.0], [1 / 5, 1 / 3, 1.0]])
    cr = consistency_ratio(B)
    assert 0.0 < cr < 0.1
    assert cr == pytest.approx(0.0332, abs=2e-3)


def test_comparison_matrix_and_disconnected_graph():
    A = comparison_matrix(["a", "b", "c"], [("a", "b", 2.0)])
    assert A[0, 1] == 2.0 and A[1, 0] == 0.5 and np.isnan(A[0, 2])
    with pytest.raises(ValueError):
        incomplete_lls(["a", "b", "c", "d"], [("a", "b", 2.0), ("c", "d", 3.0)])
    with pytest.raises(ValueError):
        incomplete_lls(["a", "b"], [("a", "b", -1.0)])
    np.testing.assert_allclose(completed_matrix([0.0, np.log(2.0)]), [[1.0, 0.5], [2.0, 1.0]])
