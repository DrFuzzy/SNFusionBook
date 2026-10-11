"""Unit tests for snfusion.consensus (track-level fusion, CI, linear consensus)."""

import numpy as np
import pytest

from snfusion.consensus import (correlated_fusion, covariance_intersection, information_fusion,
                                is_doubly_stochastic, metropolis_weights, run_consensus, step_size_weights)


def _spd(rng, n=2, scale=1.0):
    A = rng.normal(size=(n, n))
    return scale * (A @ A.T + n * np.eye(n))


def _ring(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    return A


@pytest.mark.parametrize("weights", [metropolis_weights, step_size_weights])
def test_consensus_converges_to_the_average(weights):
    A = _ring(6)
    A[0, 3] = A[3, 0] = 1.0                      # a chord to make the graph non-bipartite
    W = weights(A)
    assert is_doubly_stochastic(W)
    x0 = np.arange(6, dtype=float) ** 2
    x = run_consensus(x0, W, 400)
    np.testing.assert_allclose(x, np.full(6, x0.mean()), atol=1e-8)
    xm, hist = run_consensus(np.c_[x0, -x0], W, 400, return_history=True)
    assert hist.shape == (401, 6, 2)
    np.testing.assert_allclose(xm[:, 1], -x0.mean(), atol=1e-8)


def test_information_fusion_matches_weighted_least_squares():
    x1, x2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    P1, P2 = np.diag([1.0, 4.0]), np.diag([4.0, 1.0])
    xf, Pf = information_fusion(x1, P1, x2, P2)
    np.testing.assert_allclose(Pf, np.diag([0.8, 0.8]))
    np.testing.assert_allclose(xf, [0.8, 0.8])
    xc, Pc = correlated_fusion(x1, P1, x2, P2, np.zeros((2, 2)))
    np.testing.assert_allclose(xc, xf)
    np.testing.assert_allclose(Pc, Pf)


def test_covariance_intersection_is_consistent_for_any_cross_covariance():
    rng = np.random.default_rng(5)
    for _ in range(10):
        P1, P2 = _spd(rng), _spd(rng, scale=0.7)
        x1, x2 = rng.normal(size=2), rng.normal(size=2)
        for criterion in ("trace", "det", "logdet"):
            x, P, w = covariance_intersection(x1, P1, x2, P2, criterion=criterion)
            assert 0.0 <= w <= 1.0
            I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
            np.testing.assert_allclose(np.linalg.inv(P), w * I1 + (1 - w) * I2)
            # consistency: P_CI dominates the true error covariance for ANY admissible P12
            A1, A2 = P @ (w * I1), P @ ((1 - w) * I2)
            for rho in (-0.9, 0.0, 0.5, 0.95):
                L1, L2 = np.linalg.cholesky(P1), np.linalg.cholesky(P2)
                P12 = rho * L1 @ L2.T
                P_true = A1 @ P1 @ A1.T + A2 @ P2 @ A2.T + A1 @ P12 @ A2.T + A2 @ P12.T @ A1.T
                assert np.linalg.eigvalsh(P - P_true).min() >= -1e-9
            # the naive fusion is tighter (and only valid when the tracks are independent)
            _, Pn = information_fusion(x1, P1, x2, P2)
            assert np.linalg.eigvalsh(P - Pn).min() >= -1e-9


def test_covariance_intersection_special_cases():
    P = np.diag([1.0, 2.0])
    x, Pci, w = covariance_intersection(np.zeros(2), P, np.ones(2), P)
    np.testing.assert_allclose(Pci, P)                   # identical covariances: no gain
    x1, P1 = np.array([1.0, 1.0]), np.eye(2)
    x, Pci, w = covariance_intersection(x1, P1, np.zeros(2), 100 * np.eye(2))
    assert w == pytest.approx(1.0, abs=1e-6)             # a much worse second track is ignored
    np.testing.assert_allclose(x, x1, atol=1e-5)
    x, Pci, w = covariance_intersection(x1, P1, np.zeros(2), np.eye(2), omega=0.5)
    np.testing.assert_allclose(x, [0.5, 0.5])
    np.testing.assert_allclose(Pci, np.eye(2))
