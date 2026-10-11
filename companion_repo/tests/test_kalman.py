"""Unit tests for snfusion.kalman (Chapter 3, Bayesian Filtering)."""

import numpy as np
import pytest

from snfusion.kalman import ExtendedKalmanFilter, KalmanFilter, UnscentedKalmanFilter, wrap_angle


def test_kf_matches_hand_computation_1d():
    # x ~ N(0, 1), static model, measure x directly with R = 1, observe z = 2:
    # K = 1/2, x+ = 1, P+ = 1/2, NIS = 2^2 / (1 + 1) = 2
    kf = KalmanFilter(F=np.eye(1), H=np.eye(1), Q=np.zeros((1, 1)), R=np.eye(1))
    x, P = kf.predict(np.zeros(1), np.eye(1))
    upd = kf.update(x, P, np.array([2.0]))
    assert upd.x[0] == pytest.approx(1.0)
    assert upd.P[0, 0] == pytest.approx(0.5)
    assert upd.K[0, 0] == pytest.approx(0.5)
    assert upd.nis == pytest.approx(2.0)


def test_kf_matches_hand_computation_2d():
    # constant-velocity 1-D model, position measurement, hand-derived numbers
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    kf = KalmanFilter(F=F, H=H, Q=np.zeros((2, 2)), R=np.eye(1))
    x, P = kf.predict(np.zeros(2), np.eye(2))
    np.testing.assert_allclose(P, [[2.0, 1.0], [1.0, 1.0]])
    upd = kf.update(x, P, np.array([1.0]))
    np.testing.assert_allclose(upd.S, [[3.0]])
    np.testing.assert_allclose(upd.K, [[2.0 / 3.0], [1.0 / 3.0]])
    np.testing.assert_allclose(upd.x, [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(upd.P, [[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]], atol=1e-14)
    assert upd.nis == pytest.approx(1.0 / 3.0)


def test_joseph_and_standard_covariance_updates_agree():
    F, H = np.eye(2), np.array([[1.0, 0.0]])
    kf_j = KalmanFilter(F, H, 0.1 * np.eye(2), np.eye(1), joseph=True)
    kf_s = KalmanFilter(F, H, 0.1 * np.eye(2), np.eye(1), joseph=False)
    P = np.array([[2.0, 0.5], [0.5, 1.0]])
    a, b = kf_j.update(np.zeros(2), P, np.array([1.0])), kf_s.update(np.zeros(2), P, np.array([1.0]))
    np.testing.assert_allclose(a.P, b.P, atol=1e-12)
    np.testing.assert_allclose(a.x, b.x)


def _linear_system(seed=0):
    rng = np.random.default_rng(seed)
    F = np.array([[1.0, 0.1, 0.0], [0.0, 0.95, 0.1], [0.0, 0.0, 0.9]])
    H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Q = 0.01 * np.eye(3)
    R = 0.1 * np.eye(2)
    x = rng.normal(size=3)
    zs = []
    for _ in range(30):
        x = F @ x + rng.multivariate_normal(np.zeros(3), Q)
        zs.append(H @ x + rng.multivariate_normal(np.zeros(2), R))
    return F, H, Q, R, zs


def test_ekf_and_ukf_agree_with_kf_on_linear_model():
    F, H, Q, R, zs = _linear_system()
    kf = KalmanFilter(F, H, Q, R)
    ekf = ExtendedKalmanFilter(f=lambda x, u: F @ x, F_jac=lambda x, u: F,
                               h=lambda x: H @ x, H_jac=lambda x: H, Q=Q, R=R)
    ukf = UnscentedKalmanFilter(f=lambda x, u: F @ x, h=lambda x: H @ x, Q=Q, R=R)
    x_k = x_e = x_u = np.zeros(3)
    P_k = P_e = P_u = np.eye(3)
    for z in zs:
        x_k, P_k = kf.predict(x_k, P_k)
        x_e, P_e = ekf.predict(x_e, P_e)
        x_u, P_u = ukf.predict(x_u, P_u)
        np.testing.assert_allclose(x_u, x_k, atol=1e-10)
        np.testing.assert_allclose(P_u, P_k, atol=1e-10)
        uk, ue, uu = kf.update(x_k, P_k, z), ekf.update(x_e, P_e, z), ukf.update(x_u, P_u, z)
        x_k, P_k, x_e, P_e, x_u, P_u = uk.x, uk.P, ue.x, ue.P, uu.x, uu.P
        np.testing.assert_allclose(x_e, x_k, atol=1e-12)
        np.testing.assert_allclose(P_e, P_k, atol=1e-12)
        np.testing.assert_allclose(x_u, x_k, atol=1e-9)
        np.testing.assert_allclose(P_u, P_k, atol=1e-9)
        assert uu.nis == pytest.approx(uk.nis, abs=1e-8)


def test_ukf_sigma_point_weights():
    ukf = UnscentedKalmanFilter(None, lambda x: x, np.eye(4), np.eye(3), alpha=1.0, beta=2.0, kappa=1.0)
    lam, wm, wc = ukf.weights(4)
    assert lam == pytest.approx(1.0)
    assert wm.sum() == pytest.approx(1.0)
    assert np.all(wm >= 0) and np.all(wc >= 0)
    assert wm[0] == pytest.approx(0.2) and wm[1] == pytest.approx(0.1)
    assert wc[0] == pytest.approx(0.2 + 2.0)
    X = ukf.sigma_points(np.zeros(4), np.eye(4))
    assert X.shape == (9, 4)
    np.testing.assert_allclose(wm @ X, np.zeros(4), atol=1e-15)
    np.testing.assert_allclose(sum(wc[i] * np.outer(X[i], X[i]) for i in range(9)), np.eye(4), atol=1e-12)


def test_ukf_angle_channel_is_averaged_on_the_circle():
    # measurement = bearing of a 2-D position; near +-pi the linear average is wrong
    h = lambda x: np.array([np.arctan2(x[1], x[0])])
    ukf = UnscentedKalmanFilter(None, h, np.eye(2), np.array([[1e-4]]), angle_idx=(0,), linear_predict=np.eye(2))
    x, P = np.array([-10.0, 0.0]), np.diag([0.01, 1.0])
    upd = ukf.update(x, P, np.array([np.pi - 0.01]))
    assert abs(upd.innovation[0]) < 0.5          # wrapped innovation, no 2 pi jump
    assert upd.nis < 50


def test_wrap_angle():
    np.testing.assert_allclose(wrap_angle(np.array([0.0, np.pi / 2, 3 * np.pi, -3 * np.pi / 2])),
                               [0.0, np.pi / 2, -np.pi, np.pi / 2], atol=1e-12)
    assert -np.pi <= wrap_angle(123.456) < np.pi
