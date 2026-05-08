"""Tests for response.py: identity for B, time-average identity."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from response import (
    build_quadratic_form,
    frequency_response,
    functional_over_time,
    steady_state_trajectory,
)


def _random_stable_W(n, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n)) * 0.1
    W = 0.5 * (W + W.T)
    # Shrink to ensure rho < 1
    eigs = np.linalg.eigvalsh(W)
    s = max(np.abs(eigs).max(), 1e-12)
    W = 0.8 * W / s
    return W


def test_B_satisfies_defining_identity():
    n = 6
    W = _random_stable_W(n, seed=0)
    D = np.diag(np.linspace(0.1, 0.5, n))
    omega = 0.7
    B = frequency_response(W, D, omega)
    M = np.exp(1j * omega) * np.eye(n) - W
    assert np.allclose(M @ B, D, atol=1e-10)


def test_time_average_equals_quadratic_form_norm():
    n = 5
    W = _random_stable_W(n, seed=1)
    D = np.diag(np.linspace(0.05, 0.3, n))
    omega = 0.4
    B = frequency_response(W, D, omega)
    rng = np.random.default_rng(2)
    a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    # Time-average over a full period via large T fine grid
    period_samples = int(round(2 * np.pi / omega))
    T = 200 * period_samples  # many periods, fine sampling
    X = steady_state_trajectory(B, a, omega, T)
    V = functional_over_time(X, "norm")
    J_TA_empirical = V.mean()
    Q = build_quadratic_form(B, "norm", n)
    J_TA_quad = float(np.real(np.vdot(a, Q @ a)))
    assert np.isclose(J_TA_empirical, J_TA_quad, rtol=1e-3, atol=1e-6)


def test_time_average_equals_quadratic_form_variance():
    n = 5
    W = _random_stable_W(n, seed=3)
    D = np.diag(np.linspace(0.05, 0.3, n))
    omega = 0.55
    B = frequency_response(W, D, omega)
    rng = np.random.default_rng(4)
    a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    period_samples = int(round(2 * np.pi / omega))
    T = 200 * period_samples
    X = steady_state_trajectory(B, a, omega, T)
    V = functional_over_time(X, "variance")
    J_TA_empirical = V.mean()
    Q = build_quadratic_form(B, "variance", n)
    J_TA_quad = float(np.real(np.vdot(a, Q @ a)))
    assert np.isclose(J_TA_empirical, J_TA_quad, rtol=1e-3, atol=1e-6)


def test_time_average_equals_quadratic_form_direction():
    n = 6
    W = _random_stable_W(n, seed=11)
    D = np.diag(np.linspace(0.05, 0.3, n))
    omega = 0.6
    B = frequency_response(W, D, omega)
    rng = np.random.default_rng(12)
    a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = rng.standard_normal(n)
    period_samples = int(round(2 * np.pi / omega))
    T = 200 * period_samples
    X = steady_state_trajectory(B, a, omega, T)
    V = functional_over_time(X, "direction", v=v)
    J_emp = V.mean()
    Q = build_quadratic_form(B, "direction", n, v=v)
    J_quad = float(np.real(np.vdot(a, Q @ a)))
    assert np.isclose(J_emp, J_quad, rtol=1e-3, atol=1e-6)


def test_direction_quadratic_form_is_rank_one():
    n = 5
    W = _random_stable_W(n, seed=13)
    D = np.diag(np.linspace(0.05, 0.3, n))
    B = frequency_response(W, D, 0.4)
    v = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    Q = build_quadratic_form(B, "direction", n, v=v)
    svals = np.linalg.svd(Q, compute_uv=False)
    # Exactly one singular value should be non-negligible
    assert svals[0] > 1e-6
    assert svals[1] < 1e-9


def test_omega_zero_recovers_static_fixed_point():
    n = 5
    W = _random_stable_W(n, seed=5)
    D = np.diag(np.linspace(0.05, 0.3, n))
    B0 = frequency_response(W, D, 0.0)
    expected = np.linalg.solve(np.eye(n) - W, D)
    assert np.allclose(B0, expected, atol=1e-10)
